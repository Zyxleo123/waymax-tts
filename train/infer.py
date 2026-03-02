from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion_policy import DiffusionPolicy
from train.checkpoints import restore_checkpoint
from train.postprocess import postprocess_predictions
from train.preprocess import preprocess_simulator_state
from train.types import PreprocessConfig


@dataclass
class InferenceBundle:
    graphdef: Any
    nonparam_state: Any
    params_state: Any
    preprocess_cfg: PreprocessConfig
    model_config: dict[str, Any]
    rng_key: jax.Array
    checkpoint_path: str


@dataclass
class PredictionBatch:
    start_t_b: jax.Array
    trajectories_world_bkt5: jax.Array
    world_t_seconds_bk: jax.Array
    world_t_valid_bk: jax.Array
    aux: dict[str, jax.Array]


@dataclass
class RolloutMetricsBatch:
    metric_timeseries: dict[str, jax.Array]
    metric_valid: dict[str, jax.Array]
    metric_names: tuple[str, ...]
    rollout_num_steps: int
    batch_size: int
    num_samples: int


_REQUIRED_META_KEYS = (
    "target_dim",
    "hidden_dim",
    "cond_dim",
    "map_attr_dim",
    "tl_attr_dim",
    "predict_horizon",
    "predict_type",
    "model_dt",
    "max_range",
    "ego_range",
    "max_velocity",
    "max_width",
    "max_map_points",
    "max_tl_points",
    "num_map_type_classes",
)


_DEFAULT_METRIC_NAMES = ("overlap", "offroad", "sdc_progression")


def _load_checkpoint_metadata(
    checkpoint_path: str, metadata_path: str | None
) -> dict[str, Any]:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    meta = Path(metadata_path) if metadata_path is not None else ckpt / "metadata.json"
    if not meta.exists():
        raise ValueError(f"Metadata file not found: {meta}")

    with meta.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing = [k for k in _REQUIRED_META_KEYS if k not in metadata]
    if missing:
        raise KeyError(f"Missing required metadata keys: {missing}")

    return metadata


def _build_preprocess_cfg(metadata: dict[str, Any]) -> PreprocessConfig:
    return PreprocessConfig(
        model_dt=float(metadata["model_dt"]),
        world_dt_fallback=0.1,
        max_range=float(metadata["max_range"]),
        ego_range=float(metadata["ego_range"]),
        max_velocity=float(metadata["max_velocity"]),
        max_width=float(metadata["max_width"]),
        max_map_points=int(metadata["max_map_points"]),
        max_tl_points=int(metadata["max_tl_points"]),
        num_map_type_classes=int(metadata["num_map_type_classes"]),
        predict_horizon=int(metadata["predict_horizon"]),
        map_unknown_type_index=20,
    )


def _build_model_from_metadata(metadata: dict[str, Any], seed: int) -> DiffusionPolicy:
    return DiffusionPolicy(
        target_dim=int(metadata["target_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        cond_dim=int(metadata["cond_dim"]),
        map_attr_dim=int(metadata["map_attr_dim"]),
        tl_attr_dim=int(metadata["tl_attr_dim"]),
        predict_horizon=int(metadata["predict_horizon"]),
        predict_type=str(metadata["predict_type"]),
        rngs=nnx.Rngs(int(seed)),
    )


def _select_sample_indices(total_k: int, sample_indices: list[int] | None) -> np.ndarray:
    if sample_indices is None:
        return np.arange(total_k, dtype=np.int32)
    if len(sample_indices) == 0:
        raise ValueError("sample_indices must be non-empty when provided.")
    selected = np.asarray(sample_indices, dtype=np.int32)
    if np.any(selected < 0) or np.any(selected >= total_k):
        raise IndexError(
            f"sample_indices out of bounds for K={total_k}: {selected.tolist()}."
        )
    return selected


def _validate_metric_names(metric_names: tuple[str, ...]) -> tuple[str, ...]:
    if not metric_names:
        raise ValueError("metric_names must be non-empty.")
    from waymax.metrics import metric_factory

    available = set(metric_factory._METRICS_REGISTRY.keys())  # pylint: disable=protected-access
    unknown = [m for m in metric_names if m not in available]
    if unknown:
        raise ValueError(
            f"Unknown metric names: {unknown}. Available metrics: {sorted(available)}"
        )
    return tuple(metric_names)


def _expand_sim_state_for_samples(sim_state, k_sel: int):
    """Repeats each batch item k_sel times along the leading batch axis."""
    if sim_state.log_trajectory.x.ndim < 3:
        raise ValueError("Expected batched SimulatorState with shape [B, N, T].")

    batch_size = int(sim_state.log_trajectory.x.shape[0])

    def _repeat_leaf(x):
        if not hasattr(x, "ndim") or x.ndim == 0:
            return x
        return jnp.repeat(x, repeats=k_sel, axis=0)

    expanded = jax.tree_util.tree_map(_repeat_leaf, sim_state)
    return expanded, batch_size


def _get_sdc_indices_for_batched_state(state_batched) -> np.ndarray:
    """Returns one SDC index per batch item; errors if not exactly one."""
    is_sdc = np.asarray(state_batched.object_metadata.is_sdc).astype(bool)
    if is_sdc.ndim != 2:
        raise ValueError(f"Expected is_sdc shape [B,N], got {is_sdc.shape}.")
    sdc_count = np.sum(is_sdc, axis=1)
    if np.any(sdc_count != 1):
        bad = np.where(sdc_count != 1)[0].tolist()
        raise ValueError(f"Expected exactly one SDC per batch item; bad indices: {bad}.")
    return np.argmax(is_sdc.astype(np.int32), axis=1)


def _apply_ego_replacements_to_expanded_state(
    expanded_state,
    *,
    start_t_bk: np.ndarray,
    traj_bkl5: np.ndarray,
):
    """Applies ego trajectory replacements to expanded batched state."""
    traj = expanded_state.log_trajectory
    x = np.array(traj.x, copy=True)
    y = np.array(traj.y, copy=True)
    yaw = np.array(traj.yaw, copy=True)
    vel_x = np.array(traj.vel_x, copy=True)
    vel_y = np.array(traj.vel_y, copy=True)
    valid = np.array(traj.valid, copy=True)

    bksz, num_objects, num_steps = x.shape
    if traj_bkl5.shape[0] != bksz:
        raise ValueError(
            f"Replacement trajectory batch mismatch: expected {bksz}, got {traj_bkl5.shape[0]}."
        )
    if traj_bkl5.ndim != 3 or traj_bkl5.shape[-1] != 5:
        raise ValueError(
            f"Replacement trajectories must have shape [BK,L,5], got {traj_bkl5.shape}."
        )
    if start_t_bk.shape[0] != bksz:
        raise ValueError(
            f"start_t length mismatch: expected {bksz}, got {start_t_bk.shape[0]}."
        )

    sdc_idx = _get_sdc_indices_for_batched_state(expanded_state)
    horizon = int(traj_bkl5.shape[1])

    for bk in range(bksz):
        start_t = int(start_t_bk[bk])
        if start_t < 0:
            raise ValueError(f"start_t must be >=0, got {start_t} at index {bk}.")
        end_t = start_t + horizon
        if end_t > num_steps:
            raise ValueError(
                f"Replacement out of bounds at index {bk}: start={start_t}, len={horizon}, num_steps={num_steps}."
            )
        ego = int(sdc_idx[bk])
        sl = slice(start_t, end_t)
        repl = traj_bkl5[bk]
        x[bk, ego, sl] = repl[:, 0]
        y[bk, ego, sl] = repl[:, 1]
        yaw[bk, ego, sl] = repl[:, 2]
        vel_x[bk, ego, sl] = repl[:, 3]
        vel_y[bk, ego, sl] = repl[:, 4]
        valid[bk, ego, sl] = True

    new_log = traj.replace(
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=valid,
    )
    return dataclasses.replace(expanded_state, log_trajectory=new_log)


def _build_rollout_env_and_actor(
    metric_names: tuple[str, ...], max_num_objects: int
):
    from waymax import agents
    from waymax import config as waymax_config
    from waymax import dynamics as waymax_dynamics
    from waymax import env as waymax_env

    state_dynamics = waymax_dynamics.StateDynamics()
    planning_dynamics = waymax_env.PlanningAgentDynamics(state_dynamics)
    env_cfg = waymax_config.EnvironmentConfig(
        max_num_objects=int(max_num_objects),
        controlled_object=waymax_config.ObjectType.SDC,
        metrics=waymax_config.MetricsConfig(metrics_to_run=metric_names),
    )
    env_obj = waymax_env.PlanningAgentEnvironment(
        dynamics_model=state_dynamics,
        config=env_cfg,
    )
    actor = agents.create_expert_actor(
        dynamics_model=planning_dynamics,
        is_controlled_func=lambda state: state.object_metadata.is_sdc,
    )
    return env_obj, actor


def _run_batched_rollout(replaced_state, env_obj, actor, rng_key, rollout_num_steps: int):
    from waymax import env as waymax_env

    return waymax_env.rollout(
        scenario=replaced_state,
        actor=actor,
        env=env_obj,
        rng=rng_key,
        rollout_num_steps=rollout_num_steps,
    )


def _extract_metric_tensor_bkt(
    metric_result, *, batch_size: int, num_samples: int
) -> tuple[jax.Array, jax.Array]:
    value = metric_result.value
    valid = metric_result.valid

    if value.ndim != 2 or valid.ndim != 2:
        raise ValueError(
            f"Expected metric value/valid with shape [T,BK], got value={value.shape}, valid={valid.shape}."
        )
    t_dim, bk_dim = int(value.shape[0]), int(value.shape[1])
    expected = batch_size * num_samples
    if bk_dim != expected:
        raise ValueError(
            f"Metric batch dim mismatch: expected BK={expected}, got {bk_dim}."
        )

    # [T, BK] -> [BK, T] -> [B, K, T]
    value_bkt = jnp.transpose(value, (1, 0)).reshape(batch_size, num_samples, t_dim)
    valid_bkt = jnp.transpose(valid, (1, 0)).reshape(batch_size, num_samples, t_dim)
    return value_bkt, valid_bkt


def load_model_for_inference(
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    metadata_path: str | None = None,
    seed: int = 0,
    skip_checkpoint_load: bool = False,
) -> InferenceBundle:
    metadata = _load_checkpoint_metadata(checkpoint_path, metadata_path)
    preprocess_cfg = _build_preprocess_cfg(metadata)
    model = _build_model_from_metadata(metadata, seed)

    graphdef, template_params_state, template_nonparam_state = nnx.split(
        model, nnx.Param, ...
    )
    params_state = template_params_state
    nonparam_state = template_nonparam_state

    if not skip_checkpoint_load:
        restored = restore_checkpoint(checkpoint_path)
        if "params_state" not in restored or "ema_params" not in restored:
            raise KeyError("Checkpoint is missing 'params_state' or 'ema_params'.")

        params_state = restored["ema_params"] if use_ema else restored["params_state"]
        nonparam_state = restored.get("nonparam_state", template_nonparam_state)

    model_cfg = {
        "target_dim": int(metadata["target_dim"]),
        "predict_horizon": int(metadata["predict_horizon"]),
        "predict_type": str(metadata["predict_type"]),
    }

    return InferenceBundle(
        graphdef=graphdef,
        nonparam_state=nonparam_state,
        params_state=params_state,
        preprocess_cfg=preprocess_cfg,
        model_config=model_cfg,
        rng_key=jax.random.PRNGKey(int(seed)),
        checkpoint_path=checkpoint_path,
    )


def predict_replacement_trajectories_for_batch(
    sim_state,
    inference: InferenceBundle,
    *,
    rng_key: jax.Array | None = None,
    num_samples: int = 4,
    anchor_step_override_b: jax.Array | np.ndarray | None = None,
) -> PredictionBatch:
    if int(num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if sim_state.log_trajectory.x.ndim < 3:
        raise ValueError(
            "Expected batched SimulatorState with shape [..., N, T] for inference."
        )

    key = inference.rng_key if rng_key is None else rng_key
    anchor_override = None
    if anchor_step_override_b is not None:
        anchor_override = jnp.asarray(anchor_step_override_b, dtype=jnp.int32)
    pre_batch, next_key = preprocess_simulator_state(
        sim_state,
        key,
        inference.preprocess_cfg,
        anchor_step_override_b=anchor_override,
    )

    model = nnx.merge(
        inference.graphdef, inference.params_state, inference.nonparam_state
    )
    features = pre_batch.features
    cond = model._condition_impl(model._as_features(features))

    sample_keys = jax.random.split(next_key, int(num_samples))
    world_samples = []
    for i in range(int(num_samples)):
        pred_norm_btd = model._sample_from_condition_impl(cond, sample_keys[i])
        post = postprocess_predictions(pred_norm_btd, pre_batch.aux, inference.preprocess_cfg)
        traj_world_btd = post["trajectory_world_world_dt"]
        if traj_world_btd.shape[-1] != 5:
            raise ValueError(
                f"Postprocessed trajectory must have last dim 5, got {traj_world_btd.shape}."
            )
        world_samples.append(traj_world_btd)

    trajectories_world_bkt5 = jnp.stack(world_samples, axis=1)
    start_t_b = pre_batch.aux["anchor_step"].astype(jnp.int32)

    return PredictionBatch(
        start_t_b=start_t_b,
        trajectories_world_bkt5=trajectories_world_bkt5,
        world_t_seconds_bk=pre_batch.aux["world_t_seconds"],
        world_t_valid_bk=pre_batch.aux["world_t_valid"],
        aux=pre_batch.aux,
    )


def predict_replacement_trajectories_with_periodic_replan(
    sim_state,
    inference: InferenceBundle,
    *,
    rng_key: jax.Array | None = None,
    num_samples: int = 4,
    replan_interval_steps: int = 10,
) -> PredictionBatch:
    if int(replan_interval_steps) <= 0:
        return predict_replacement_trajectories_for_batch(
            sim_state,
            inference,
            rng_key=rng_key,
            num_samples=num_samples,
        )

    total_k = int(num_samples)
    if total_k < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")

    key = inference.rng_key if rng_key is None else rng_key
    expanded_state, batch_size = _expand_sim_state_for_samples(sim_state, total_k)
    initial_anchor_bk = np.zeros((batch_size * total_k,), dtype=np.int32)

    initial = predict_replacement_trajectories_for_batch(
        expanded_state,
        inference,
        rng_key=key,
        num_samples=1,
        anchor_step_override_b=initial_anchor_bk,
    )

    start_t_bk = np.asarray(initial.start_t_b, dtype=np.int32)
    if not np.all(start_t_bk == 0):
        raise ValueError(
            f"Expected initial anchor_step to be all zeros, got min={int(start_t_bk.min())}, max={int(start_t_bk.max())}."
        )

    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    traj_bk1l5 = np.asarray(initial.trajectories_world_bkt5)
    if traj_bk1l5.ndim != 4 or traj_bk1l5.shape[1] != 1 or traj_bk1l5.shape[-1] != 5:
        raise ValueError(
            f"Expected initial trajectories shape [BK,1,L,5], got {traj_bk1l5.shape}."
        )
    initial_traj_bkl5 = np.array(traj_bk1l5[:, 0, :, :], dtype=np.float32, copy=True)
    model_horizon_len = int(initial_traj_bkl5.shape[1])

    traj_bkl5 = np.zeros((batch_size * total_k, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_bkl5[:, :init_copy_len, :] = initial_traj_bkl5[:, :init_copy_len, :]

    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in range(interval, max_steps + 1, interval):
        replaced_state = _apply_ego_replacements_to_expanded_state(
            expanded_state,
            start_t_bk=start_t_bk,
            traj_bkl5=traj_bkl5,
        )

        forced_anchor_bk = start_t_bk + int(step_offset)
        repl = predict_replacement_trajectories_for_batch(
            replaced_state,
            inference,
            rng_key=key,
            num_samples=1,
            anchor_step_override_b=forced_anchor_bk,
        )

        repl_bk1l5 = np.asarray(repl.trajectories_world_bkt5)
        repl_bkl5 = np.asarray(repl_bk1l5[:, 0, :, :], dtype=np.float32)

        remaining = episode_num_steps - int(step_offset)
        if remaining <= 0:
            continue
        copy_len = min(remaining, int(repl_bkl5.shape[1]))
        traj_bkl5[:, step_offset : step_offset + copy_len, :] = repl_bkl5[:, :copy_len, :]

    traj_bktl5 = traj_bkl5.reshape(batch_size, total_k, episode_num_steps, 5)
    start_t_b = start_t_bk.reshape(batch_size, total_k)[:, 0]

    world_dt_b = np.asarray(initial.aux["world_dt_seconds"], dtype=np.float32).reshape(
        batch_size, total_k
    )[:, 0]
    world_t = (
        np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt_b[:, None]
    )
    world_valid = np.ones((batch_size, episode_num_steps), dtype=bool)

    return PredictionBatch(
        start_t_b=jnp.asarray(start_t_b, dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_bktl5, dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial.aux,
    )


def to_replacement_lists(
    pred: PredictionBatch,
    *,
    sample_index: int = 0,
    trim_anchor: bool = False,
) -> tuple[list[int], list[np.ndarray]]:
    traj_bkt5 = np.asarray(pred.trajectories_world_bkt5)
    if traj_bkt5.ndim != 4 or traj_bkt5.shape[-1] != 5:
        raise ValueError(
            f"Expected trajectories with shape [B,K,L,5], got {traj_bkt5.shape}."
        )

    batch_size, k, _, _ = traj_bkt5.shape
    if not (0 <= int(sample_index) < int(k)):
        raise IndexError(
            f"sample_index {sample_index} out of bounds for K={k}."
        )

    start_t = np.asarray(pred.start_t_b, dtype=np.int32).copy()
    if start_t.shape[0] != batch_size:
        raise ValueError(
            f"start_t length mismatch: expected {batch_size}, got {start_t.shape[0]}."
        )

    ego_start_times: list[int] = []
    ego_trajectories: list[np.ndarray] = []
    for b in range(batch_size):
        traj_l5 = np.asarray(traj_bkt5[b, int(sample_index)], dtype=np.float32)
        start_b = int(start_t[b])
        if trim_anchor:
            if traj_l5.shape[0] <= 1:
                raise ValueError(
                    "Cannot trim anchor from trajectory with <=1 timestep."
                )
            traj_l5 = traj_l5[1:]
            start_b += 1
        ego_start_times.append(start_b)
        ego_trajectories.append(traj_l5)

    return ego_start_times, ego_trajectories


def rollout_predicted_trajectories_with_metrics(
    sim_state,
    pred: PredictionBatch,
    *,
    metric_names: tuple[str, ...] = _DEFAULT_METRIC_NAMES,
    rollout_num_steps: int | None = None,
    sample_indices: list[int] | None = None,
    rng_key: jax.Array | None = None,
) -> RolloutMetricsBatch:
    traj_bkt5 = np.asarray(pred.trajectories_world_bkt5)
    if traj_bkt5.ndim != 4 or traj_bkt5.shape[-1] != 5:
        raise ValueError(
            f"Expected trajectories with shape [B,K,L,5], got {traj_bkt5.shape}."
        )

    batch_size, total_k, length, _ = traj_bkt5.shape
    selected_k = _select_sample_indices(total_k, sample_indices)
    k_sel = int(selected_k.shape[0])
    selected_traj = traj_bkt5[:, selected_k, :, :]  # [B, K_sel, L, 5]

    if rollout_num_steps is None:
        rollout_num_steps = int(length - 1)
    rollout_num_steps = int(rollout_num_steps)
    if rollout_num_steps < 1:
        raise ValueError(f"rollout_num_steps must be >=1, got {rollout_num_steps}.")

    start_t_b = np.asarray(pred.start_t_b, dtype=np.int32)
    if start_t_b.shape[0] != batch_size:
        raise ValueError(
            f"start_t length mismatch: expected {batch_size}, got {start_t_b.shape[0]}."
        )

    metric_names = _validate_metric_names(tuple(metric_names))
    expanded_state, state_batch = _expand_sim_state_for_samples(sim_state, k_sel)
    if state_batch != batch_size:
        raise ValueError(
            f"sim_state batch size {state_batch} does not match prediction batch size {batch_size}."
        )

    start_t_bk = np.repeat(start_t_b, repeats=k_sel, axis=0)  # [BK]
    traj_bkl5 = selected_traj.reshape(batch_size * k_sel, length, 5)
    replaced_state = _apply_ego_replacements_to_expanded_state(
        expanded_state,
        start_t_bk=start_t_bk,
        traj_bkl5=traj_bkl5,
    )

    env_obj, actor = _build_rollout_env_and_actor(
        metric_names, int(expanded_state.log_trajectory.x.shape[1])
    )
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    rollout_out = _run_batched_rollout(
        replaced_state, env_obj, actor, key, rollout_num_steps
    )

    metric_timeseries: dict[str, jax.Array] = {}
    metric_valid: dict[str, jax.Array] = {}
    for metric_name in metric_names:
        if metric_name not in rollout_out.metrics:
            raise ValueError(
                f"Metric '{metric_name}' missing in rollout output; available={list(rollout_out.metrics.keys())}."
            )
        value_bkt, valid_bkt = _extract_metric_tensor_bkt(
            rollout_out.metrics[metric_name],
            batch_size=batch_size,
            num_samples=k_sel,
        )
        metric_timeseries[metric_name] = value_bkt
        metric_valid[metric_name] = valid_bkt

    return RolloutMetricsBatch(
        metric_timeseries=metric_timeseries,
        metric_valid=metric_valid,
        metric_names=metric_names,
        rollout_num_steps=rollout_num_steps,
        batch_size=batch_size,
        num_samples=k_sel,
    )


def predict_and_rollout_batch(
    sim_state,
    inference: InferenceBundle,
    *,
    num_samples: int = 4,
    metric_names: tuple[str, ...] = _DEFAULT_METRIC_NAMES,
    rollout_num_steps: int | None = None,
    sample_indices: list[int] | None = None,
    rng_key: jax.Array | None = None,
) -> tuple[PredictionBatch, RolloutMetricsBatch]:
    pred = predict_replacement_trajectories_for_batch(
        sim_state,
        inference,
        rng_key=rng_key,
        num_samples=num_samples,
    )
    rollout = rollout_predicted_trajectories_with_metrics(
        sim_state,
        pred,
        metric_names=metric_names,
        rollout_num_steps=rollout_num_steps,
        sample_indices=sample_indices,
        rng_key=rng_key,
    )
    return pred, rollout


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--tfrecord_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--do_rollout", action="store_true")
    parser.add_argument("--rollout_num_steps", type=int, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="overlap,offroad,sdc_progression",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    inference = load_model_for_inference(
        args.checkpoint_path,
        use_ema=bool(args.use_ema),
        metadata_path=args.metadata_path,
        seed=int(args.seed),
        skip_checkpoint_load=bool(args.skip_checkpoint_load),
    )

    from waymax import config as waymax_config
    from waymax import dataloader

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=args.tfrecord_path,
        max_num_objects=int(args.max_num_objects),
        batch_dims=(int(args.batch_size),),
        shuffle_seed=int(args.shuffle_seed),
    )
    scenarios = dataloader.simulator_state_generator(ds_cfg)
    sim_state = next(scenarios)

    pred = predict_replacement_trajectories_with_periodic_replan(
        sim_state,
        inference,
        num_samples=int(args.num_samples),
        replan_interval_steps=int(args.replan_interval_steps),
    )
    starts, trajs = to_replacement_lists(pred, sample_index=0, trim_anchor=False)

    starts_np = np.asarray(starts, dtype=np.int32)
    print("start_t_min=", int(starts_np.min()))
    print("start_t_max=", int(starts_np.max()))
    print("start_t_mean=", float(starts_np.mean()))
    print("trajectories_world_bkt5.shape=", tuple(np.asarray(pred.trajectories_world_bkt5).shape))
    print("world_t_seconds_bk.shape=", tuple(np.asarray(pred.world_t_seconds_bk).shape))
    print("world_t_valid_bk.shape=", tuple(np.asarray(pred.world_t_valid_bk).shape))
    print("first_start_t=", starts[0])
    print("first_traj_head=\n", trajs[0][: min(5, trajs[0].shape[0])])

    if args.do_rollout:
        metric_names = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
        rollout = rollout_predicted_trajectories_with_metrics(
            sim_state,
            pred,
            metric_names=metric_names,
            rollout_num_steps=args.rollout_num_steps,
        )
        print("rollout.metric_names=", rollout.metric_names)
        print("rollout.rollout_num_steps=", rollout.rollout_num_steps)
        for name in rollout.metric_names:
            print(
                f"rollout.metric[{name}].shape=",
                tuple(np.asarray(rollout.metric_timeseries[name]).shape),
            )


if __name__ == "__main__":
    main()
