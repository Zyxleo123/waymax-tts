#!/usr/bin/env python3
"""Roll a trained V-Max policy on failure cases via ScenarioMax scenarios.

Workflow (uses ``rl/trace_failure_to_womd.py`` output):

1. Trace maps each failure JSON -> WOMD ``scenario/id`` -> ScenarioMax record index.
2. This script loads those ScenarioMax records, rolls the SAC/BC/... checkpoint, and
   writes ``*.trajectory.npz`` files matching the failure-sample layout
   (``scenario_idx``, ``ego_idx``, ``predicted_trajectory`` (91,5), ``start_timestep``,
   ``goal_xy``).

Run on a GPU node::

    conda activate waymax
    srun --gres=gpu:1 --partition=general --mem=32G --cpus-per-task=8 bash -lc '
      source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh
      conda activate waymax
      cd /zfsauton2/home/yixiz/waymax_rs
      python -m rl.rollout_pseudo_gt \
        --trace-json /zfsauton/scratch/yixiz/failure_womd_trace.json \
        --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac \
        --output-dir /zfsauton/scratch/yixiz/pseudo_gt/repro_sac \
        --limit 5
    '
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from waymax import dynamics

_REPO = Path(__file__).resolve().parents[1]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl.womd_compare import (  # noqa: E402
    SCENARIOMAX_NUM_PATHS,
    SCENARIOMAX_NUM_POINTS_PER_PATH,
    load_waymax_state,
    sdc_object_index,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace-json",
        required=True,
        help="Output of trace_failure_to_womd.py (with --check-scenariomax).",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="V-Max training run dir (contains .hydra/ and model/).",
    )
    p.add_argument(
        "--src-dir",
        default=None,
        help="Parent of run-dir passed to V-Max eval utils (default: parent of run-dir).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Explicit checkpoint .pkl (default: latest in run-dir/model/).",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split records into this many shards for parallel invocations "
        "(each scenario writes its own file, so shards can run concurrently "
        "into the same --output-dir).",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process handles, in [0, num_shards).",
    )
    p.add_argument(
        "--start-timestep",
        type=int,
        default=None,
        help="Override start_timestep in outputs (default: read from existing .trajectory.npz).",
    )
    return p.parse_args()


def _sdc_sim_trajectory_bt5(state) -> np.ndarray:
    """SDC sim trajectory as (T, 5): x, y, yaw, vel_x, vel_y."""
    idx = sdc_object_index(state, timestep=0)
    traj = state.sim_trajectory
    x = np.asarray(traj.x[0, idx])
    y = np.asarray(traj.y[0, idx])
    yaw = np.asarray(traj.yaw[0, idx])
    vel_x = np.asarray(traj.vel_x[0, idx])
    vel_y = np.asarray(traj.vel_y[0, idx])
    return np.stack([x, y, yaw, vel_x, vel_y], axis=-1).astype(np.float32)


def _sdc_log_trajectory_bt5(state) -> np.ndarray:
    idx = sdc_object_index(state, timestep=0)
    traj = state.log_trajectory
    x = np.asarray(traj.x[0, idx])
    y = np.asarray(traj.y[0, idx])
    yaw = np.asarray(traj.yaw[0, idx])
    vel_x = np.asarray(traj.vel_x[0, idx])
    vel_y = np.asarray(traj.vel_y[0, idx])
    return np.stack([x, y, yaw, vel_x, vel_y], axis=-1).astype(np.float32)


def _read_start_timestep(failure_json: str, failure_dir: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    stem = Path(failure_json).name.replace(".json", "")
    npz_path = Path(failure_dir) / f"{stem}.trajectory.npz"
    if npz_path.is_file():
        return int(np.load(npz_path)["start_timestep"])
    return 0


def _build_trajectory_npz(
    log_bt5: np.ndarray,
    sim_bt5: np.ndarray,
    *,
    start_t: int,
    horizon: int = 91,
) -> np.ndarray:
    """Match failure-sample layout: log prefix, sim suffix from ``start_t``."""
    out = log_bt5.copy()
    end = min(horizon, sim_bt5.shape[0])
    out[start_t:end] = sim_bt5[start_t:end]
    if out.shape[0] < horizon:
        pad = np.repeat(out[-1:], horizon - out.shape[0], axis=0)
        out = np.concatenate([out, pad], axis=0)
    return out[:horizon].astype(np.float32)


def _setup_policy(run_dir: str, src_dir: str | None, model_path: str | None):
    from vmax.scripts.evaluate import utils as eval_utils
    from vmax.simulator import make_env_for_evaluation

    run_dir = os.path.abspath(run_dir)
    src_dir = src_dir or os.path.dirname(run_dir)
    name_run = os.path.basename(run_dir)

    if model_path is None:
        model_path, model_name = eval_utils.get_model_path(os.path.join(run_dir, "model/"))
        print(f"[pseudo_gt] checkpoint: {model_name}")
    else:
        model_path = os.path.abspath(model_path)
        print(f"[pseudo_gt] checkpoint: {model_path}")

    eval_config = eval_utils.load_yaml_config(os.path.join(run_dir, ".hydra/config.yaml"))
    eval_config["encoder"] = eval_config["network"]["encoder"]
    eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
    eval_config["value"] = eval_config["algorithm"]["network"]["value"]
    eval_config["unflatten_config"] = eval_config["observation_config"]
    eval_config["action_distribution"] = eval_config["algorithm"]["network"]["action_distribution"]

    env = make_env_for_evaluation(
        max_num_objects=int(eval_config["max_num_objects"]),
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=eval_config["observation_type"],
        observation_config=eval_config["observation_config"],
        termination_keys=eval_config["termination_keys"],
        noisy_init=False,
    )
    policy = eval_utils.load_model(
        env,
        eval_config["algorithm"]["name"],
        eval_config,
        model_path,
    )
    step_fn = jax.jit(eval_utils.make_step_fn(env, "ai", policy))
    reset_fn = jax.jit(env.reset)
    max_num_objects = int(eval_config["max_num_objects"])
    return env, step_fn, reset_fn, max_num_objects


def _rollout(scenario, key: jax.Array, step_fn, reset_fn):
    """Roll one scenario; keys are batched for VmapWrapper (batch size 1)."""
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, 1)
    transition = reset_fn(scenario, reset_key)

    def cond_fn(carry):
        tr, _, _ = carry
        return jnp.any(jnp.logical_not(tr.done))

    def body_fn(carry):
        tr, k, _ = carry
        k, sk = jax.random.split(k)
        sk = jax.random.split(sk, 1)
        tr, _ = step_fn(tr, key=sk)
        return tr, k, None

    transition, _, _ = jax.lax.while_loop(cond_fn, body_fn, (transition, key, None))
    return transition.state


def main() -> int:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trace = json.load(open(args.trace_json))
    records = trace["records"]
    records = [r for r in records if r.get("scenariomax_hit") and r.get("error") is None]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(
            "No traced failures with scenariomax_hit=True. "
            "Run trace_failure_to_womd.py with --check-scenariomax first."
        )
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError(f"--shard-index must be in [0, {args.num_shards}).")
        records = records[args.shard_index :: args.num_shards]
        if not records:
            print(f"[pseudo_gt] shard {args.shard_index}/{args.num_shards}: no records assigned.")
            return 0

    print(
        f"[pseudo_gt] rolling {len(records)} scenarios "
        + (f"(shard {args.shard_index}/{args.num_shards}) " if args.num_shards > 1 else "")
        + "..."
    )
    _, step_fn, reset_fn, max_num_objects = _setup_policy(args.run_dir, args.src_dir, args.model)
    print(f"[pseudo_gt] max_num_objects: {max_num_objects}")

    key = jax.random.PRNGKey(args.seed)
    written = 0
    for rec in records:
        tfrecord = rec["scenariomax_tfrecord_path"]
        idx = int(rec["scenariomax_record_index"])
        failure_json = rec["json_path"]
        payload = json.load(open(failure_json))

        host_state = load_waymax_state(
            tfrecord,
            idx,
            num_paths=SCENARIOMAX_NUM_PATHS,
            num_points=SCENARIOMAX_NUM_POINTS_PER_PATH,
            max_num_objects=max_num_objects,
        )
        n_obj = int(host_state.log_trajectory.num_objects)
        if n_obj != max_num_objects:
            raise ValueError(
                f"ScenarioMax record {idx}: loaded {n_obj} objects, env expects "
                f"{max_num_objects}. Update rl/womd_compare.load_waymax_state."
            )
        if written == 0:
            print(f"[pseudo_gt] state shape: {tuple(host_state.log_trajectory.x.shape)}")
        device_state = jax.device_put(host_state)
        key, sub = jax.random.split(key)
        final_state = _rollout(device_state, sub, step_fn, reset_fn)
        final_host = jax.device_get(final_state)

        log_bt5 = _sdc_log_trajectory_bt5(host_state)
        sim_bt5 = _sdc_sim_trajectory_bt5(final_host)
        start_t = _read_start_timestep(failure_json, args.failure_dir, args.start_timestep)
        pred_traj = _build_trajectory_npz(log_bt5, sim_bt5, start_t=start_t)

        ego_idx = sdc_object_index(host_state, timestep=start_t)
        goal_xy = np.asarray(payload.get("goal_xy", pred_traj[-1, :2]), dtype=np.float32)
        scenario_idx = int(payload["scenario_idx"])

        stem = Path(failure_json).stem
        out_path = Path(args.output_dir) / f"{stem}.trajectory.npz"
        np.savez(
            out_path,
            scenario_idx=scenario_idx,
            ego_idx=int(ego_idx),
            predicted_trajectory=pred_traj,
            start_timestep=int(start_t),
            goal_xy=goal_xy,
        )
        meta_path = Path(args.output_dir) / f"{stem}.pseudo_gt.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "failure_json": failure_json,
                    "tf2_scenario_id": rec.get("tf2_scenario_id"),
                    "scenariomax_tfrecord": tfrecord,
                    "scenariomax_record_index": idx,
                    "scenario_idx": scenario_idx,
                    "start_timestep": start_t,
                },
                f,
                indent=2,
            )
        written += 1
        print(f"  wrote {out_path.name}  scenariomax_idx={idx}  start_t={start_t}")

    print(f"[pseudo_gt] done: {written} trajectories -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
