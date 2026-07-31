"""Behavior cloning + SAC helpers for the SB3 (non-V-Max) pipeline.

BC warm-up uses expert inverse-dynamics actions from the WOMD training split
*excluding* shards that contain failure cases. SAC fine-tuning can sample mixed
scenarios from that same expert stream and the harvested failure set.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import torch as th
import torch.nn.functional as F
from waymax import config as waymax_config
from waymax import datatypes
from waymax import dataloader
from waymax.agents.expert import infer_expert_action

from rl.scenario_source import (
    CachedScenarioSource,
    ScenarioSource,
    ScenarioSpec,
    _ensure_unbatched_scenario,
    _slice_batched_state,
    _to_host_pytree,
)
from rl.waymax_env import (
    POLICY_FRIENDLY_COLLISION,
    POLICY_FRIENDLY_OFFROAD,
    RewardConfig,
    WaymaxGymEnv,
    _compute_goal_xy,
    _compute_observation,
)

# Default data locations (see rl/CONTEXT.md).
DEFAULT_FAILURE_DIR = "/zfsauton/scratch/mineuih/waymax_rs/failure_samples"
DEFAULT_WOMD_DIR = "/zfsauton/scratch/eshau/womd/tf_example/training"
DEFAULT_RUNS_DIR = "/zfsauton/scratch/yixiz/waymax_rs/runs"
DEFAULT_BC_CACHE_DIR = "/zfsauton/scratch/yixiz/waymax_rs/bc_cache"


# --------------------------------------------------------------------------- #
# Expert shard path (exclude failure shards from BC / expert SAC sampling)
# --------------------------------------------------------------------------- #
def parse_failure_specs(failure_dir: str, limit: int | None = None) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for jp in sorted(glob(str(Path(failure_dir) / "**" / "*.json"), recursive=True)):
        name = Path(jp).name
        if name == "summary.json" or name.endswith("instructions.json"):
            continue
        try:
            rec = json.load(open(jp, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict) or "tfrecord" not in rec or "scenario_idx" not in rec:
            continue
        if _is_failure_record(rec):
            specs.append((str(rec["tfrecord"]), int(rec["scenario_idx"])))
    specs = sorted(set(specs))
    if limit is not None:
        specs = specs[: int(limit)]
    return specs


def _is_failure_record(record: dict) -> bool:
    if "success" in record:
        return not bool(record["success"])
    if "task_result" in record:
        return not bool(record["task_result"])
    return True


def failure_shard_indices(failure_dir: str, total_shards: int = 1000) -> set[int]:
    shards: set[int] = set()
    for tfrecord, _idx in parse_failure_specs(failure_dir):
        name = os.path.basename(tfrecord)
        try:
            shards.add(int(name.split("-")[-3]))
        except (IndexError, ValueError):
            continue
    return shards


def build_expert_shard_path(
    womd_dir: str,
    *,
    exclude_shards: set[int] | None = None,
    total_shards: int = 1000,
    work_dir: str | None = None,
    file_prefix: str = "training_tfexample.tfrecord",
) -> str:
    """Symlink farm of kept shards; returns a Waymax ``prefix@K`` path."""
    exclude = set(exclude_shards or set())
    full_path = os.path.join(womd_dir, f"{file_prefix}@{total_shards}")
    if not exclude:
        return full_path

    keep = [i for i in range(total_shards) if i not in exclude]
    k = len(keep)
    if work_dir is None:
        work_dir = os.path.join("/tmp", f"sb3_expert_shards_{total_shards}_excl{len(exclude)}")
    os.makedirs(work_dir, exist_ok=True)

    base = os.path.join(work_dir, file_prefix)
    for j, shard_i in enumerate(keep):
        src = os.path.join(womd_dir, f"{file_prefix}-{shard_i:05d}-of-{total_shards:05d}")
        dst = f"{base}-{j:05d}-of-{k:05d}"
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
    return f"{base}@{k}"


def _scenario_batch_size(state: Any) -> int:
    """Leading batch dim of a Waymax state, or 1 if already unbatched."""
    is_sdc = np.asarray(state.object_metadata.is_sdc)
    if is_sdc.ndim >= 2:
        return int(is_sdc.shape[0])
    return 1


def _iter_unbatched_scenarios(host: Any) -> Iterator[Any]:
    """Yield unbatched scenarios from a host state (batch dim 0 or already flat)."""
    is_sdc = np.asarray(host.object_metadata.is_sdc)
    if is_sdc.ndim == 1:
        yield host
        return
    batch_size = _scenario_batch_size(host)
    for i in range(batch_size):
        yield _slice_batched_state(host, i)


def make_expert_scenario_generator(
    expert_path: str,
    *,
    max_num_objects: int | None = None,
    batch_size: int = 1,
    seed: int | None = 0,
    repeat: int | None = None,
    distributed: bool | None = None,
) -> Iterator[Any]:
    """Stream unbatched host NumPy scenarios from the expert split.

    Default ``repeat=None`` loops forever (SAC expert stream). ``repeat=1`` yields
    each scenario in the split exactly once then stops.
    """
    if distributed is None:
        distributed = repeat is None
    config = waymax_config.DatasetConfig(
        path=expert_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        batch_dims=(int(batch_size),),
        shuffle_seed=seed,
        repeat=repeat,
        distributed=distributed,
        drop_remainder=False if repeat == 1 else True,
        data_format=waymax_config.DataFormat.TFRECORD,
    )
    gen = dataloader.simulator_state_generator(config)
    while True:
        try:
            host = _to_host_pytree(next(gen))
        except StopIteration:
            return
        for scen in _iter_unbatched_scenarios(host):
            yield scen


# --------------------------------------------------------------------------- #
# Mixed scenario source for SAC (expert stream + cached failures)
# --------------------------------------------------------------------------- #
class MixedScenarioSource:
    """Randomly samples expert-stream scenarios or cached failure scenarios."""

    def __init__(
        self,
        expert_gen: Iterator[Any],
        failure_source: ScenarioSource,
        *,
        expert_prob: float = 0.5,
        seed: int = 0,
    ):
        self._expert_gen = expert_gen
        self._failure_source = failure_source
        self._expert_prob = float(expert_prob)
        self._rng = np.random.default_rng(seed)
        self.num_objects = int(failure_source.num_objects)

    def __len__(self) -> int:
        return len(self._failure_source)

    def get(self, index: int) -> Any:
        return self._failure_source.get(index)

    def spec(self, index: int) -> ScenarioSpec:
        return self._failure_source.spec(index)

    def sample_index(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._failure_source)))

    def sample_scenario(self, rng: np.random.Generator | None = None) -> Any:
        rng = rng or self._rng
        if rng.random() < self._expert_prob:
            return next(self._expert_gen)
        return self._failure_source.get(self.sample_index(rng))


class MixedWaymaxGymEnv(WaymaxGymEnv):
    """WaymaxGymEnv that resets from a :class:`MixedScenarioSource`."""

    def _next_scenario_index(self) -> int:
        return 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        scen_np = _ensure_unbatched_scenario(
            _to_host_pytree(self._source.sample_scenario(self._np_rng))
        )
        scen = jax.tree_util.tree_map(jnp.asarray, scen_np)
        self._state = self._jit_reset(scen)
        self._goal_xy = jnp.asarray(_compute_goal_xy(scen_np), dtype=jnp.float32)

        from rl.waymax_env import _compute_route, _project_to_route

        self._route_xy, self._route_s = _compute_route(scen_np)

        obs, goal_dist, ego_xy = self._jit_obs(self._state, self._goal_xy)
        self._prev_dist = float(goal_dist)
        self._prev_s, init_lateral = _project_to_route(
            np.asarray(ego_xy), self._route_xy, self._route_s
        )
        self._step_count = 0

        num_timesteps = int(np.asarray(scen_np.log_trajectory.x).shape[-1])
        init_steps = self._env.config.init_steps
        self._max_steps = min(self._max_episode_steps_cap, num_timesteps - init_steps)

        info = {
            "goal_dist": self._prev_dist,
            "lateral_deviation_m": float(init_lateral),
            "scenario": {},
        }
        return np.asarray(obs, dtype=np.float32), info


# --------------------------------------------------------------------------- #
# Expert action + BC dataset
# --------------------------------------------------------------------------- #
def expert_action_sdc(state: Any, dynamics: Any) -> np.ndarray:
    """Per-env SDC inverse-dynamics action in [-1, 1] (bicycle, normalized)."""
    actions = infer_expert_action(state, dynamics).data
    is_sdc = state.object_metadata.is_sdc
    sdc_idx = jnp.argmax(is_sdc.astype(jnp.int32), axis=-1)
    if sdc_idx.ndim == 0:
        action = actions[sdc_idx]
    else:
        action = jnp.take_along_axis(actions, sdc_idx[..., None, None], axis=-2)
        action = jnp.squeeze(action, axis=-2)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def collect_expert_transitions(
    scen_np: Any,
    *,
    action_space_type: str = "bicycle",
    delta_max_dx: float = 6.0,
    delta_max_dy: float = 6.0,
    delta_max_dyaw: float = float(np.pi),
    max_episode_steps: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop expert rollout → (observations [T,D], actions [T,A])."""
    from waymax import dynamics as waymax_dynamics
    from waymax import env as waymax_env

    scen_np = _ensure_unbatched_scenario(_to_host_pytree(scen_np))

    if action_space_type == "bicycle":
        dynamics = waymax_dynamics.InvertibleBicycleModel(normalize_actions=True)
        action_dim = 2
        action_scale = None
    else:
        dynamics = waymax_dynamics.DeltaLocal(
            max_dx=float(delta_max_dx),
            max_dy=float(delta_max_dy),
            max_dyaw=float(delta_max_dyaw),
        )
        action_dim = 3
        action_scale = np.array([delta_max_dx, delta_max_dy, delta_max_dyaw], dtype=np.float32)

    max_num_objects = int(np.asarray(scen_np.object_metadata.is_sdc).shape[-1])
    env_cfg = waymax_config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        controlled_object=waymax_config.ObjectType.SDC,
        compute_reward=False,
        metrics=waymax_config.MetricsConfig(metrics_to_run=("overlap", "offroad")),
    )
    env = waymax_env.PlanningAgentEnvironment(dynamics_model=dynamics, config=env_cfg)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    scen = jax.tree_util.tree_map(jnp.asarray, scen_np)
    state = jit_reset(scen)
    goal_xy = jnp.asarray(_compute_goal_xy(scen_np), dtype=jnp.float32)

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    num_timesteps = int(np.asarray(scen_np.log_trajectory.x).shape[-1])
    init_steps = env.config.init_steps
    max_steps = min(max_episode_steps, num_timesteps - init_steps)

    for _ in range(max_steps):
        obs, _, _ = _compute_observation(state, goal_xy)
        action = expert_action_sdc(state, dynamics)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(action[:action_dim])

        if action_scale is not None:
            data = np.clip(action[:action_dim], -1.0, 1.0) * action_scale
        else:
            data = action[:action_dim]
        wx_action = datatypes.Action(
            data=jnp.asarray(data, dtype=jnp.float32),
            valid=jnp.ones((1,), dtype=jnp.bool_),
        )
        state = jit_step(state, wx_action)
        t = int(np.asarray(state.timestep))
        if t >= num_timesteps - 1:
            break

    if not obs_list:
        from rl.waymax_env import observation_dim

        return (
            np.zeros((0, observation_dim()), dtype=np.float32),
            np.zeros((0, action_dim), dtype=np.float32),
        )
    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def build_bc_dataset(
    scenario_iter: Iterator[Any],
    *,
    num_scenarios: int | None,
    action_space_type: str = "bicycle",
    delta_max_dx: float = 6.0,
    delta_max_dy: float = 6.0,
    delta_max_dyaw: float = float(np.pi),
    max_episode_steps: int = 80,
    verbose: bool = True,
    cache_scenarios: bool = False,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, list[Any]]:
    """Collect BC transitions from ``scenario_iter``.

    ``num_scenarios=None`` consumes the iterator until exhaustion (full expert split).
    """
    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    scenario_cache: list[Any] = []
    collected = 0
    progress_every = 100 if num_scenarios is None else 10
    while num_scenarios is None or collected < num_scenarios:
        try:
            scen_np = _ensure_unbatched_scenario(_to_host_pytree(next(scenario_iter)))
        except StopIteration:
            if num_scenarios is None:
                break
            raise RuntimeError(
                f"BC scenario iterator exhausted after {collected} scenarios; "
                f"requested {num_scenarios}."
            ) from None
        obs, act = collect_expert_transitions(
            scen_np,
            action_space_type=action_space_type,
            delta_max_dx=delta_max_dx,
            delta_max_dy=delta_max_dy,
            delta_max_dyaw=delta_max_dyaw,
            max_episode_steps=max_episode_steps,
        )
        if obs.shape[0] == 0:
            continue
        obs_chunks.append(obs)
        act_chunks.append(act)
        if cache_scenarios:
            scenario_cache.append(scen_np)
        collected += 1
        if verbose and collected % progress_every == 0:
            n_trans = sum(c.shape[0] for c in obs_chunks)
            if num_scenarios is None:
                print(f"[bc_core] collected {collected} scenarios ({n_trans} transitions)")
            else:
                print(
                    f"[bc_core] collected {collected}/{num_scenarios} scenarios "
                    f"({n_trans} transitions)"
                )

    if not obs_chunks:
        raise RuntimeError("BC dataset is empty — no expert transitions collected.")
    all_obs = np.concatenate(obs_chunks, axis=0)
    all_act = np.concatenate(act_chunks, axis=0)
    if verbose:
        print(f"[bc_core] BC dataset: {all_obs.shape[0]} transitions from "
              f"{len(obs_chunks)} scenarios")
        _print_bc_action_stats(all_act, action_space_type)
    if cache_scenarios:
        return all_obs, all_act, len(obs_chunks), scenario_cache
    return all_obs, all_act, len(obs_chunks)


def collect_bc_scenarios(
    scenario_iter: Iterator[Any],
    *,
    num_scenarios: int,
    verbose: bool = True,
) -> list[Any]:
    """Re-sample ``num_scenarios`` from ``scenario_iter`` (for BC sanity after cache load)."""
    scenario_cache: list[Any] = []
    collected = 0
    while collected < num_scenarios:
        try:
            scen_np = _ensure_unbatched_scenario(_to_host_pytree(next(scenario_iter)))
        except StopIteration as exc:
            raise RuntimeError(
                f"BC scenario iterator exhausted after {collected} scenarios; "
                f"requested {num_scenarios}."
            ) from exc
        scenario_cache.append(scen_np)
        collected += 1
        if verbose and collected % 10 == 0:
            print(f"[bc_core] cached {collected}/{num_scenarios} BC sanity scenarios")
    if verbose:
        print(f"[bc_core] cached {collected} BC sanity scenarios")
    return scenario_cache


def bc_dataset_cache_dir(
    cache_root: str | Path,
    args: Any,
    *,
    exclude_shards: set[int],
) -> Path:
    """Stable on-disk directory for a BC dataset configuration."""
    scen = "all" if args.bc_scenarios is None else str(int(args.bc_scenarios))
    objs = "all" if args.max_num_objects is None else str(int(args.max_num_objects))
    parts = [
        f"scenarios_{scen}",
        f"seed_{int(args.bc_seed)}",
        f"action_{args.action_space}",
        f"steps_{int(args.max_episode_steps)}",
        f"objs_{objs}",
        f"excl_{len(exclude_shards)}shards",
    ]
    if args.action_space == "delta":
        parts.append(
            f"dx{float(args.delta_max_dx)}_dy{float(args.delta_max_dy)}"
            f"_dyaw{float(args.delta_max_dyaw)}"
        )
    return Path(cache_root) / "_".join(parts)


def bc_dataset_meta(
    *,
    expert_path: str,
    args: Any,
    exclude_shards: set[int],
    total_shards: int,
    n_scenarios: int,
    n_transitions: int,
) -> dict[str, Any]:
    from rl.waymax_env import observation_dim

    return {
        # v2 adds obs_dim: the observation layout gained agent size/type,
        # roadgraph direction and traffic-light blocks (rl/obs_layout.py), so
        # caches built against the old flat layout must not be reused.
        "version": 2,
        "obs_dim": observation_dim(),
        "expert_path": expert_path,
        "excluded_shards": sorted(int(s) for s in exclude_shards),
        "total_shards": int(total_shards),
        "bc_scenarios": "all" if args.bc_scenarios is None else int(args.bc_scenarios),
        "bc_seed": int(args.bc_seed),
        "action_space": args.action_space,
        "max_episode_steps": int(args.max_episode_steps),
        "max_num_objects": args.max_num_objects,
        "delta_max_dx": float(args.delta_max_dx),
        "delta_max_dy": float(args.delta_max_dy),
        "delta_max_dyaw": float(args.delta_max_dyaw),
        "n_scenarios": int(n_scenarios),
        "n_transitions": int(n_transitions),
    }


def _bc_cache_matches(cached: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "version",
        "obs_dim",
        "expert_path",
        "excluded_shards",
        "total_shards",
        "bc_scenarios",
        "bc_seed",
        "action_space",
        "max_episode_steps",
        "max_num_objects",
        "delta_max_dx",
        "delta_max_dy",
        "delta_max_dyaw",
    )
    return all(cached.get(k) == expected.get(k) for k in keys)


def save_bc_dataset_cache(
    cache_path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    meta: dict[str, Any],
) -> None:
    cache_path = Path(cache_path)
    staging = cache_path.with_name(cache_path.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    np.save(staging / "observations.npy", observations)
    np.save(staging / "actions.npy", actions)
    with open(staging / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if cache_path.exists():
        shutil.rmtree(cache_path)
    staging.rename(cache_path)
    print(
        f"[bc_core] saved BC dataset cache to {cache_path} "
        f"({meta['n_transitions']} transitions, {meta['n_scenarios']} scenarios)"
    )


def load_bc_dataset_cache(
    cache_path: str | Path,
    *,
    expected_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    meta_path = cache_path / "meta.json"
    obs_path = cache_path / "observations.npy"
    act_path = cache_path / "actions.npy"
    for path in (meta_path, obs_path, act_path):
        if not path.is_file():
            raise FileNotFoundError(f"BC cache incomplete: missing {path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if expected_meta is not None and not _bc_cache_matches(meta, expected_meta):
        raise ValueError("BC cache metadata does not match the requested dataset config.")
    observations = np.load(obs_path, mmap_mode="r")
    actions = np.load(act_path, mmap_mode="r")
    print(
        f"[bc_core] loaded BC dataset cache from {cache_path} "
        f"({meta['n_transitions']} transitions, {meta['n_scenarios']} scenarios)"
    )
    return np.asarray(observations), np.asarray(actions), meta


def load_or_build_bc_dataset(
    *,
    expert_path: str,
    exclude_shards: set[int],
    total_shards: int,
    scenario_iter: Iterator[Any],
    args: Any,
    cache_scenarios: bool = False,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, list[Any]]:
    """Load cached BC transitions or collect once and persist to ``cache_dir``."""
    cache_root = Path(cache_dir or DEFAULT_BC_CACHE_DIR)
    cache_path = bc_dataset_cache_dir(cache_root, args, exclude_shards=exclude_shards)
    expected_meta = bc_dataset_meta(
        expert_path=expert_path,
        args=args,
        exclude_shards=exclude_shards,
        total_shards=total_shards,
        n_scenarios=0,
        n_transitions=0,
    )

    if use_cache and not rebuild_cache:
        try:
            obs, act, meta = load_bc_dataset_cache(cache_path, expected_meta=expected_meta)
            n_scenarios = int(meta["n_scenarios"])
            if cache_scenarios:
                print(
                    "[bc_core] BC transitions loaded from cache; "
                    "re-sampling scenarios for sanity check"
                )
                scenarios = collect_bc_scenarios(
                    scenario_iter,
                    num_scenarios=int(args.bc_scenarios),
                )
                return obs, act, n_scenarios, scenarios
            return obs, act, n_scenarios
        except FileNotFoundError:
            pass
        except ValueError as exc:
            print(f"[bc_core] ignoring stale BC cache at {cache_path}: {exc}")

    result = build_bc_dataset(
        scenario_iter,
        num_scenarios=args.bc_scenarios,
        action_space_type=args.action_space,
        delta_max_dx=args.delta_max_dx,
        delta_max_dy=args.delta_max_dy,
        delta_max_dyaw=args.delta_max_dyaw,
        max_episode_steps=args.max_episode_steps,
        cache_scenarios=cache_scenarios,
    )
    if cache_scenarios:
        obs, act, n_scenarios, scenarios = result
    else:
        obs, act, n_scenarios = result
        scenarios = None

    if use_cache:
        meta = bc_dataset_meta(
            expert_path=expert_path,
            args=args,
            exclude_shards=exclude_shards,
            total_shards=total_shards,
            n_scenarios=n_scenarios,
            n_transitions=int(obs.shape[0]),
        )
        save_bc_dataset_cache(cache_path, obs, act, meta)

    if cache_scenarios:
        return obs, act, n_scenarios, scenarios
    return obs, act, n_scenarios


def _print_bc_action_stats(actions: np.ndarray, action_space_type: str) -> None:
    """Debug: verify expert actions are in the expected normalized [-1, 1] range."""
    print(f"[bc_core] BC action stats ({action_space_type}):")
    print(f"  mean: {np.mean(actions, axis=0)}")
    print(f"  std : {np.std(actions, axis=0)}")
    print(f"  min : {np.min(actions, axis=0)}")
    print(f"  max : {np.max(actions, axis=0)}")


# --------------------------------------------------------------------------- #
# BC supervised training on SAC actor
# --------------------------------------------------------------------------- #
def bc_actor_loss(actor: Any, obs: th.Tensor, target_actions: th.Tensor) -> th.Tensor:
    """MSE between tanh-Gaussian mean and expert actions in [-1, 1]."""
    features = actor.extract_features(obs, actor.features_extractor)
    latent_pi = actor.latent_pi(features)
    mean_actions = actor.mu(latent_pi)
    pred = th.tanh(mean_actions)
    return F.mse_loss(pred, target_actions)


def train_bc_actor(
    model: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    log_prefix: str = "bc",
    wandb_log: bool = False,
) -> dict[str, float]:
    """Supervised BC warm-up of ``model.actor`` (SAC instance)."""
    device = model.device
    obs_t = th.as_tensor(observations, device=device, dtype=th.float32)
    act_t = th.as_tensor(actions, device=device, dtype=th.float32)
    optimizer = th.optim.Adam(model.actor.parameters(), lr=lr)

    n = obs_t.shape[0]
    losses: list[float] = []
    for epoch in range(int(epochs)):
        perm = th.randperm(n, device=device)
        epoch_losses: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            loss = bc_actor_loss(model.actor, obs_t[idx], act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        logger = getattr(model, "_logger", None)
        if logger is not None:
            logger.record(f"{log_prefix}/epoch", epoch)
            logger.record(f"{log_prefix}/loss", mean_loss)
        if wandb_log:
            import wandb

            wandb.log(
                {
                    f"{log_prefix}/loss": mean_loss,
                    "bc/epoch": float(epoch),
                    "bc/transitions": float(n),
                }
            )
        print(f"[bc_core] epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}")

    return {"bc/final_loss": losses[-1], "bc/epochs": float(epochs)}


# --------------------------------------------------------------------------- #
# SAC with optional frozen actor (staged Q warm-up)
# --------------------------------------------------------------------------- #
class FrozenActorSAC:
    """Factory: returns a SAC subclass that can skip actor updates."""

    @staticmethod
    def build_class():
        from gymnasium import spaces
        from stable_baselines3 import SAC
        from stable_baselines3.common.utils import polyak_update

        class _FrozenActorSAC(SAC):
            def __init__(self, *args, actor_freeze_timesteps: int = 0, **kwargs):
                super().__init__(*args, **kwargs)
                self.actor_freeze_timesteps = int(actor_freeze_timesteps)
                self._actor_frozen = False

            def _in_actor_freeze_phase(self) -> bool:
                return self.num_timesteps < self.actor_freeze_timesteps

            def _set_actor_frozen(self, frozen: bool) -> None:
                if frozen == self._actor_frozen:
                    return
                self._actor_frozen = frozen
                for param in self.actor.parameters():
                    param.requires_grad = not frozen
                state = "frozen" if frozen else "trainable"
                print(f"[bc_core] actor is {state} at env step {self.num_timesteps}")

            def _sample_action(
                self,
                learning_starts: int,
                action_noise=None,
                n_envs: int = 1,
            ) -> tuple[np.ndarray, np.ndarray]:
                # During critic warm-up, roll out the BC actor (deterministic), not
                # SB3's uniform random warmup before learning_starts.
                if self._in_actor_freeze_phase():
                    assert self._last_obs is not None, "self._last_obs was not set"
                    unscaled_action, _ = self.predict(self._last_obs, deterministic=True)
                    if isinstance(self.action_space, spaces.Box):
                        scaled_action = self.policy.scale_action(unscaled_action)
                        buffer_action = scaled_action
                        action = self.policy.unscale_action(scaled_action)
                    else:
                        buffer_action = unscaled_action
                        action = buffer_action
                    return action, buffer_action
                return super()._sample_action(learning_starts, action_noise, n_envs)

            def _deterministic_actor_actions(self, obs: th.Tensor) -> th.Tensor:
                features = self.actor.extract_features(obs, self.actor.features_extractor)
                latent_pi = self.actor.latent_pi(features)
                return th.tanh(self.actor.mu(latent_pi))

            def train(self, gradient_steps: int, batch_size: int = 64) -> None:
                self._set_actor_frozen(self._in_actor_freeze_phase())

                self.policy.set_training_mode(True)
                optimizers = [self.actor.optimizer, self.critic.optimizer]
                if self.ent_coef_optimizer is not None:
                    optimizers += [self.ent_coef_optimizer]
                self._update_learning_rate(optimizers)

                ent_coef_losses, ent_coefs = [], []
                actor_losses, critic_losses = [], []
                frozen = self._actor_frozen

                for gradient_step in range(gradient_steps):
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                    discounts = (
                        replay_data.discounts if replay_data.discounts is not None else self.gamma
                    )

                    if self.use_sde:
                        self.actor.reset_noise()

                    actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                    log_prob = log_prob.reshape(-1, 1)

                    ent_coef_loss = None
                    if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                        ent_coef = th.exp(self.log_ent_coef.detach())
                        ent_coef_loss = -(
                            self.log_ent_coef * (log_prob + self.target_entropy).detach()
                        ).mean()
                        ent_coef_losses.append(ent_coef_loss.item())
                    else:
                        ent_coef = self.ent_coef_tensor

                    ent_coefs.append(ent_coef.item())

                    if (
                        not frozen
                        and ent_coef_loss is not None
                        and self.ent_coef_optimizer is not None
                    ):
                        self.ent_coef_optimizer.zero_grad()
                        ent_coef_loss.backward()
                        self.ent_coef_optimizer.step()

                    with th.no_grad():
                        if frozen:
                            next_actions = self._deterministic_actor_actions(
                                replay_data.next_observations
                            )
                            next_q_values = th.cat(
                                self.critic_target(replay_data.next_observations, next_actions),
                                dim=1,
                            )
                            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                            # Critic warm-up: Q of BC mean policy, no entropy term.
                            target_q_values = replay_data.rewards + (
                                1 - replay_data.dones
                            ) * discounts * next_q_values
                        else:
                            next_actions, next_log_prob = self.actor.action_log_prob(
                                replay_data.next_observations
                            )
                            next_q_values = th.cat(
                                self.critic_target(replay_data.next_observations, next_actions),
                                dim=1,
                            )
                            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                                -1, 1
                            )
                            target_q_values = replay_data.rewards + (
                                1 - replay_data.dones
                            ) * discounts * next_q_values

                    current_q_values = self.critic(replay_data.observations, replay_data.actions)
                    critic_loss = 0.5 * sum(
                        F.mse_loss(current_q, target_q_values) for current_q in current_q_values
                    )
                    critic_losses.append(float(critic_loss.item()))

                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()

                    if not frozen:
                        q_values_pi = th.cat(
                            self.critic(replay_data.observations, actions_pi), dim=1
                        )
                        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                        actor_losses.append(float(actor_loss.item()))

                        self.actor.optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor.optimizer.step()

                    if gradient_step % self.target_update_interval == 0:
                        polyak_update(
                            self.critic.parameters(), self.critic_target.parameters(), self.tau
                        )
                        polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                self._n_updates += gradient_steps
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/ent_coef", np.mean(ent_coefs))
                if actor_losses:
                    self.logger.record("train/actor_loss", np.mean(actor_losses))
                self.logger.record("train/critic_loss", np.mean(critic_losses))
                if ent_coef_losses:
                    self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
                self.logger.record("train/actor_frozen", float(frozen))
                self.logger.record(
                    "train/actor_freeze_timesteps", float(self.actor_freeze_timesteps)
                )
                self.logger.record("train/env_timesteps", float(self.num_timesteps))

        return _FrozenActorSAC


FrozenActorSACClass = FrozenActorSAC.build_class()


# --------------------------------------------------------------------------- #
# Env / reward helpers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BCTrainPaths:
    failure_dir: str = DEFAULT_FAILURE_DIR
    womd_dir: str = DEFAULT_WOMD_DIR
    expert_work_dir: str | None = None


def resolve_expert_path(
    paths: BCTrainPaths,
    save_dir: str,
    *,
    total_shards: int = 1000,
) -> str:
    exclude = failure_shard_indices(paths.failure_dir, total_shards=total_shards)
    work = paths.expert_work_dir or os.path.join(save_dir, "expert_shards")
    expert_path = build_expert_shard_path(
        paths.womd_dir,
        exclude_shards=exclude,
        total_shards=total_shards,
        work_dir=work,
    )
    print(f"[bc_core] expert path: {expert_path} (excluded shards: {sorted(exclude)})")
    return expert_path


def policy_friendly_reward_config(args: Any) -> RewardConfig:
    """SAC defaults aligned with vmax_rl policy-friendly sim."""
    collision = getattr(args, "r_collision", POLICY_FRIENDLY_COLLISION)
    offroad = getattr(args, "r_offroad", POLICY_FRIENDLY_OFFROAD)
    return RewardConfig(
        progress=getattr(args, "r_progress", 1.0),
        action_penalty=getattr(args, "r_action_penalty", 0.01),
        collision=collision,
        offroad=offroad,
        goal_bonus=getattr(args, "r_goal_bonus", 10.0),
        goal_threshold_m=getattr(args, "goal_threshold_m", 3.0),
        terminate_on_collision=getattr(args, "terminate_on_collision", False),
        terminate_on_offroad=getattr(args, "terminate_on_offroad", False),
        route_reward=getattr(args, "route_reward", False),
        lateral_penalty=getattr(args, "r_lateral_penalty", 0.5),
    )


def make_waymax_env(
    source: Any,
    args: Any,
    *,
    seed: int,
    sequential: bool,
    mixed: bool = False,
    reactive_agents: bool | None = None,
) -> WaymaxGymEnv:
    reward_cfg = policy_friendly_reward_config(args)
    env_cls = MixedWaymaxGymEnv if mixed else WaymaxGymEnv
    if reactive_agents is None:
        reactive_agents = getattr(args, "reactive_agents", True)
    return env_cls(
        source,
        reward_config=reward_cfg,
        max_episode_steps=args.max_episode_steps,
        sequential=sequential,
        seed=seed,
        action_space_type=args.action_space,
        delta_max_dx=args.delta_max_dx,
        delta_max_dy=args.delta_max_dy,
        delta_max_dyaw=args.delta_max_dyaw,
        reactive_agents=reactive_agents,
        idm_desired_vel=getattr(args, "idm_desired_vel", 30.0),
    )


def load_expert_eval_source(
    expert_path: str,
    *,
    max_num_objects: int | None = None,
    num_scenarios: int,
    seed: int = 0,
) -> CachedScenarioSource:
    """Cache ``num_scenarios`` expert scenarios for sequential eval."""
    gen = make_expert_scenario_generator(
        expert_path, max_num_objects=max_num_objects, batch_size=1, seed=seed
    )
    specs: list[ScenarioSpec] = []
    scenarios: list[Any] = []
    for i in range(num_scenarios):
        scen_np = _ensure_unbatched_scenario(_to_host_pytree(next(gen)))
        specs.append(ScenarioSpec(expert_path, i))
        scenarios.append(scen_np)
    return CachedScenarioSource(scenarios, specs, label=expert_path)


def evaluate_policy_on_source(
    model: Any,
    source: ScenarioSource | CachedScenarioSource,
    args: Any,
    *,
    n_episodes: int | None = None,
    sequential: bool = True,
    reactive_agents: bool | None = None,
    metrics_prefix: str = "eval",
) -> dict[str, float]:
    from rl.sac_callbacks import evaluate_waymax_policy

    env = make_waymax_env(
        source, args, seed=0, sequential=sequential, reactive_agents=reactive_agents
    )
    n = n_episodes if n_episodes is not None else len(source)
    metrics = evaluate_waymax_policy(model, env, n_episodes=n, deterministic=True)
    if metrics_prefix == "eval":
        return metrics
    renamed: dict[str, float] = {}
    for key, val in metrics.items():
        if key.startswith("eval/"):
            renamed[f"{metrics_prefix}/{key[5:]}"] = val
        else:
            renamed[key] = val
    return renamed


def evaluate_expert_replay_on_source(
    source: ScenarioSource | CachedScenarioSource,
    args: Any,
    *,
    n_episodes: int | None = None,
    sequential: bool = True,
    reactive_agents: bool | None = None,
    metrics_prefix: str = "eval",
) -> dict[str, float]:
    from rl.sac_callbacks import evaluate_expert_replay

    env = make_waymax_env(
        source, args, seed=0, sequential=sequential, reactive_agents=reactive_agents
    )
    n = n_episodes if n_episodes is not None else len(source)
    return evaluate_expert_replay(env, n_episodes=n, prefix=metrics_prefix)


def run_bc_sanity_check(
    model: Any,
    scenarios: Sequence[Any],
    args: Any,
    *,
    expert_path: str = "",
) -> dict[str, float]:
    """Eval BC policy vs expert replay on the exact BC training scenarios."""
    source = CachedScenarioSource(scenarios, label=expert_path or "bc_train")
    n = len(source)
    reactive = bool(getattr(args, "bc_sanity_reactive_agents", False))

    print(f"[bc_core] BC sanity check on {n} training scenarios "
          f"(reactive_agents={reactive})")

    bc_metrics = evaluate_policy_on_source(
        model,
        source,
        args,
        n_episodes=n,
        reactive_agents=not reactive,
        metrics_prefix="sanity/bc_policy",
    )
    expert_metrics = evaluate_expert_replay_on_source(
        source,
        args,
        n_episodes=n,
        reactive_agents=not reactive,
        metrics_prefix="sanity/expert_replay",
    )
    summary: dict[str, float] = {
        "sanity/n_scenarios": float(n),
        "sanity/reactive_agents": float(reactive),
        **bc_metrics,
        **expert_metrics,
    }

    if reactive:
        bc_idm = evaluate_policy_on_source(
            model,
            source,
            args,
            n_episodes=n,
            reactive_agents=True,
            metrics_prefix="sanity/bc_policy_idm",
        )
        expert_idm = evaluate_expert_replay_on_source(
            source,
            args,
            n_episodes=n,
            reactive_agents=True,
            metrics_prefix="sanity/expert_replay_idm",
        )
        summary.update(bc_idm)
        summary.update(expert_idm)

    print("[bc_core] BC sanity (log-replay agents):")
    print(
        f"  expert replay clean={summary['sanity/expert_replay/clean_success_rate']:.3f} "
        f"(collision={summary['sanity/expert_replay/collision_rate']:.3f}, "
        f"offroad={summary['sanity/expert_replay/offroad_rate']:.3f})"
    )
    print(
        f"  BC policy     clean={summary['sanity/bc_policy/clean_success_rate']:.3f} "
        f"(collision={summary['sanity/bc_policy/collision_rate']:.3f}, "
        f"offroad={summary['sanity/bc_policy/offroad_rate']:.3f})"
    )
    if reactive:
        print("[bc_core] BC sanity (IDM reactive agents):")
        print(
            f"  expert replay clean={summary['sanity/expert_replay_idm/clean_success_rate']:.3f}"
        )
        print(
            f"  BC policy     clean={summary['sanity/bc_policy_idm/clean_success_rate']:.3f}"
        )
    return summary
