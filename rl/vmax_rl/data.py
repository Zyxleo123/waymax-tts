"""Data generators that point V-Max training at our failure / non-failure splits.

V-Max consumes batches of Waymax ``SimulatorState`` from an iterator (one
``next()`` per training iteration). For the standard pipeline that iterator just
streams a WOMD TFRecord. Here we provide two specialised iterators:

* :func:`make_failure_generator` -- yields batches drawn (with replacement) from
  the **failure cases** harvested under ``failure_samples/``. The 245 scenarios
  are loaded once and cached on-device, then re-sampled each iteration. Used for
  the **SAC / RL** half of the loop.
* :func:`make_expert_generator` -- streams the **non-failure** WOMD expert data
  via V-Max's own dataloader, excluding the shards that contain failure cases so
  none of them leak into the behavior-cloning warm-up. Used for the **imitation**
  half of the loop.

Both yield states with batch dims ``(num_devices, num_envs, num_episode_per_epoch)``
and ``include_sdc_paths=False`` -- the SDC route is generated on reset by
V-Max's ``SDCPathWrapper`` (i.e. ``waymo_dataset=True`` mode), exactly matching
what ``simulator.make_data_generator`` produces.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.vmax_rl import compat  # noqa: F401  (JAX compat shims)

import jax
import jax.numpy as jnp
from waymax import config as waymax_config
from waymax import dataloader

from vmax import simulator


# WOMD 1.3.1 tf_example ships real SDC route paths (``path_samples/*``) baked in:
# 45 candidate paths x 800 points each, with per-path ``on_route`` flags. These
# match Waymax's ``WOD_1_3_1_TRAINING`` dataset config. We load them directly
# (``include_sdc_paths=True``) instead of regenerating an approximate single-path
# route at reset with V-Max's heuristic ``SDCPathWrapper`` -- the route feeds the
# progression / off_route rewards, the path_target observation, and red-light /
# route metrics, so using the dataset's curated paths removes a real train/eval
# discrepancy vs the ScenarioMax-style data V-Max was tuned on.
WOMD_NUM_SDC_PATHS = 45
WOMD_NUM_POINTS_PER_SDC_PATH = 800


# --------------------------------------------------------------------------- #
# Failure-case set (RL half)
# --------------------------------------------------------------------------- #
def _stack_host_scenarios(scenarios: Sequence[Any]) -> Any:
    """Stack a list of unbatched host pytrees into a single ``[N, ...]`` pytree."""
    return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves, axis=0), *scenarios)


def _is_failure(record: dict) -> bool:
    """True if the result record marks a failure (the cases we want to solve)."""
    if "success" in record:
        return not bool(record["success"])
    if "task_result" in record:
        return not bool(record["task_result"])
    return False


def parse_failure_specs(failure_dir: str, limit: int | None = None) -> list[tuple[str, int]]:
    """Parse failure JSONs into a sorted list of ``(tfrecord, scenario_idx)`` pairs."""
    specs: list[tuple[str, int]] = []
    for jp in sorted(glob(os.path.join(failure_dir, "**", "*.json"), recursive=True)):
        name = Path(jp).name
        if name == "summary.json" or name.endswith("instructions.json"):
            continue
        try:
            rec = json.load(open(jp))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict) or "tfrecord" not in rec or "scenario_idx" not in rec:
            continue
        if not _is_failure(rec):
            continue
        specs.append((str(rec["tfrecord"]), int(rec["scenario_idx"])))
    specs = sorted(set(specs))
    if limit is not None:
        specs = specs[: int(limit)]
    return specs


def _dataset_config(shard_path: str, max_num_objects: int) -> waymax_config.DatasetConfig:
    """A WOMD DatasetConfig that keeps the dataset's real SDC route paths."""
    return waymax_config.DatasetConfig(
        path=shard_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        num_paths=WOMD_NUM_SDC_PATHS,
        num_points_per_path=WOMD_NUM_POINTS_PER_SDC_PATH,
        batch_dims=(),
        shuffle_seed=None,
        repeat=1,
        distributed=False,
        data_format=waymax_config.DataFormat.TFRECORD,
    )


def load_failure_scenarios(
    failure_dir: str,
    *,
    max_num_objects: int = 64,
    limit: int | None = None,
    refit_sdc_log: bool = False,
    verbose: bool = True,
) -> tuple[Any, int]:
    """Load every failure scenario into a single host-resident ``[N, ...]`` state.

    Uses Waymax's ``simulator_state_generator`` (the exact pipeline V-Max trains
    on) per shard, so the object count is truncated to ``max_num_objects`` with
    the SDC preserved -- structurally identical to the expert generator and the
    training env. ``scenario_idx`` is the sequential record position in the shard.

    Args:
        failure_dir: Directory of per-scenario result JSONs (``success == false``).
        max_num_objects: Object cap; must match the training env's ``max_num_objects``.
        limit: Optional cap on the number of scenarios (for smoke tests).
        refit_sdc_log: If True, rewrite each SDC's logged ``yaw``/velocity to be
            kinematically consistent with the bicycle dynamics
            (:func:`rl.vmax_rl.kinematic_refit.refit_state_sdc_log`). Defensive /
            optional -- the decisive tracking fix is the per-env SDC action
            selection in ``inference.expert_step``; this only shaves the residual
            sub-metre integration error and well-conditions the low-speed inverse.
        verbose: Print loading progress.

    Returns:
        A tuple ``(stacked_state, num_scenarios)`` where ``stacked_state`` is a
        host (NumPy) ``SimulatorState`` with leading batch dim ``N``.
    """
    specs = parse_failure_specs(failure_dir, limit=limit)
    if not specs:
        raise RuntimeError(f"No failure scenarios found under {failure_dir}.")

    by_shard: dict[str, list[int]] = defaultdict(list)
    for shard, idx in specs:
        by_shard[shard].append(idx)

    collected: list[Any] = []
    dropped = 0
    for shard, idxs in by_shard.items():
        need = set(idxs)
        max_idx = max(idxs)
        gen = dataloader.simulator_state_generator(_dataset_config(shard, max_num_objects))
        for i, state in enumerate(gen):
            if i in need:
                is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
                if int(is_sdc.sum()) == 1:
                    collected.append(jax.tree_util.tree_map(np.asarray, state))
                else:
                    dropped += 1  # truncation lost the SDC; skip (rare)
            if i >= max_idx:
                break
        if verbose:
            print(f"[vmax_rl.data] {Path(shard).name}: requested {len(need)} scenarios.")

    if not collected:
        raise RuntimeError("Loaded 0 valid failure scenarios (all lost their SDC?).")
    stacked = _stack_host_scenarios(collected)
    if refit_sdc_log:
        from rl.vmax_rl.kinematic_refit import refit_state_batch_host

        stacked = jax.tree_util.tree_map(np.asarray, refit_state_batch_host(stacked))
        if verbose:
            print("[vmax_rl.data] Refit SDC log trajectories to be bicycle-consistent.")
    if verbose:
        print(f"[vmax_rl.data] Stacked {len(collected)} failure scenarios "
              f"(max_num_objects={max_num_objects}, dropped={dropped}).")
    return stacked, len(collected)


def make_failure_generator(
    stacked_state: Any,
    *,
    num_envs: int,
    num_episode_per_epoch: int,
    num_devices: int = 1,
    seed: int = 0,
) -> Iterator[Any]:
    """Infinite generator of device-batched failure scenarios.

    Each ``next()`` samples ``num_devices * num_envs * num_episode_per_epoch``
    scenarios (with replacement) from the cached set and reshapes them to batch
    dims ``(num_devices, num_envs, num_episode_per_epoch)``.
    """
    num_scenarios = int(jax.tree_util.tree_leaves(stacked_state)[0].shape[0])
    per_batch = num_devices * num_envs * num_episode_per_epoch
    rng = np.random.default_rng(seed)

    # Cache the (small) failure set on-device once; only integer indices move
    # host->device per iteration.
    device_state = jax.device_put(jax.tree_util.tree_map(jnp.asarray, stacked_state))

    @jax.jit
    def _gather(indices: jax.Array) -> Any:
        return jax.tree_util.tree_map(
            lambda x: x[indices].reshape((num_devices, num_envs, num_episode_per_epoch) + x.shape[1:]),
            device_state,
        )

    while True:
        indices = jnp.asarray(rng.integers(0, num_scenarios, size=per_batch))
        yield _gather(indices)


def make_eval_scenarios(
    stacked_state: Any,
    *,
    num_scenarios: int | None = None,
    num_devices: int = 1,
    rows: int = 8,
) -> Any:
    """Build a fixed, deterministic eval batch from the failure set.

    Returns a state with batch dims ``(num_devices, rows, cols)`` covering the
    first ``num_scenarios`` failure cases (tiled if fewer than requested).
    """
    total = int(jax.tree_util.tree_leaves(stacked_state)[0].shape[0])
    n = total if num_scenarios is None else int(num_scenarios)
    per_device_rows = rows
    # cols chosen so num_devices * rows * cols >= n, then tile/truncate to fit.
    cols = max(1, int(np.ceil(n / (num_devices * per_device_rows))))
    needed = num_devices * per_device_rows * cols
    idx = (np.arange(needed) % total).astype(np.int64)
    device_state = jax.tree_util.tree_map(jnp.asarray, stacked_state)
    return jax.tree_util.tree_map(
        lambda x: x[idx].reshape((num_devices, per_device_rows, cols) + x.shape[1:]),
        device_state,
    )


# --------------------------------------------------------------------------- #
# Non-failure expert set (imitation half)
# --------------------------------------------------------------------------- #
def failure_shard_indices(failure_dir: str, total_shards: int = 1000) -> set[int]:
    """Return the set of WOMD shard indices that contain any failure scenario."""
    shards: set[int] = set()
    for tfrecord, _idx in parse_failure_specs(failure_dir):
        name = os.path.basename(tfrecord)
        # training_tfexample.tfrecord-00002-of-01000 -> 2
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
    """Create a symlink farm of the kept shards and return a Waymax ``prefix@K`` path.

    Waymax only understands a single file or the ``name@N`` sharded notation
    (which expands to ``name-00000-of-000N``..). To *exclude* the failure shards
    we symlink the kept shards into ``work_dir`` under contiguous indices and
    return a ``@K`` path over them.

    If ``exclude_shards`` is empty/None we skip the farm and return the plain
    ``<womd_dir>/<file_prefix>@<total_shards>`` path.
    """
    exclude = set(exclude_shards or set())
    full_path = os.path.join(womd_dir, f"{file_prefix}@{total_shards}")
    if not exclude:
        return full_path

    keep = [i for i in range(total_shards) if i not in exclude]
    k = len(keep)
    if work_dir is None:
        work_dir = os.path.join("/tmp", f"vmax_expert_shards_{total_shards}_excl{len(exclude)}")
    os.makedirs(work_dir, exist_ok=True)

    base = os.path.join(work_dir, file_prefix)
    for j, i in enumerate(keep):
        src = os.path.join(womd_dir, f"{file_prefix}-{i:05d}-of-{total_shards:05d}")
        dst = f"{base}-{j:05d}-of-{k:05d}"
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
    return f"{base}@{k}"


def make_expert_generator(
    expert_path: str,
    *,
    max_num_objects: int = 64,
    num_envs: int,
    num_episode_per_epoch: int,
    seed: int = 0,
) -> Iterator[Any]:
    """Stream non-failure expert scenarios with their real SDC route paths.

    ``include_sdc_paths=True`` (45 x 800, matching WOMD 1.3.1) so the structure
    matches the failure generator and the env can consume the dataset's curated
    route directly. We build the ``DatasetConfig`` here rather than calling
    ``simulator.make_data_generator`` because the latter hardcodes V-Max's
    ScenarioMax path dims (10 x 300), which would mismatch the WOMD path tensors.
    """
    config = waymax_config.DatasetConfig(
        path=expert_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        num_paths=WOMD_NUM_SDC_PATHS,
        num_points_per_path=WOMD_NUM_POINTS_PER_SDC_PATH,
        batch_dims=(num_envs, num_episode_per_epoch),
        shuffle_seed=seed,
        distributed=True,
        data_format=waymax_config.DataFormat.TFRECORD,
    )
    return dataloader.simulator_state_generator(config)
