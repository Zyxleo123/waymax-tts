"""Loads Waymax scenarios for RL training/eval.

A "scenario" here is a single, unbatched Waymax ``SimulatorState`` (prefix shape
``()``). Scenarios are loaded once and cached on the host as NumPy pytrees so that
``WaymaxGymEnv.reset()`` is cheap (it only needs to move one scenario to device).

Two sources are supported:

* ``ScenarioSource.from_failure_dir(dir)`` — reads the per-scenario JSON files
  emitted by goal-reaching / reward-search runs. Each JSON carries a ``tfrecord``
  path and a ``scenario_idx``; by default only failures (``success == false`` or
  ``task_result == false``) are kept.
* ``ScenarioSource.from_tfrecord(path, indices)`` — loads explicit scenario
  indices from a single TFRecord file (useful for "train on some data" before
  pointing at real failure cases).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
from waymax import config as waymax_config
from waymax import dataloader

from viz.render import _load_scenario_state_batch_fast


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    """Pointer to a single scenario inside a TFRecord file."""

    tfrecord: str
    scenario_idx: int


def _to_host_pytree(tree: Any) -> Any:
    """Detaches a (possibly device-resident) pytree to host NumPy arrays."""
    return jax.tree_util.tree_map(lambda a: np.asarray(a), tree)


def _slice_batched_state(state_batch: Any, batch_idx: int) -> Any:
    """Extracts one unbatched scenario (prefix shape ()) from a [B, ...] state."""
    return jax.tree_util.tree_map(lambda a: np.asarray(a[batch_idx]), state_batch)


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
    num_paths: int = 45,
    num_points_per_path: int = 800,
) -> Iterator[Any]:
    """Stream unbatched host NumPy scenarios from ``expert_path``.

    Wraps Waymax's own shuffled ``simulator_state_generator`` so a training loop
    can draw from an arbitrarily large TFRecord glob (a full WOMD split is
    ~500k scenarios) without ever materializing more than one batch at a time --
    the eager ``ScenarioSource`` above caches every requested scenario as a host
    NumPy pytree and is only viable for a few hundred to a few thousand of them.

    Default ``repeat=None`` loops forever (an SAC training stream). ``repeat=1``
    yields each scenario in the split exactly once then stops (a BC epoch).

    ``num_paths``/``num_points_per_path`` default to WOD 1.3.1's shape (see
    ``waymax.config.WOD_1_3_1_TRAINING``) -- with ``include_sdc_paths=True``
    Waymax's dataloader requires both to be set (it can't infer them from the
    TFRecord), and building a ``DatasetConfig`` from scratch instead of via
    ``dataclasses.replace(WOD_1_3_1_TRAINING, ...)`` (as ``ScenarioSource``
    does) drops them.
    """
    if distributed is None:
        distributed = repeat is None
    config = waymax_config.DatasetConfig(
        path=expert_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        num_paths=num_paths,
        num_points_per_path=num_points_per_path,
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


class StreamingScenarioSource:
    """A ``sample_scenario()``-only source backed by an infinite scenario stream.

    Unlike :class:`ScenarioSource`, nothing is cached: every ``sample_scenario``
    call pulls the next scenario off ``scenario_iter``. Waymax's dataloader does
    its own shuffling as it streams (``shuffle_seed``), so there is no notion of
    "index" here -- this exists for :class:`~rl.waymax_env.StreamingWaymaxGymEnv`,
    which resets by sampling rather than by index, the same way
    :class:`~rl.bc_core.MixedScenarioSource` does for the expert-stream half of
    BC-SAC's mix.
    """

    def __init__(self, scenario_iter: Iterator[Any], *, num_objects: int):
        self._iter = scenario_iter
        self.num_objects = int(num_objects)

    def sample_scenario(self, rng: np.random.Generator | None = None) -> Any:
        del rng  # shuffling happens inside the Waymax dataloader, not here
        return next(self._iter)


class CachedScenarioSource:
    """In-memory list of unbatched scenarios for ``WaymaxGymEnv``."""

    def __init__(
        self,
        scenarios: Sequence[Any],
        specs: Sequence[ScenarioSpec] | None = None,
        *,
        label: str = "cached",
    ):
        if not scenarios:
            raise ValueError("CachedScenarioSource requires at least one scenario.")
        self._scenarios = list(scenarios)
        if specs is not None:
            self._specs = list(specs)
        else:
            self._specs = [ScenarioSpec(label, i) for i in range(len(self._scenarios))]
        self.num_objects = int(
            np.asarray(self._scenarios[0].object_metadata.is_sdc).shape[-1]
        )

    def __len__(self) -> int:
        return len(self._scenarios)

    def get(self, index: int) -> Any:
        return self._scenarios[index % len(self._scenarios)]

    def spec(self, index: int) -> ScenarioSpec:
        return self._specs[index % len(self._specs)]

    def sample_index(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._scenarios)))


def _ensure_unbatched_scenario(state: Any) -> Any:
    """Return a single unbatched scenario (prefix shape ``()``).

    Waymax dataloaders with ``batch_dims=(1,)`` still prefix every leaf with a
  batch axis (``is_sdc`` shape ``[1, N]``). Slice it off before env reset / goal
    helpers that expect ``[N]``.
    """
    is_sdc = np.asarray(state.object_metadata.is_sdc)
    if is_sdc.ndim == 1:
        return state
    if is_sdc.ndim >= 2:
        batch_size = int(is_sdc.shape[0])
        if batch_size == 1:
            return _slice_batched_state(state, 0)
        raise ValueError(
            f"Expected unbatched scenario (is_sdc shape {is_sdc.shape}); "
            f"got batch_size={batch_size}."
        )
    raise ValueError(f"Unexpected is_sdc shape {is_sdc.shape}")


# Lazily jitted Waymax metric check (log-as-sim at a fixed timestep).
_SDC_VIOLATION_FN = None


def _sdc_log_violations_at(scen: Any, timestep: int) -> tuple[bool, bool]:
    """Whether the SDC is overlapped / offroad at ``timestep`` in the log.

    OffroadMetric / OverlapMetric read ``sim_trajectory``, so we point sim at
    the log (same trick as the diagnose_replay GT path). Compiled once.
    """
    global _SDC_VIOLATION_FN
    if _SDC_VIOLATION_FN is None:
        from waymax import metrics as waymax_metrics

        ov_m = waymax_metrics.OverlapMetric()
        off_m = waymax_metrics.OffroadMetric()

        def _fn(state, t):
            st = state.replace(sim_trajectory=state.log_trajectory, timestep=t)
            is_sdc = st.object_metadata.is_sdc
            ov = (ov_m.compute(st).value * is_sdc).sum()
            off = (off_m.compute(st).value * is_sdc).sum()
            return ov, off

        _SDC_VIOLATION_FN = jax.jit(_fn)

    import jax.numpy as jnp

    state = jax.tree_util.tree_map(jnp.asarray, scen)
    ov, off = _SDC_VIOLATION_FN(state, jnp.asarray(int(timestep), dtype=jnp.int32))
    return float(ov) > 0.5, float(off) > 0.5


class ScenarioSource:
    """Holds a list of cached, unbatched Waymax scenarios."""

    def __init__(
        self,
        specs: Sequence[ScenarioSpec],
        *,
        max_num_objects: int | None = None,
        include_sdc_paths: bool = True,
        verbose: bool = True,
        min_goal_distance_m: float | None = 3.0,
        drop_init_violations: bool = False,
    ):
        if not specs:
            raise ValueError("ScenarioSource requires at least one ScenarioSpec.")
        # ``None`` keeps all objects in the scene (WOMD = 128). Truncating risks
        # dropping the SDC, so it is opt-in.
        self.max_num_objects = None if max_num_objects is None else int(max_num_objects)
        self._include_sdc_paths = bool(include_sdc_paths)
        self._verbose = bool(verbose)
        # Drop scenarios where the ego is already within this radius of the
        # goal at the env's reset timestep: the episode terminates after one
        # step (goal "reached" immediately), so any offroad/overlap on that
        # single frame is a boundary artifact, not a real driving failure.
        # Set to ``None`` to disable.
        self._min_goal_distance_m = (
            None if min_goal_distance_m is None else float(min_goal_distance_m)
        )
        # Drop scenes where the SDC is already overlapped / offroad at the env
        # reset timestep (log positions). Those are unsolvable under clean-success
        # and would permanently cap an overfit unit's success rate. Opt-in
        # (overfit enables it by default) — the metric compile is non-trivial.
        self._drop_init_violations = bool(drop_init_violations)
        self._scenarios: list[Any] = []
        self._specs: list[ScenarioSpec] = []
        self.num_objects: int = 0  # actual object count of cached scenarios
        self._load(specs)

    def _dataset_config(self, tfrecord_path: str, num_indices: int) -> waymax_config.DatasetConfig:
        return dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=str(tfrecord_path),
            max_num_objects=self.max_num_objects,
            batch_dims=(int(num_indices),),
            include_sdc_paths=self._include_sdc_paths,
            shuffle_seed=None,
        )

    def _load(self, specs: Sequence[ScenarioSpec]) -> None:
        reset_t: int | None = None
        _compute_goal_xy = None
        if self._min_goal_distance_m is not None:
            from rl.waymax_env import _compute_goal_xy as _cgx  # local: avoid import cycle

            _compute_goal_xy = _cgx
            reset_t = waymax_config.EnvironmentConfig().init_steps - 1

        # Group requested indices by TFRecord so each file is scanned only once.
        by_file: dict[str, list[int]] = defaultdict(list)
        for spec in specs:
            by_file[spec.tfrecord].append(int(spec.scenario_idx))

        shard_errors: list[str] = []
        n_drop_goal = 0
        for tfrecord_path, indices in by_file.items():
            unique_indices = sorted(set(indices))
            cfg = self._dataset_config(tfrecord_path, len(unique_indices))
            try:
                state_batch, idx_to_batch = _load_scenario_state_batch_fast(cfg, unique_indices)
            except Exception as exc:  # noqa: BLE001 - surface and skip bad shards.
                msg = f"{tfrecord_path}: {type(exc).__name__}: {exc}"
                shard_errors.append(msg)
                # Always surface shard-load failures (a wholly skipped shard is an
                # error, not a per-scenario filter), even with verbose=False.
                print(f"[ScenarioSource] ERROR: failed to load shard {msg}", file=sys.stderr)
                continue

            for scenario_idx in unique_indices:
                batch_idx = idx_to_batch[scenario_idx]
                scen = _slice_batched_state(state_batch, batch_idx)
                # Keep only scenarios with exactly one SDC (required by the env).
                is_sdc = np.asarray(scen.object_metadata.is_sdc).astype(bool)
                if int(is_sdc.sum()) != 1:
                    if self._verbose:
                        print(
                            f"[ScenarioSource] Skipping {Path(tfrecord_path).name} "
                            f"scenario {scenario_idx}: SDC count={int(is_sdc.sum())}."
                        )
                    continue
                if reset_t is not None and _compute_goal_xy is not None:
                    ego = int(np.argmax(is_sdc))
                    ego_xy0 = np.array(
                        [
                            np.asarray(scen.log_trajectory.x)[ego, reset_t],
                            np.asarray(scen.log_trajectory.y)[ego, reset_t],
                        ],
                        dtype=np.float32,
                    )
                    goal_xy = _compute_goal_xy(scen)
                    dist0 = float(np.linalg.norm(ego_xy0 - goal_xy))
                    if dist0 <= self._min_goal_distance_m:
                        n_drop_goal += 1
                        if self._verbose:
                            print(
                                f"[ScenarioSource] Skipping {Path(tfrecord_path).name} "
                                f"scenario {scenario_idx}: already at goal at reset "
                                f"(dist={dist0:.2f}m)."
                            )
                        continue
                self._scenarios.append(scen)
                self._specs.append(ScenarioSpec(tfrecord_path, scenario_idx))

            if self._verbose:
                n_kept = sum(1 for s in self._specs if s.tfrecord == tfrecord_path)
                print(
                    f"[ScenarioSource] Loaded {n_kept}/{len(unique_indices)} scenarios "
                    f"from {Path(tfrecord_path).name}"
                )

        if not self._scenarios:
            detail = ""
            if shard_errors:
                detail = (
                    f" All {len(shard_errors)}/{len(by_file)} shard(s) failed to load:\n  "
                    + "\n  ".join(shard_errors)
                )
            raise RuntimeError(f"ScenarioSource loaded 0 scenarios; check paths/indices.{detail}")

        # Second pass: drop scenes already overlapped/offroad at reset. Done after
        # TF loading so we compile the metric fn once and avoid peak-memory overlap.
        n_drop_init = 0
        if self._drop_init_violations:
            init_t = waymax_config.EnvironmentConfig().init_steps - 1
            kept_scen: list[Any] = []
            kept_specs: list[ScenarioSpec] = []
            for scen, spec in zip(self._scenarios, self._specs):
                ov, off = _sdc_log_violations_at(scen, init_t)
                if ov or off:
                    n_drop_init += 1
                    if self._verbose:
                        why = "+".join(
                            p for p, b in (("overlap", ov), ("offroad", off)) if b
                        )
                        print(
                            f"[ScenarioSource] Skipping {Path(spec.tfrecord).name} "
                            f"scenario {spec.scenario_idx}: unsolvable at reset ({why})."
                        )
                    continue
                kept_scen.append(scen)
                kept_specs.append(spec)
            self._scenarios = kept_scen
            self._specs = kept_specs
            if not self._scenarios:
                raise RuntimeError(
                    "ScenarioSource loaded 0 scenarios after dropping init "
                    "overlap/offroad violations."
                )

        if self._verbose and (n_drop_goal or n_drop_init):
            print(
                f"[ScenarioSource] Filtered unsolvable: "
                f"already_at_goal={n_drop_goal}, init_overlap_or_offroad={n_drop_init}"
            )

        # Object count is fixed across WOMD scenarios; record it so the env can
        # configure itself to match exactly (avoids max_num_objects mismatch).
        object_counts = {
            int(np.asarray(s.object_metadata.is_sdc).shape[-1]) for s in self._scenarios
        }
        if len(object_counts) != 1:
            raise RuntimeError(
                f"Cached scenarios have inconsistent object counts: {sorted(object_counts)}."
            )
        self.num_objects = object_counts.pop()
        if self._verbose:
            print(
                f"[ScenarioSource] Total cached scenarios: {len(self._scenarios)} "
                f"| num_objects={self.num_objects}"
            )

    def __len__(self) -> int:
        return len(self._scenarios)

    def get(self, index: int) -> Any:
        """Returns the (host NumPy) unbatched scenario at ``index``."""
        return self._scenarios[index % len(self._scenarios)]

    def spec(self, index: int) -> ScenarioSpec:
        return self._specs[index % len(self._specs)]

    def sample_index(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._scenarios)))

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_failure_dir(
        cls,
        failure_dir: str,
        *,
        max_num_objects: int | None = None,
        failures_only: bool = True,
        limit: int | None = None,
        include_sdc_paths: bool = True,
        verbose: bool = True,
        min_goal_distance_m: float | None = 3.0,
        drop_init_violations: bool = False,
        scenario_idxs: Sequence[int] | None = None,
        tfrecord_substr: str | None = None,
    ) -> "ScenarioSource":
        """Builds a source from per-scenario JSON result files.

        Recognizes the ``success`` (goal-reaching) and ``task_result``
        (reward-search) flags; when ``failures_only`` is True, keeps only the
        scenarios where the policy failed. ``scenario_idxs`` restricts to those
        WOMD scenario indices when set; ``tfrecord_substr`` further requires the
        tfrecord path to contain that substring (disambiguate shared indices).
        """
        want = None if scenario_idxs is None else {int(i) for i in scenario_idxs}
        needle = None if not tfrecord_substr else str(tfrecord_substr)
        json_paths = sorted(glob(str(Path(failure_dir) / "**" / "*.json"), recursive=True))
        specs: list[ScenarioSpec] = []
        for jp in json_paths:
            if Path(jp).name == "summary.json":
                continue
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(record, dict):
                continue
            if "tfrecord" not in record or "scenario_idx" not in record:
                continue
            if failures_only and _is_success(record):
                continue
            if want is not None and int(record["scenario_idx"]) not in want:
                continue
            if needle is not None and needle not in str(record["tfrecord"]):
                continue
            specs.append(ScenarioSpec(str(record["tfrecord"]), int(record["scenario_idx"])))
            if limit is not None and len(specs) >= int(limit):
                break

        if not specs:
            raise RuntimeError(
                f"No matching scenarios found under {failure_dir} "
                f"(failures_only={failures_only}"
                + (f", scenario_idxs={sorted(want)}" if want is not None else "")
                + (f", tfrecord_substr={needle!r}" if needle is not None else "")
                + ")."
            )
        return cls(
            specs,
            max_num_objects=max_num_objects,
            include_sdc_paths=include_sdc_paths,
            verbose=verbose,
            min_goal_distance_m=min_goal_distance_m,
            drop_init_violations=drop_init_violations,
        )

    @classmethod
    def from_tfrecord(
        cls,
        tfrecord_path: str,
        indices: Sequence[int],
        *,
        max_num_objects: int | None = None,
        include_sdc_paths: bool = True,
        verbose: bool = True,
        min_goal_distance_m: float | None = 3.0,
        drop_init_violations: bool = False,
    ) -> "ScenarioSource":
        specs = [ScenarioSpec(str(tfrecord_path), int(i)) for i in indices]
        return cls(
            specs,
            max_num_objects=max_num_objects,
            include_sdc_paths=include_sdc_paths,
            verbose=verbose,
            min_goal_distance_m=min_goal_distance_m,
            drop_init_violations=drop_init_violations,
        )


def _is_success(record: dict[str, Any]) -> bool:
    """True if the record represents a successful run (so we drop it as a failure)."""
    if "success" in record:
        return bool(record["success"])
    if "task_result" in record:
        return bool(record["task_result"])
    # No known flag -> treat as not-a-failure so it is excluded by default.
    return True
