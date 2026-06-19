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
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
from waymax import config as waymax_config

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


class ScenarioSource:
    """Holds a list of cached, unbatched Waymax scenarios."""

    def __init__(
        self,
        specs: Sequence[ScenarioSpec],
        *,
        max_num_objects: int | None = None,
        include_sdc_paths: bool = False,
        verbose: bool = True,
    ):
        if not specs:
            raise ValueError("ScenarioSource requires at least one ScenarioSpec.")
        # ``None`` keeps all objects in the scene (WOMD = 128). Truncating risks
        # dropping the SDC, so it is opt-in.
        self.max_num_objects = None if max_num_objects is None else int(max_num_objects)
        self._include_sdc_paths = bool(include_sdc_paths)
        self._verbose = bool(verbose)
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
        # Group requested indices by TFRecord so each file is scanned only once.
        by_file: dict[str, list[int]] = defaultdict(list)
        for spec in specs:
            by_file[spec.tfrecord].append(int(spec.scenario_idx))

        for tfrecord_path, indices in by_file.items():
            unique_indices = sorted(set(indices))
            cfg = self._dataset_config(tfrecord_path, len(unique_indices))
            try:
                state_batch, idx_to_batch = _load_scenario_state_batch_fast(cfg, unique_indices)
            except Exception as exc:  # noqa: BLE001 - surface and skip bad shards.
                if self._verbose:
                    print(f"[ScenarioSource] Skipping {tfrecord_path}: {exc}")
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
                self._scenarios.append(scen)
                self._specs.append(ScenarioSpec(tfrecord_path, scenario_idx))

            if self._verbose:
                print(
                    f"[ScenarioSource] Loaded {len(unique_indices)} scenarios "
                    f"from {Path(tfrecord_path).name}"
                )

        if not self._scenarios:
            raise RuntimeError("ScenarioSource loaded 0 scenarios; check paths/indices.")

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
        include_sdc_paths: bool = False,
        verbose: bool = True,
    ) -> "ScenarioSource":
        """Builds a source from per-scenario JSON result files.

        Recognizes the ``success`` (goal-reaching) and ``task_result``
        (reward-search) flags; when ``failures_only`` is True, keeps only the
        scenarios where the policy failed.
        """
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
            specs.append(ScenarioSpec(str(record["tfrecord"]), int(record["scenario_idx"])))
            if limit is not None and len(specs) >= int(limit):
                break

        if not specs:
            raise RuntimeError(
                f"No matching scenarios found under {failure_dir} "
                f"(failures_only={failures_only})."
            )
        return cls(
            specs,
            max_num_objects=max_num_objects,
            include_sdc_paths=include_sdc_paths,
            verbose=verbose,
        )

    @classmethod
    def from_tfrecord(
        cls,
        tfrecord_path: str,
        indices: Sequence[int],
        *,
        max_num_objects: int | None = None,
        include_sdc_paths: bool = False,
        verbose: bool = True,
    ) -> "ScenarioSource":
        specs = [ScenarioSpec(str(tfrecord_path), int(i)) for i in indices]
        return cls(
            specs,
            max_num_objects=max_num_objects,
            include_sdc_paths=include_sdc_paths,
            verbose=verbose,
        )


def _is_success(record: dict[str, Any]) -> bool:
    """True if the record represents a successful run (so we drop it as a failure)."""
    if "success" in record:
        return bool(record["success"])
    if "task_result" in record:
        return bool(record["task_result"])
    # No known flag -> treat as not-a-failure so it is excluded by default.
    return True
