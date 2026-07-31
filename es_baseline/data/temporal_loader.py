from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vla.bev_qa.instruction_parse import (
    DIRECTION_BUCKETS,
    direction_bucket,
    normalize_instruction,
)
from vla.temporal_vla.npz_cache import NpzShardCache
from vla.temporal_vla.scene_features import (
    DEFAULT_GOAL_STEP,
    SCENE_FEATURE_KEYS,
    build_scene_features_for_timesteps,
)

DEFAULT_NPZ_CACHE_SIZE = 2

_INDEX_CACHE_KEYS: tuple[str, ...] = (
    "inst_timesteps",
    "inst_valid",
    "scenario_indices",
)

DEFAULT_INSTRUCTION_DATA_DIR = "/zfsauton/scratch/mineuih/waymax_rs/annotations"
DEFAULT_SCENE_FEATURE_CACHE_DIR = (
    "/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz"
)
DEFAULT_OUTPUT_DIR = "/zfsauton/scratch/sbellad/temporal_output"
DEFAULT_CACHE_TIMESTEPS: tuple[int, ...] | None = None
DEFAULT_ALLOWED_TIMESTEPS = DEFAULT_CACHE_TIMESTEPS

SPEED_AUX_BUCKETS: tuple[str, ...] = (
    "accelerating",
    "slowing_down",
    "constant",
    "stop",
    "unknown",
)

_JSONL_PATTERN = re.compile(
    r"training_tfexample\.tfrecord-(\d+)-of-01000_t(\d+)\.jsonl$"
)
_SIM_STATE_CACHE_PATTERN = re.compile(
    r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$"
)

_RAW_SCENE_CACHE_KEYS: tuple[str, ...] = (
    "inst_timesteps",
    "inst_valid",
    "scenario_indices",
    "log_trajectory_x",
    "log_trajectory_y",
    "log_trajectory_yaw",
    "log_trajectory_vel_x",
    "log_trajectory_vel_y",
    "log_trajectory_speed",
    "log_trajectory_valid",
    "log_trajectory_timestamp_micros",
    "log_trajectory_length",
    "log_trajectory_width",
    "object_metadata_is_sdc",
    "object_metadata_object_types",
    "roadgraph_points_x",
    "roadgraph_points_y",
    "roadgraph_points_types",
    "roadgraph_points_valid",
    "roadgraph_points_dir_x",
    "roadgraph_points_dir_y",
    "roadgraph_points_ids",
    "log_traffic_light_state",
    "log_traffic_light_x",
    "log_traffic_light_y",
    "log_traffic_light_valid",
)


def speed_aux_bucket(text: str) -> str:
    normalized = normalize_instruction(text)
    if not normalized:
        return "unknown"
    if normalized == "stop" or normalized.startswith("stop "):
        return "stop"
    if "accelerat" in normalized:
        return "accelerating"
    if (
        "slowing down" in normalized
        or "slow down" in normalized
        or "slowing" in normalized
    ):
        return "slowing_down"
    return "constant"


def speed_aux_label(text: str) -> int:
    return SPEED_AUX_BUCKETS.index(speed_aux_bucket(text))


def direction_aux_label(text: str) -> int:
    return DIRECTION_BUCKETS.index(direction_bucket(text))


@dataclass(frozen=True)
class ScenarioKey:
    tfrecord_index: int
    scenario_index: int

    @property
    def scenario_id(self) -> str:
        return f"{self.tfrecord_index:05d}_{self.scenario_index}"


@dataclass(frozen=True)
class TemporalScenarioIndex:
    """One training sample = full valid timestep sequence for a scenario."""

    scenario: ScenarioKey
    scenario_timesteps: tuple[int, ...]


@dataclass
class TemporalScenarioBatch:
    features: dict[str, torch.Tensor]
    timestep_valid: torch.Tensor
    history_timesteps: torch.Tensor
    instruction_valid: torch.Tensor
    prompt: list[str]
    instruction: list[list[str]]
    scenario_id: list[str]
    tfrecord_index: list[int]
    direction_label: torch.Tensor
    speed_label: torch.Tensor


def _parse_jsonl_path(path: Path) -> tuple[int, int]:
    match = _JSONL_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse tfrecord/timestep from jsonl path: {path}")
    return int(match.group(1)), int(match.group(2))


def _parse_sim_state_cache_path(path: Path) -> int:
    match = _SIM_STATE_CACHE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"Could not parse tfrecord index from sim_state_cache path: {path}"
        )
    return int(match.group(1))


def _sim_state_cache_path(cache_dir: Path, tfrecord_index: int) -> Path:
    return (
        cache_dir
        / f"training_tfexample.tfrecord-{tfrecord_index:05d}-of-01000.sim_state_cache.npz"
    )


def _instruction_jsonl_path(
    instruction_dir: Path,
    tfrecord_index: int,
    timestep: int,
) -> Path:
    return (
        instruction_dir
        / f"training_tfexample.tfrecord-{tfrecord_index:05d}-of-01000_t{timestep}.jsonl"
    )


def _resolve_sim_state_cache_paths(
    cache_dir: Path,
    *,
    file_indices: list[int] | None,
) -> list[Path]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"scene_feature_cache_dir does not exist: {cache_dir}")

    paths: list[Path] = []
    for path in sorted(cache_dir.glob("training_tfexample.tfrecord-*-of-01000.sim_state_cache.npz")):
        match = _SIM_STATE_CACHE_PATTERN.match(path.name)
        if match is None:
            continue
        tfrecord_index = int(match.group(1))
        if file_indices is not None and tfrecord_index not in file_indices:
            continue
        paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No sim_state_cache npz files found in {cache_dir}"
            + (f" for file_indices={file_indices}" if file_indices else "")
        )
    return paths


@lru_cache(maxsize=8)
def _load_sim_state_cache_metadata(scene_cache_path: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with np.load(scene_cache_path, allow_pickle=False) as npz_data:
        inst_timesteps = tuple(int(x) for x in npz_data["inst_timesteps"])
        scenario_indices = tuple(int(x) for x in npz_data["scenario_indices"])
    return inst_timesteps, scenario_indices


@lru_cache(maxsize=32)
def _load_sim_state_index_arrays(scene_cache_path: str) -> dict[str, np.ndarray]:
    """Lightweight shard metadata for dataset index construction only."""
    with np.load(scene_cache_path, allow_pickle=False) as npz_data:
        missing = [key for key in _INDEX_CACHE_KEYS if key not in npz_data.files]
        if missing:
            raise KeyError(
                f"Missing index arrays in {scene_cache_path}: {missing}"
            )
        return {key: np.asarray(npz_data[key]) for key in _INDEX_CACHE_KEYS}


def _load_sim_state_cache_arrays(scene_cache_path: str) -> dict[str, np.ndarray]:
    """Load full raw scene arrays for feature preprocessing (heavy)."""
    with np.load(scene_cache_path, allow_pickle=False) as npz_data:
        missing = [key for key in _RAW_SCENE_CACHE_KEYS if key not in npz_data.files]
        if missing:
            raise KeyError(
                f"Missing raw scene arrays in {scene_cache_path}: {missing}"
            )
        return {key: np.asarray(npz_data[key]) for key in _RAW_SCENE_CACHE_KEYS}


def discover_cache_timesteps(
    scene_feature_cache_dir: str | Path,
    *,
    file_indices: list[int] | None = None,
) -> tuple[int, ...]:
    """Return sorted timesteps present in sim_state_cache inst_timesteps."""
    cache_paths = _resolve_sim_state_cache_paths(
        Path(scene_feature_cache_dir),
        file_indices=file_indices,
    )
    timesteps: set[int] = set()
    for cache_path in cache_paths:
        inst_timesteps, _ = _load_sim_state_cache_metadata(str(cache_path))
        timesteps.update(inst_timesteps)
    if not timesteps:
        raise FileNotFoundError(
            f"No inst_timesteps found in {scene_feature_cache_dir}"
            + (f" for file_indices={file_indices}" if file_indices else "")
        )
    return tuple(sorted(timesteps))


def _timestep_to_index(
    inst_timesteps: np.ndarray | tuple[int, ...],
    timestep: int,
) -> int:
    matches = np.where(np.asarray(inst_timesteps) == int(timestep))[0]
    if len(matches) == 0:
        raise KeyError(f"Timestep {timestep} not found in inst_timesteps={inst_timesteps}")
    return int(matches[0])


def _scenario_row_index(
    scenario_indices: np.ndarray,
    scenario_index: int,
) -> int:
    rows = np.where(scenario_indices == int(scenario_index))[0]
    if len(rows) == 0:
        raise KeyError(f"scenario_index={scenario_index} not found in cache")
    return int(rows[0])


@lru_cache(maxsize=128)
def _load_instruction_records(jsonl_path: str) -> dict[int, str]:
    path = Path(jsonl_path)
    records: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scenario_index = int(record["scenario_index"])
            instruction = str(record.get("instruction", "")).strip()
            if instruction:
                records[scenario_index] = instruction
    return records


def build_temporal_scenario_indices(
    *,
    instruction_data_dir: str | Path,
    scene_feature_cache_dir: str | Path,
    file_indices: list[int] | None = None,
    timesteps: tuple[int, ...] | None = None,
    max_samples: int | None = None,
) -> list[TemporalScenarioIndex]:
    """
    Build one sample per scenario with all valid instruction timesteps.

    inst_valid is used only to decide which timesteps exist; inst_features are never
    loaded. Instructions come from annotation jsonl files (labels only).
    """
    instruction_dir = Path(instruction_data_dir)
    cache_dir = Path(scene_feature_cache_dir)

    if timesteps is None:
        timesteps = discover_cache_timesteps(cache_dir, file_indices=file_indices)
    else:
        timesteps = tuple(int(t) for t in timesteps)

    cache_paths = _resolve_sim_state_cache_paths(cache_dir, file_indices=file_indices)

    samples: list[TemporalScenarioIndex] = []
    skipped_missing_instruction = 0
    skipped_invalid_cache = 0
    instruction_files_loaded = 0

    for cache_path in cache_paths:
        tfrecord_index = _parse_sim_state_cache_path(cache_path)
        arrays = _load_sim_state_index_arrays(str(cache_path))
        inst_timesteps = arrays["inst_timesteps"]
        inst_valid = arrays["inst_valid"]
        scenario_indices = arrays["scenario_indices"]

        instruction_records_by_timestep: dict[int, dict[int, str]] = {}
        for timestep in timesteps:
            try:
                timestep_idx = _timestep_to_index(inst_timesteps, timestep)
            except KeyError:
                continue

            jsonl_path = _instruction_jsonl_path(
                instruction_dir,
                tfrecord_index,
                timestep,
            )
            if not jsonl_path.exists():
                skipped_missing_instruction += int(inst_valid[:, timestep_idx].sum())
                continue

            instruction_records_by_timestep[timestep] = _load_instruction_records(
                str(jsonl_path)
            )
            instruction_files_loaded += 1

        for row, scenario_index in enumerate(scenario_indices):
            scenario = ScenarioKey(tfrecord_index, int(scenario_index))
            scenario_timesteps: list[int] = []

            for timestep, instruction_records in instruction_records_by_timestep.items():
                timestep_idx = _timestep_to_index(inst_timesteps, timestep)
                if not bool(inst_valid[row, timestep_idx]):
                    skipped_invalid_cache += 1
                    continue

                instruction = instruction_records.get(int(scenario_index))
                if not instruction:
                    skipped_missing_instruction += 1
                    continue

                scenario_timesteps.append(int(timestep))

            if scenario_timesteps:
                samples.append(
                    TemporalScenarioIndex(
                        scenario=scenario,
                        scenario_timesteps=tuple(sorted(scenario_timesteps)),
                    )
                )

    if max_samples is not None:
        samples = samples[: int(max_samples)]

    if not samples:
        raise RuntimeError(
            "No temporal scenario samples found. "
            f"instruction_data_dir={instruction_data_dir}, "
            f"scene_feature_cache_dir={scene_feature_cache_dir}, "
            f"timesteps={timesteps}, "
            f"cache_files={len(cache_paths)}, "
            f"instruction_files_loaded={instruction_files_loaded}, "
            f"skipped_missing_instruction={skipped_missing_instruction}, "
            f"skipped_invalid_cache={skipped_invalid_cache}"
        )
    return samples


# Backwards-compatible alias used by older scripts.
build_temporal_sample_indices = build_temporal_scenario_indices


def summarize_temporal_scenarios(
    samples: list[TemporalScenarioIndex],
) -> dict[str, Any]:
    """Return counts useful for startup logging."""
    timestep_counts: dict[int, int] = {}
    lengths: dict[int, int] = {}
    for sample in samples:
        length = len(sample.scenario_timesteps)
        lengths[length] = lengths.get(length, 0) + 1
        for timestep in sample.scenario_timesteps:
            timestep_counts[timestep] = timestep_counts.get(timestep, 0) + 1
    return {
        "num_scenarios": len(samples),
        "by_timestep_presence": dict(sorted(timestep_counts.items())),
        "by_sequence_length": dict(sorted(lengths.items())),
    }


summarize_temporal_samples = summarize_temporal_scenarios


def split_scenario_indices(
    samples: list[TemporalScenarioIndex],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Scenario-level train/val split (one index per scenario)."""
    scenario_keys = [sample.scenario for sample in samples]
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n_val = int(round(len(indices) * validation_fraction))
    if validation_fraction > 0 and len(indices) > 1:
        n_val = max(1, n_val)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def verify_no_scenario_overlap(
    samples: list[TemporalScenarioIndex],
    train_indices: list[int],
    val_indices: list[int],
) -> None:
    train_scenarios = {samples[idx].scenario for idx in train_indices}
    val_scenarios = {samples[idx].scenario for idx in val_indices}
    overlap = train_scenarios.intersection(val_scenarios)
    if overlap:
        raise RuntimeError(
            f"Train/val scenario overlap detected: {sorted(overlap, key=lambda s: s.scenario_id)}"
        )


def _zero_padded_scene_features(max_history_steps: int) -> dict[str, torch.Tensor]:
    return {
        "ego_state": torch.zeros(max_history_steps, 5, dtype=torch.float32),
        "goal_xy": torch.zeros(max_history_steps, 2, dtype=torch.float32),
        "remaining_timesteps": torch.zeros(max_history_steps, 1, dtype=torch.float32),
        "other_states": torch.zeros(max_history_steps, 128, 15, dtype=torch.float32),
        "other_valid": torch.zeros(max_history_steps, 128, dtype=torch.bool),
        "map_features": torch.zeros(max_history_steps, 128, 128, 25, dtype=torch.float32),
        "map_valid": torch.zeros(max_history_steps, 128, 128, dtype=torch.bool),
        "traffic_light_features": torch.zeros(max_history_steps, 16, 9, dtype=torch.float32),
        "traffic_light_valid": torch.zeros(max_history_steps, 16, dtype=torch.bool),
    }


class TemporalScenarioDataset(Dataset):
    def __init__(
        self,
        samples: list[TemporalScenarioIndex],
        *,
        instruction_data_dir: str | Path,
        scene_feature_cache_dir: str | Path,
        max_history_steps: int = 9,
        goal_step: int = DEFAULT_GOAL_STEP,
        npz_cache_size: int = DEFAULT_NPZ_CACHE_SIZE,
        prompt: str,
    ) -> None:
        self.samples = samples
        self.instruction_data_dir = Path(instruction_data_dir)
        self.scene_cache_dir = Path(scene_feature_cache_dir)
        self.max_history_steps = int(max_history_steps)
        self.goal_step = int(goal_step)
        self.prompt = prompt

        self._instruction_cache: dict[tuple[int, int, int], str] = {}
        self._npz_cache = NpzShardCache(maxsize=int(npz_cache_size))

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_arrays(self, cache_path: str) -> dict[str, np.ndarray]:
        return self._npz_cache.get(cache_path, _load_sim_state_cache_arrays)

    def _load_instruction(
        self,
        tfrecord_index: int,
        scenario_index: int,
        timestep: int,
    ) -> str:
        key = (tfrecord_index, scenario_index, timestep)
        if key in self._instruction_cache:
            return self._instruction_cache[key]

        jsonl_path = _instruction_jsonl_path(
            self.instruction_data_dir,
            tfrecord_index,
            timestep,
        )
        records = _load_instruction_records(str(jsonl_path))
        instruction = records[scenario_index]
        self._instruction_cache[key] = instruction
        return instruction

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        scenario = sample.scenario
        cache_path = str(
            _sim_state_cache_path(
                self.scene_cache_dir,
                scenario.tfrecord_index,
            )
        )
        arrays = self._cache_arrays(cache_path)
        row = _scenario_row_index(arrays["scenario_indices"], scenario.scenario_index)

        num_timesteps = len(sample.scenario_timesteps)
        if num_timesteps > self.max_history_steps:
            raise ValueError(
                f"Scenario {scenario.scenario_id} has {num_timesteps} timesteps, "
                f"exceeding max_history_steps={self.max_history_steps}"
            )

        step_features = build_scene_features_for_timesteps(
            arrays,
            row,
            sample.scenario_timesteps,
            goal_step=self.goal_step,
        )

        features = _zero_padded_scene_features(self.max_history_steps)
        for key in SCENE_FEATURE_KEYS:
            features[key][:num_timesteps] = step_features[key]

        instructions: list[str] = []
        direction_labels: list[int] = []
        speed_labels: list[int] = []
        for timestep in sample.scenario_timesteps:
            instruction = self._load_instruction(
                scenario.tfrecord_index,
                scenario.scenario_index,
                timestep,
            )
            instructions.append(instruction)
            direction_labels.append(direction_aux_label(instruction))
            speed_labels.append(speed_aux_label(instruction))

        padded_instructions = instructions + [""] * (self.max_history_steps - num_timesteps)
        padded_timesteps = list(sample.scenario_timesteps) + [-1] * (
            self.max_history_steps - num_timesteps
        )
        timestep_valid = [True] * num_timesteps + [False] * (
            self.max_history_steps - num_timesteps
        )
        instruction_valid = timestep_valid.copy()

        return {
            "features": features,
            "timestep_valid": torch.tensor(timestep_valid, dtype=torch.bool),
            "instruction_valid": torch.tensor(instruction_valid, dtype=torch.bool),
            "history_timesteps": torch.tensor(padded_timesteps, dtype=torch.long),
            "prompt": self.prompt,
            "instruction": padded_instructions,
            "scenario_id": scenario.scenario_id,
            "tfrecord_index": scenario.tfrecord_index,
            "direction_label": torch.tensor(
                direction_labels + [0] * (self.max_history_steps - num_timesteps),
                dtype=torch.long,
            ),
            "speed_label": torch.tensor(
                speed_labels + [0] * (self.max_history_steps - num_timesteps),
                dtype=torch.long,
            ),
        }


TemporalInstructionDataset = TemporalScenarioDataset


def temporal_collate_fn(batch: list[dict[str, Any]]) -> TemporalScenarioBatch:
    feature_keys = list(batch[0]["features"].keys())
    features = {
        key: torch.stack([item["features"][key] for item in batch], dim=0)
        for key in feature_keys
    }
    return TemporalScenarioBatch(
        features=features,
        timestep_valid=torch.stack([item["timestep_valid"] for item in batch], dim=0),
        instruction_valid=torch.stack([item["instruction_valid"] for item in batch], dim=0),
        history_timesteps=torch.stack([item["history_timesteps"] for item in batch], dim=0),
        prompt=[item["prompt"] for item in batch],
        instruction=[item["instruction"] for item in batch],
        scenario_id=[item["scenario_id"] for item in batch],
        tfrecord_index=[item["tfrecord_index"] for item in batch],
        direction_label=torch.stack([item["direction_label"] for item in batch], dim=0),
        speed_label=torch.stack([item["speed_label"] for item in batch], dim=0),
    )


def build_temporal_dataloader(
    *,
    instruction_data_dir: str | Path,
    scene_feature_cache_dir: str | Path,
    prompt: str,
    file_indices: list[int] | None = None,
    timesteps: tuple[int, ...] | None = None,
    max_history_steps: int = 9,
    goal_step: int = DEFAULT_GOAL_STEP,
    npz_cache_size: int = DEFAULT_NPZ_CACHE_SIZE,
    batch_size: int = 1,
    shuffle: bool = True,
    shuffle_seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_samples: int | None = None,
    sample_indices: list[int] | None = None,
    dataset: Dataset | None = None,
) -> DataLoader:
    if dataset is None:
        all_samples = build_temporal_scenario_indices(
            instruction_data_dir=instruction_data_dir,
            scene_feature_cache_dir=scene_feature_cache_dir,
            file_indices=file_indices,
            timesteps=timesteps,
            max_samples=None,
        )
        if sample_indices is not None:
            selected = [all_samples[idx] for idx in sample_indices]
            dataset = TemporalScenarioDataset(
                selected,
                instruction_data_dir=instruction_data_dir,
                scene_feature_cache_dir=scene_feature_cache_dir,
                max_history_steps=max_history_steps,
                goal_step=goal_step,
                npz_cache_size=npz_cache_size,
                prompt=prompt,
            )
        else:
            dataset = TemporalScenarioDataset(
                all_samples if max_samples is None else all_samples[: int(max_samples)],
                instruction_data_dir=instruction_data_dir,
                scene_feature_cache_dir=scene_feature_cache_dir,
                max_history_steps=max_history_steps,
                goal_step=goal_step,
                npz_cache_size=npz_cache_size,
                prompt=prompt,
            )

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=temporal_collate_fn,
    )


def _parse_file_indices(raw_file_indices: list[str] | None) -> list[int] | None:
    if raw_file_indices is None:
        return None
    parsed: list[int] = []
    for token in raw_file_indices:
        for part in token.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed or None


def _parse_timesteps(raw_timesteps: list[str] | None) -> tuple[int, ...] | None:
    if raw_timesteps is None:
        return None
    parsed: list[int] = []
    for token in raw_timesteps:
        for part in token.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return tuple(parsed) if parsed else None


def _smoke_test_main() -> None:
    parser = argparse.ArgumentParser(description="Temporal VLA dataloader smoke test.")
    parser.add_argument(
        "--instruction_data_dir",
        type=str,
        default=DEFAULT_INSTRUCTION_DATA_DIR,
    )
    parser.add_argument(
        "--scene_feature_cache_dir",
        type=str,
        default=DEFAULT_SCENE_FEATURE_CACHE_DIR,
    )
    parser.add_argument("--file_indices", type=str, nargs="*", default=None)
    parser.add_argument("--max_history_steps", type=int, default=9)
    parser.add_argument("--goal_step", type=int, default=DEFAULT_GOAL_STEP)
    parser.add_argument("--npz_cache_size", type=int, default=DEFAULT_NPZ_CACHE_SIZE)
    parser.add_argument("--timesteps", type=str, nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    from vla.temporal_vla.model import TEMPORAL_INSTRUCTION_PROMPT

    timesteps = _parse_timesteps(args.timesteps)
    if timesteps is None:
        timesteps = discover_cache_timesteps(args.scene_feature_cache_dir)
    print(f"using_timesteps={list(timesteps)}")
    loader = build_temporal_dataloader(
        instruction_data_dir=args.instruction_data_dir,
        scene_feature_cache_dir=args.scene_feature_cache_dir,
        prompt=TEMPORAL_INSTRUCTION_PROMPT,
        file_indices=_parse_file_indices(args.file_indices),
        timesteps=timesteps,
        max_history_steps=args.max_history_steps,
        goal_step=args.goal_step,
        npz_cache_size=args.npz_cache_size,
        batch_size=args.batch_size,
        shuffle=False,
        max_samples=args.max_samples,
    )
    batch = next(iter(loader))
    print(f"dataset_size={len(loader.dataset)}")
    print(f"features.ego_state={tuple(batch.features['ego_state'].shape)}")
    print(f"timestep_valid={tuple(batch.timestep_valid.shape)}")
    print(f"instruction_valid={tuple(batch.instruction_valid.shape)}")
    print(f"history_timesteps={batch.history_timesteps.tolist()}")
    print(f"num_valid_timesteps={int(batch.timestep_valid[0].sum())}")
    print(f"instruction[0][0]={batch.instruction[0][0]!r}")


if __name__ == "__main__":
    _smoke_test_main()