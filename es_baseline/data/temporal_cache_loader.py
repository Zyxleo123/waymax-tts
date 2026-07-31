from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.preprocess_cached import preprocess_cached_npz
from data.types import CacheLoaderConfig, PreprocessConfig
from data.utils import (
    VLA_PROMPT,
    VLA_FUNCTION_PROMPT,
    build_question_bank,
    npz_to_jax,
    npz_to_torch,
    resolve_cache_paths,
)


_SIM_STATE_CACHE_PATTERN = re.compile(
    r"training_tfexample\.tfrecord-(\d+)-of-01000\.sim_state_cache\.npz$"
)


@dataclass
class TemporalCacheBatch:
    features: dict[str, Any]
    aux: dict[str, Any]
    timestep_valid: torch.Tensor
    history_timesteps: torch.Tensor
    prompts: list[list[str]] | None = None
    answers: list[list[str]] | None = None
    keys: list[list[str]] | None = None
    scenario_id: list[str] | None = None
    tfrecord_index: list[int] | None = None
    target_lane_ids: list[list[int]] | None = None


class TemporalCacheDataset(IterableDataset[TemporalCacheBatch]):
    def __init__(self, cfg: CacheLoaderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if self.cfg.backend == "torch":
            self.to_tensor = npz_to_torch
        elif self.cfg.backend == "jax":
            self.to_tensor = npz_to_jax
        else:
            raise ValueError(f"Unsupported backend: {self.cfg.backend}")

    def build_annotation_paths(self, cache_file: Path, anchor_step: int) -> tuple[Path, Path]:
        annotation_dir = Path(self.cfg.annotation_dir)
        stem = cache_file.name.removesuffix(".sim_state_cache.npz")
        return (
            annotation_dir / f"{stem}_t{anchor_step}.jsonl",
            annotation_dir / f"{stem}_t{anchor_step}.idx.json",
        )

    def load_byte_offsets(self, idx_path: Path) -> list[int]:
        with idx_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        if isinstance(index_data, list):
            return [int(value) for value in index_data]
        if isinstance(index_data, dict):
            if "byte_offsets" in index_data and isinstance(index_data["byte_offsets"], list):
                return [int(value) for value in index_data["byte_offsets"]]
            if len(index_data) == 1:
                value = next(iter(index_data.values()))
                if isinstance(value, list):
                    return [int(item) for item in value]
        raise ValueError(f"Unsupported annotation index format: {idx_path}")

    def load_annotation(self, cache_file: Path, scenario_index: int, anchor_step: int) -> dict[str, Any]:
        jsonl_path, idx_path = self.build_annotation_paths(cache_file, anchor_step)
        offsets = self.load_byte_offsets(idx_path)
        offset = offsets[int(scenario_index)]
        with jsonl_path.open("rb") as handle:
            handle.seek(int(offset))
            raw = handle.readline()
        obj = json.loads(raw.decode("utf-8"))
        annotation = obj["annotation"]
        annotation["instruction"] = str(obj.get("instruction", "")).strip()
        annotation['functions'] = str(obj.get("functions", "")).strip()
        return annotation

    def _parse_tfrecord_index(self, cache_file: Path) -> int:
        match = _SIM_STATE_CACHE_PATTERN.match(cache_file.name)
        if match is None:
            raise ValueError(f"Could not parse tfrecord index from cache file: {cache_file}")
        return int(match.group(1))

    def _stack_temporal_arrays(self, step_arrays: list[dict[str, np.ndarray]], start: int, end: int) -> dict[str, Any]:
        keys = list(step_arrays[0].keys())
        stacked: dict[str, Any] = {}
        for key in keys:
            stacked[key] = self.to_tensor(
                np.stack([step_arrays[t][key][start:end] for t in range(len(step_arrays))], axis=1)
            )
        return stacked

    def _transpose_nested(self, rows_by_step: list[list[str]]) -> list[list[str]]:
        if not rows_by_step:
            return []
        return [list(items) for items in zip(*rows_by_step, strict=False)]

    def __iter__(self) -> Iterator[TemporalCacheBatch]:
        worker_info = get_worker_info()
        cache_paths = list(self.cfg.cache_paths)
        if self.cfg.shuffle_seed:
            random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
        if worker_info is not None:
            cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

        for cache_path in cache_paths:
            cache_file = Path(cache_path)
            tfrecord_index = self._parse_tfrecord_index(cache_file)

            step_features: list[dict[str, np.ndarray]] = []
            step_aux: list[dict[str, np.ndarray]] = []
            scenario_indices: np.ndarray | None = None
            goal_step_override = np.random.randint(50, 91)
            for anchor_step in self.cfg.anchor_steps:
                preprocessed = preprocess_cached_npz(
                    cache_file,
                    cfg=self.cfg.preprocess_cfg,
                    anchor_step_override=anchor_step,
                    goal_step_override=goal_step_override,
                    include_inst_features=self.cfg.include_inst_features,
                )

                features = preprocessed["features"]
                aux = preprocessed["aux"]
                current_indices = np.asarray(preprocessed["metadata"]["scenario_indices"], dtype=np.int32)

                if scenario_indices is None:
                    scenario_indices = current_indices
                    if self.cfg.shuffle_seed:
                        rng = random.Random(int(self.cfg.shuffle_seed) + tfrecord_index)
                        perm = list(range(int(scenario_indices.shape[0])))
                        rng.shuffle(perm)
                        scenario_indices = scenario_indices[perm]
                        features = {key: value[perm] for key, value in features.items()}
                        aux = {key: value[perm] for key, value in aux.items()}
                else:
                    if int(current_indices.shape[0]) != int(scenario_indices.shape[0]):
                        raise RuntimeError(
                            f"Scenario count changed across anchor steps for {cache_file}: "
                            f"{current_indices.shape[0]} != {scenario_indices.shape[0]}"
                        )
                    if self.cfg.shuffle_seed:
                        features = {key: value[perm] for key, value in features.items()}
                        aux = {key: value[perm] for key, value in aux.items()}

                step_features.append(features)
                step_aux.append(aux)

            if scenario_indices is None:
                continue

            total_examples = int(scenario_indices.shape[0])
            tfrecord_ids = [tfrecord_index] * total_examples

            for start_index in range(0, total_examples, self.cfg.batch_size):
                end_index = min(start_index + self.cfg.batch_size, total_examples)
                if self.cfg.drop_last and end_index == total_examples:
                    continue

                batch_features = self._stack_temporal_arrays(step_features, start_index, end_index)
                batch_aux = self._stack_temporal_arrays(step_aux, start_index, end_index)

                history_timesteps = torch.tensor(list(self.cfg.anchor_steps), dtype=torch.long)
                history_timesteps = history_timesteps.unsqueeze(0).repeat(end_index - start_index, 1)
                timestep_valid = torch.ones_like(history_timesteps, dtype=torch.bool)

                if self.cfg.language_label is not None:
                    prompts_by_step: list[list[str]] = []
                    answers_by_step: list[list[str]] = []
                    keys_by_step: list[list[str]] = []
                    target_lane_ids_by_step: list[list[int]] = []

                    for anchor_step in self.cfg.anchor_steps:
                        step_prompts: list[str] = []
                        step_answers: list[str] = []
                        step_keys: list[str] = []
                        step_scenario_indices = scenario_indices[start_index:end_index]
                        step_target_lane_ids: list[int] = []

                        for scenario_index in step_scenario_indices:
                            annotation = self.load_annotation(cache_file, int(scenario_index), int(anchor_step))
                            if self.cfg.language_label == "qa":
                                qa = build_question_bank(annotation)
                                for item in qa:
                                    step_prompts.append(item["question"])
                                    step_answers.append(item["answer"])
                                    step_keys.append(item["key"])
                            elif self.cfg.language_label == "instruction":
                                step_prompts.append(VLA_PROMPT)
                                step_answers.append(f"{annotation['instruction']}")
                            elif self.cfg.language_label == "function":
                                step_prompts.append(VLA_FUNCTION_PROMPT)
                                step_answers.append(f"{annotation['functions']}")
                            else:
                                raise ValueError(
                                    f"Unsupported language_label: {self.cfg.language_label}"
                                )
                            step_target_lane_ids.append(int(annotation.get("target_lane_id", -1)))
                        prompts_by_step.append(step_prompts)
                        answers_by_step.append(step_answers)
                        target_lane_ids_by_step.append(step_target_lane_ids)
                        if self.cfg.language_label == "qa":
                            keys_by_step.append(step_keys)

                    prompts = self._transpose_nested(prompts_by_step)
                    answers = self._transpose_nested(answers_by_step)
                    keys = self._transpose_nested(keys_by_step) if self.cfg.language_label == "qa" else None
                    target_lane_ids = self._transpose_nested(target_lane_ids_by_step)
                else:
                    prompts = None
                    answers = None
                    keys = None
                    target_lane_ids = None

                scenario_id = [
                    f"{tfrecord_index:05d}_{int(scenario_index)}"
                    for scenario_index in scenario_indices[start_index:end_index]
                ]

                yield TemporalCacheBatch(
                    features=batch_features,
                    aux=batch_aux,
                    timestep_valid=timestep_valid,
                    history_timesteps=history_timesteps,
                    prompts=prompts,
                    answers=answers,
                    keys=keys,
                    target_lane_ids=target_lane_ids,
                    scenario_id=scenario_id,
                    tfrecord_index=tfrecord_ids[start_index:end_index],
                )


def build_temporal_dataloader(
    cache_dir: str,
    preprocess_cfg: PreprocessConfig,
    annotation_dir: str | None = None,
    anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80),
    language_label: str | None = None,
    backend: str = "torch",
    include_inst_features: bool = False,
    *,
    file_indices: list[int] | None = None,
    batch_size: int = 1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_paths: tuple[str, ...] | None = None,
    drop_last: bool = False,
) -> DataLoader[TemporalCacheBatch]:
    if cache_paths is None:
        cache_paths = resolve_cache_paths(cache_dir, file_indices)
    cfg = CacheLoaderConfig(
        cache_paths=cache_paths,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        preprocess_cfg=preprocess_cfg,
        anchor_steps=anchor_steps,
        language_label=language_label,
        backend=backend,
        include_inst_features=include_inst_features,
        drop_last=drop_last,
    )
    dataset = TemporalCacheDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataloader(
    cache_dir: str,
    preprocess_cfg: PreprocessConfig,
    annotation_dir: str | None = None,
    anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80),
    language_label: str | None = None,
    backend: str = "torch",
    include_inst_features: bool = False,
    *,
    file_indices: list[int] | None = None,
    batch_size: int = 1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_paths: tuple[str, ...] | None = None,
    drop_last: bool = False,
) -> DataLoader[TemporalCacheBatch]:
    return build_temporal_dataloader(
        cache_dir=cache_dir,
        preprocess_cfg=preprocess_cfg,
        annotation_dir=annotation_dir,
        anchor_steps=anchor_steps,
        language_label=language_label,
        backend=backend,
        include_inst_features=include_inst_features,
        file_indices=file_indices,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        cache_paths=cache_paths,
        drop_last=drop_last,
    )