from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from data.types import PreprocessConfig
from data.preprocess_cached import preprocess_cached_npz
from data.utils import (
	resolve_cache_paths,
	load_offsets,
	build_instruction_paths,
	split_cache_paths,
	npz_to_torch
)


PROMPT = """
Given a driving scenario, propose a driving instruction and a subgoal that helps the ego vehicle reach the goal.
Only output the instruction and subgoal without any additional text, in the format:
Instruction: instruction text here. Subgoal: x,y
"""


@dataclass
class InstructionBatch:
	features: dict[str, torch.Tensor]
	aux: dict[str, torch.Tensor]
	prompts: list[str]
	answers: list[str]


@dataclass(frozen=True)
class InstructionCacheLoaderConfig:
	cache_paths: tuple[str, ...]
	instruction_dir: str | None = None
	preprocess_cfg: PreprocessConfig = PreprocessConfig()
	batch_size: int = 1
	shuffle_seed: int = 42
	prompt_text: str = PROMPT
	anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40)


def _read_instruction_at_offset(data_file: Path, offset: int) -> str:
	with data_file.open("rb") as handle:
		handle.seek(int(offset))
		raw = handle.readline()
	if not raw:
		raise ValueError(f"Missing instruction at offset {offset} in {data_file}")
	obj = json.loads(raw.decode("utf-8"))
	text = str(obj.get("instruction", "")).strip()
	if not text:
		raise ValueError(f"Empty instruction at offset {offset} in {data_file}")
	return text

def _infer_subgoal_from_features(features: dict[str, torch.Tensor], ego_range=100.0) -> list[str]:
	ego_trajectory = features['ego_trajectory'].cpu().detach().numpy()
	subgoal = ego_trajectory[:, -1, :2] * ego_range
	return [f"{x:.1f},{y:.1f}" for x, y in subgoal.tolist()]


def _load_instruction_for_scenario(instruction_dir: Path, cache_file: Path, scenario_index: int, anchor_step: int) -> str:
	data_path, index_path = build_instruction_paths(
		cache_file, str(instruction_dir), anchor_step=anchor_step
	)
	if not data_path.exists():
		raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")
	offsets = load_offsets(Path(index_path))
	line_index = int(scenario_index)
	if line_index < 0 or line_index >= len(offsets):
		raise IndexError(
			f"Scenario index {line_index} is out of range for {data_path} (have {len(offsets)} lines)"
		)
	return _read_instruction_at_offset(data_path, offsets[line_index])


class CacheInstructionDataset(IterableDataset[InstructionBatch]):
	def __init__(self, cfg: InstructionCacheLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def __iter__(self) -> Iterator[InstructionBatch]:
		worker_info = get_worker_info()
		cache_paths = list(self.cfg.cache_paths)
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
		if worker_info is not None:
			cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

		for anchor_step in self.cfg.anchor_steps:
			for cache_path in cache_paths:
				cache_file = Path(cache_path)
				preprocessed = preprocess_cached_npz(cache_file, cfg=self.cfg.preprocess_cfg, anchor_step_override=anchor_step)
				feature_keys = sorted(preprocessed["features"].keys())
				total_examples = int(preprocessed["features"][feature_keys[0]].shape[0])
				for start_index in range(0, total_examples, self.cfg.batch_size):
					end_index = min(start_index + self.cfg.batch_size, total_examples)
					batch_features = {
						key: npz_to_torch(value[start_index:end_index])
						for key, value in preprocessed["features"].items()
					}
					batch_aux = {
						key: npz_to_torch(value[start_index:end_index])
						for key, value in preprocessed["aux"].items()
					}

					scenario_indices = preprocessed["metadata"]["scenario_indices"]
					subgoal_texts = _infer_subgoal_from_features(batch_features)

					prompts: list[str] = []
					answers: list[str] = []
					for scenario_index, subgoal_text in zip(scenario_indices.tolist(), subgoal_texts):
						prompts.append(self.cfg.prompt_text)
						instruction = _load_instruction_for_scenario(
							Path(self.cfg.instruction_dir),
							cache_file,
							int(scenario_index),
							anchor_step=anchor_step
						)
						answers.append(
							f"Instruction: {instruction} Subgoal: {subgoal_text}"
						)

					yield InstructionBatch(
						features=batch_features,
						aux=batch_aux,
						prompts=prompts,
						answers=answers,
					 )
				

def build_inst_dataloader(
	cache_dir: str,
	preprocess_cfg,
	*,
	file_indices: list[int] | None = None,
	instruction_dir: str | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	num_workers: int = 0,
	pin_memory: bool = False,
	cache_paths: tuple[str, ...] | None = None,
	prompt_text: str = PROMPT,
) -> DataLoader[InstructionBatch]:
	if cache_paths is None:
		cache_paths = resolve_cache_paths(cache_dir, file_indices)
	cfg = InstructionCacheLoaderConfig(
		cache_paths=cache_paths,
		instruction_dir=instruction_dir,
		preprocess_cfg=preprocess_cfg,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		prompt_text=prompt_text,
	)
	dataset = CacheInstructionDataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

