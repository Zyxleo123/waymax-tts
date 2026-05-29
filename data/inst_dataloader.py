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


DRIVING_INSTRUCTION_PROMPT = """
Propose a driving instruction that helps the ego vehicle reach the goal in one sentence.
Only output the instruction without any additional text.
""".strip()
SUBGOAL_PROMPT = """
For a given driving instruction, propose a subgoal that follows the instruction and helps the ego vehicle reach the goal.
Only output the position of the subgoal in the format (x, y) without any additional text (ex. 10.0,15.0).
""".strip()
INST_SUBGOAL_PROMPT = """
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
	batch_size: int = 1
	shuffle_seed: int = 42
	anchor_step: int = 10
	generate_subgoal: bool = False
	instruction_prompt_text: str = DRIVING_INSTRUCTION_PROMPT
	inst_subgoal_prompt_text: str = INST_SUBGOAL_PROMPT


def resolve_cache_paths(cache_dir: str, file_indices: list[int] | None, anchor_step: int) -> tuple[str, ...]:
	directory = Path(cache_dir)
	if not directory.exists():
		raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")
	if not directory.is_dir():
		raise NotADirectoryError(f"cache_dir is not a directory: {cache_dir}")

	paths = sorted(directory.glob(f"*_t{anchor_step}.npz"))
	if file_indices is None:
		if not paths:
			raise FileNotFoundError(f"No npz cache files found in {cache_dir}")
		return tuple(str(path) for path in paths)

	index_to_path: dict[int, Path] = {}
	pattern = re.compile(r"-(\d{5})-of-\d{5}(?:_t\d+)?\.npz$")
	for path in paths:
		match = pattern.search(path.name)
		if match is not None:
			index_to_path[int(match.group(1))] = path
	resolved: list[str] = []
	for file_index in file_indices:
		if int(file_index) not in index_to_path:
			raise FileNotFoundError(f"No cache file found for shard index {file_index} in {cache_dir}")
		resolved.append(str(index_to_path[int(file_index)]))
	return tuple(resolved)


@lru_cache(maxsize=None)
def _load_instruction_offsets(index_path: str) -> tuple[int, ...]:
	path = Path(index_path)
	if not path.exists():
		raise FileNotFoundError(f"Instruction index file does not exist: {index_path}")
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)

	if isinstance(data, dict) and "byte_offsets" in data and isinstance(data["byte_offsets"], list):
		return tuple(int(v) for v in data["byte_offsets"])
	if isinstance(data, list):
		return tuple(int(v) for v in data)
	raise ValueError(f"Unsupported instruction index format: {index_path}")


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

def _infer_subgoal_from_features(features: dict[str, torch.Tensor]) -> list[str]:
	ego_trajectory = features['ego_trajectory'].cpu().detach().numpy()
	subgoal = ego_trajectory[:, -1, :2] * 100.0
	return [f"{x:.1f},{y:.1f}" for x, y in subgoal.tolist()]


def _load_instruction_for_scenario(instruction_dir: Path, cache_file: Path, scenario_index: int) -> str:
	data_path = instruction_dir / f"{cache_file.stem}.jsonl"
	index_path = instruction_dir / f"{cache_file.stem}.idx.json"
	if not data_path.exists():
		raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")
	offsets = _load_instruction_offsets(str(index_path))
	line_index = int(scenario_index)
	if line_index < 0 or line_index >= len(offsets):
		raise IndexError(
			f"Scenario index {line_index} is out of range for {data_path} (have {len(offsets)} lines)"
		)
	return _read_instruction_at_offset(data_path, offsets[line_index])


def _npz_to_torch(value: np.ndarray) -> torch.Tensor:
	return torch.from_numpy(np.asarray(value))


def _split_cache_arrays(npz_data: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
	features: dict[str, np.ndarray] = {}
	aux: dict[str, np.ndarray] = {}
	metadata: dict[str, np.ndarray] = {}
	for key in npz_data.files:
		if key.startswith("features/"):
			features[key.split("/", 1)[1]] = npz_data[key]
		elif key.startswith("aux/"):
			aux[key.split("/", 1)[1]] = npz_data[key]
		else:
			metadata[key] = npz_data[key]
	return features, aux, metadata


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

		for cache_path in cache_paths:
			cache_file = Path(cache_path)
			instruction_dir = None
			try:
				if self.cfg.instruction_dir is not None:
					instruction_dir = Path(self.cfg.instruction_dir)
					if not instruction_dir.exists():
						raise FileNotFoundError(f"Instruction directory not found: {instruction_dir}")

				with np.load(cache_file, allow_pickle=False) as npz_data:
					features_np, aux_np, _ = _split_cache_arrays(npz_data)
					if not features_np:
						raise ValueError(f"No features found in cache file: {cache_path}")

					feature_keys = sorted(features_np.keys())
					total_examples = int(features_np[feature_keys[0]].shape[0])
					for key in feature_keys[1:]:
						if int(features_np[key].shape[0]) != total_examples:
							raise ValueError(f"Mismatched feature length for {key} in {cache_path}")
					for key, value in aux_np.items():
						if int(value.shape[0]) != total_examples:
							raise ValueError(f"Mismatched aux length for {key} in {cache_path}")

					for start_index in range(0, total_examples, self.cfg.batch_size):
						end_index = min(start_index + self.cfg.batch_size, total_examples)
						features = {
							key: _npz_to_torch(value[start_index:end_index])
							for key, value in features_np.items()
						}
						aux = {
							key: _npz_to_torch(value[start_index:end_index])
							for key, value in aux_np.items()
						}

						if "scenario_index" not in npz_data.files:
							raise KeyError(f"scenario_index is missing in cache file: {cache_path}")
						scenario_indices = np.asarray(npz_data["scenario_index"]).astype(np.int64)
						subgoal_texts = _infer_subgoal_from_features(features)

						prompts: list[str] = []
						answers: list[str] = []
						if instruction_dir is None:
							raise ValueError("instruction_dir is required to load instructions")
						for scenario_index, subgoal_text in zip(scenario_indices[start_index:end_index].tolist(), subgoal_texts):
							if self.cfg.generate_subgoal:
								prompts.append(self.cfg.inst_subgoal_prompt_text)
								instruction = _load_instruction_for_scenario(
									instruction_dir,
									cache_file,
									int(scenario_index)
                                )
								answers.append(
									f"Instruction: {instruction} Subgoal: {subgoal_text}"
                                )
							else:
								prompts.append(self.cfg.instruction_prompt_text)
								answers.append(
									_load_instruction_for_scenario(
										instruction_dir,
										cache_file,
										int(scenario_index),
								    )
								)
								

						yield InstructionBatch(
							features=features,
							aux=aux,
							prompts=prompts,
							answers=answers,
						)
			except Exception as e:
				print(f"Error loading cache file {cache_path}: {e}")
				continue


def split_cache_paths(cache_paths: tuple[str, ...], val_fraction: float = 0.2) -> tuple[tuple[str, ...], tuple[str, ...]]:
	if not cache_paths:
		return cache_paths, ()
	num_val = max(1, int(len(cache_paths) * val_fraction))
	num_train = len(cache_paths) - num_val
	if num_train == 0:
		num_train = len(cache_paths) - 1
		num_val = 1
	return cache_paths[:num_train], cache_paths[num_train:]


def build_inst_dataloader(
	cache_dir: str,
	anchor_step: int,
	*,
	file_indices: list[int] | None = None,
	instruction_dir: str | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	num_workers: int = 0,
	pin_memory: bool = False,
	cache_paths: tuple[str, ...] | None = None,
	generate_subgoal: bool = False,
	instruction_prompt_text: str = DRIVING_INSTRUCTION_PROMPT,
	inst_subgoal_prompt_text: str = INST_SUBGOAL_PROMPT,
) -> DataLoader[InstructionBatch]:
	if cache_paths is None:
		cache_paths = resolve_cache_paths(cache_dir, file_indices, anchor_step)
	cfg = InstructionCacheLoaderConfig(
		cache_paths=cache_paths,
		instruction_dir=instruction_dir,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		anchor_step=anchor_step,
		instruction_prompt_text=instruction_prompt_text,
		inst_subgoal_prompt_text=inst_subgoal_prompt_text,
		generate_subgoal=generate_subgoal,
	)
	dataset = CacheInstructionDataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


def _main() -> None:
	parser = argparse.ArgumentParser(description="Smoke test the instruction cache dataloader.")
	parser.add_argument(
		"--cache_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/cache/",
		help="Directory containing NPZ cache shards.",
	)
	parser.add_argument(
		"--instruction_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/manual_instruction/",
		help="Directory containing per-shard instruction JSONL files.",
	)
	parser.add_argument(
		"--file_indices",
		type=str,
		nargs="*",
		default=None,
		help="Optional shard indices to load, e.g. --file_indices 0 1 2 or --file_indices 0,1,2.",
	)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--num_workers", type=int, default=0)
	args = parser.parse_args()

	file_indices: list[int] | None = None
	if args.file_indices is not None:
		parsed: list[int] = []
		for token in args.file_indices:
			for part in token.split(","):
				part = part.strip()
				if part:
					parsed.append(int(part))
		file_indices = parsed or None

	loader = build_inst_dataloader(
		args.cache_dir,
		anchor_step=10,
		file_indices=file_indices,
		instruction_dir=args.instruction_dir,
		batch_size=args.batch_size,
		shuffle_seed=0,
		num_workers=args.num_workers,
	)

	first_batch = next(iter(loader))
	print("feature_shapes:")
	for key, value in first_batch.features.items():
		print(f"  {key}: {tuple(value.shape)} {value.dtype}")
	print("prompts:")
	for prompt in first_batch.prompts:
		print(f"  {prompt}")
	print("answers:")
	for answer in first_batch.answers:
		print(f"  {answer}")


if __name__ == "__main__":
	_main()
