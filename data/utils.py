from __future__ import annotations

import numpy as np
import torch
import json
import re
from pathlib import Path


def resolve_cache_paths(cache_dir: str, file_indices: list[int] | None) -> tuple[str, ...]:
	directory = Path(cache_dir)
	if not directory.exists():
		raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")
	if not directory.is_dir():
		raise NotADirectoryError(f"cache_dir is not a directory: {cache_dir}")

	paths = sorted(directory.glob(f"*.npz"))
	if file_indices is None:
		if not paths:
			raise FileNotFoundError(f"No npz cache files found in {cache_dir}")
		return tuple(str(path) for path in paths)

	index_to_path: dict[int, Path] = {}
	pattern = re.compile(r"-(\d{5})-of-\d{5}\.sim_state_cache\.npz$")
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


def load_offsets(index_path: Path) -> list[int]:
	if not index_path.exists():
		raise FileNotFoundError(f"Instruction index not found: {index_path}")
	with index_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return [int(v) for v in data]
	if isinstance(data, dict):
		if "byte_offsets" in data and isinstance(data["byte_offsets"], list):
			return [int(v) for v in data["byte_offsets"]]
		if len(data) == 1:
			value = next(iter(data.values()))
			if isinstance(value, list):
				return [int(v) for v in value]
	raise ValueError(f"Unsupported instruction index format: {index_path}")


def build_instruction_paths(cache_file: Path, instruction_dir: str, anchor_step: int) -> tuple[Path, Path]:
	instruction_root = Path(instruction_dir)
	stem = cache_file.name.removesuffix(".sim_state_cache.npz")
	return (
		instruction_root / f"{stem}_t{anchor_step}.jsonl",
		instruction_root / f"{stem}_t{anchor_step}.idx.json",
	)

def split_cache_paths(cache_paths: tuple[str, ...], val_fraction: float = 0.2) -> tuple[tuple[str, ...], tuple[str, ...]]:
	"""Split cache paths into train and validation sets.
	
	Returns (train_paths, val_paths).
	"""
	if not cache_paths:
		return cache_paths, ()
	num_val = max(1, int(len(cache_paths) * val_fraction))
	num_train = len(cache_paths) - num_val
	if num_train == 0:
		num_train = len(cache_paths) - 1
		num_val = 1
	return cache_paths[:num_train], cache_paths[num_train:]


def npz_to_torch(value: np.ndarray) -> torch.Tensor:
	return torch.from_numpy(np.asarray(value))