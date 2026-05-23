from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


@dataclass
class CacheBatch:
	features: dict[str, torch.Tensor]
	aux: dict[str, torch.Tensor]
	metadata: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class CacheLoaderConfig:
	cache_paths: tuple[str, ...]
	batch_size: int = 1
	shuffle_seed: int = 42
	instruction_seed: int = 0


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


def _stable_rng(seed: int, *parts: int) -> np.random.Generator:
	h = hashlib.blake2b(digest_size=8)
	h.update(str(int(seed)).encode("utf-8"))
	for part in parts:
		h.update(b":")
		h.update(str(int(part)).encode("utf-8"))
	return np.random.default_rng(int.from_bytes(h.digest(), byteorder="little", signed=False))


def _sample_instruction_features(
	features_np: dict[str, np.ndarray],
	*,
	seed: int,
	file_index: int,
	) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
	inst_multi = features_np.get("inst_features_multi")
	inst_valid_multi = features_np.get("inst_valid_multi")
	inst_single = features_np.get("inst_features")
	inst_valid_single = features_np.get("inst_valid")

	if inst_multi is None or inst_valid_multi is None:
		if inst_single is None or inst_valid_single is None:
			return None, None, None, None
		return inst_single, inst_valid_single, None, None

	if inst_multi.ndim != 3:
		raise ValueError(f"inst_features_multi must have shape [N, K, D], got {inst_multi.shape}")
	if inst_valid_multi.ndim != 2:
		raise ValueError(f"inst_valid_multi must have shape [N, K], got {inst_valid_multi.shape}")
	if inst_multi.shape[:2] != inst_valid_multi.shape:
		raise ValueError(
			f"Mismatched instruction shapes: {inst_multi.shape} vs {inst_valid_multi.shape}"
		)

	batch_size, num_instructions, embed_dim = inst_multi.shape
	selected = np.zeros((batch_size, embed_dim), dtype=np.float32)
	selected_valid = np.zeros((batch_size,), dtype=np.bool_)

	for row_idx in range(batch_size):
		valid_indices = np.flatnonzero(inst_valid_multi[row_idx])
		if valid_indices.size == 0:
			continue
		rng = _stable_rng(seed, file_index, row_idx)
		choice = int(valid_indices[rng.integers(0, valid_indices.size)])
		selected[row_idx] = np.asarray(inst_multi[row_idx, choice], dtype=np.float32)
		selected_valid[row_idx] = True

	return selected, selected_valid, inst_multi, inst_valid_multi


class CacheInstructionDataset(IterableDataset[CacheBatch]):
	def __init__(self, cfg: CacheLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def __iter__(self) -> Iterator[CacheBatch]:
		worker_info = get_worker_info()
		cache_paths = list(self.cfg.cache_paths)
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
		if worker_info is not None:
			cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

		for file_index, cache_path in enumerate(cache_paths):
			cache_file = Path(cache_path)
			try:
				with np.load(cache_file, allow_pickle=False) as npz_data:
					features_np, aux_np, metadata_np = _split_cache_arrays(npz_data)
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

					inst_selected, inst_valid, inst_multi, inst_valid_multi = _sample_instruction_features(
						features_np,
						seed=int(self.cfg.instruction_seed),
						file_index=file_index,
					)

					for start_index in range(0, total_examples, self.cfg.batch_size):
						end_index = min(start_index + self.cfg.batch_size, total_examples)
						features = {
							key: _npz_to_torch(value[start_index:end_index])
							for key, value in features_np.items()
							if key not in {"inst_features", "inst_valid", "inst_features_multi", "inst_valid_multi"}
						}
						if inst_selected is not None and inst_valid is not None:
							features["inst_features"] = _npz_to_torch(inst_selected[start_index:end_index])
							features["inst_valid"] = _npz_to_torch(inst_valid[start_index:end_index])
							if inst_multi is not None and inst_valid_multi is not None:
								features["inst_features_multi"] = _npz_to_torch(inst_multi[start_index:end_index])
								features["inst_valid_multi"] = _npz_to_torch(inst_valid_multi[start_index:end_index])

						aux = {
							key: _npz_to_torch(value[start_index:end_index])
							for key, value in aux_np.items()
						}

						metadata: dict[str, torch.Tensor] = {}
						for key, value in metadata_np.items():
							if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_examples:
								metadata[key] = _npz_to_torch(value[start_index:end_index])

						yield CacheBatch(features=features, aux=aux, metadata=metadata or None)
			except Exception as e:
				print(f"Error loading cache file {cache_path}: {e}")
				continue


def build_cache_dataloader(
	cache_dir: str,
	anchor_step: int,
	*,
	file_indices: list[int] | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	instruction_seed: int = 0,
	num_workers: int = 0,
	pin_memory: bool = False,
	cache_paths: tuple[str, ...] | None = None,
) -> DataLoader[CacheBatch]:
	if cache_paths is None:
		cache_paths = resolve_cache_paths(cache_dir, file_indices, anchor_step)
	cfg = CacheLoaderConfig(
		cache_paths=cache_paths,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		instruction_seed=instruction_seed,
	)
	dataset = CacheInstructionDataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


def _main() -> None:
	parser = argparse.ArgumentParser(description="Smoke test the cache instruction dataloader.")
	parser.add_argument("--cache_dir", type=str, required=True)
	parser.add_argument("--anchor_step", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=1)
	args = parser.parse_args()

	loader = build_cache_dataloader(
		args.cache_dir,
		anchor_step=args.anchor_step,
		batch_size=args.batch_size,
		shuffle_seed=0,
		instruction_seed=0,
	)

	for i, batch in enumerate(loader):
		print(f"batch={i} feature_keys={sorted(batch.features.keys())[:8]}...")
		if i >= 1:
			break


if __name__ == "__main__":
	_main()