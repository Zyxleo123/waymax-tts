from __future__ import annotations

import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


@dataclass
class CacheBatch:
	features: dict[str, jax.Array]
	aux: dict[str, jax.Array]
	metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CacheLoaderConfig:
	cache_paths: tuple[str, ...]
	batch_size: int = 1
	shuffle_seed: int = 42
	drop_last: bool = True


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


def _npz_to_jax(value: np.ndarray) -> jax.Array:
	return jnp.asarray(np.asarray(value))


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


def _convert_metadata_value(value: np.ndarray | Any) -> Any:
	try:
		return _npz_to_jax(np.asarray(value))
	except Exception:
		return value


class CacheInstructionDatasetJax(IterableDataset[CacheBatch]):
	def __init__(self, cfg: CacheLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg
		self._epoch_index = 0

	def __iter__(self) -> Iterator[CacheBatch]:
		worker_info = get_worker_info()
		cache_paths = list(self.cfg.cache_paths)
		epoch_index = int(self._epoch_index)
		self._epoch_index += 1
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed) + epoch_index).shuffle(cache_paths)
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

					if self.cfg.batch_size <= 0:
						raise ValueError(f"batch_size must be positive, got {self.cfg.batch_size}")

					example_order = np.arange(total_examples)
					if self.cfg.shuffle_seed:
						rng = np.random.default_rng(
							int(_stable_rng(int(self.cfg.shuffle_seed), epoch_index, file_index).integers(0, 2**63 - 1))
						)
						rng.shuffle(example_order)

					usable_examples = total_examples
					if self.cfg.drop_last:
						usable_examples = (total_examples // self.cfg.batch_size) * self.cfg.batch_size

					for start_index in range(0, usable_examples, self.cfg.batch_size):
						end_index = min(start_index + self.cfg.batch_size, usable_examples)
						batch_indices = example_order[start_index:end_index]
						features = {
							key: _npz_to_jax(value[batch_indices])
							for key, value in features_np.items()
						}
						aux = {
							key: _npz_to_jax(value[batch_indices])
							for key, value in aux_np.items()
						}

						metadata: dict[str, Any] = {}
						for key, value in metadata_np.items():
							if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_examples:
								metadata[key] = _convert_metadata_value(value[batch_indices])
							else:
								metadata[key] = _convert_metadata_value(value)

						yield CacheBatch(features=features, aux=aux, metadata=metadata or None)
			except Exception as e:
				print(f"Error loading cache file {cache_path}: {e}")
				continue


def build_cache_dataloader_jax(
	cache_dir: str,
	anchor_step: int | str,
	*,
	file_indices: list[int] | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	drop_last: bool = True,
	num_workers: int = 0,
	pin_memory: bool = False,
	cache_paths: tuple[str, ...] | None = None,
) -> DataLoader[CacheBatch]:
	if cache_paths is None:
		if anchor_step == 'all':
			for t in [10, 20, 30, 40]:
				print(f"Trying to find cache files for anchor_step={t} in {cache_dir}...")
				try:
					cache_paths = resolve_cache_paths(cache_dir, file_indices, t)
					print(f"Found cache files for anchor_step={t}: {len(cache_paths)} files")
				except FileNotFoundError:
					print(f"No cache files found for anchor_step={t} in {cache_dir}")
		else:
			cache_paths = resolve_cache_paths(cache_dir, file_indices, anchor_step)
	cfg = CacheLoaderConfig(
		cache_paths=cache_paths,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		drop_last=drop_last,
	)
	dataset = CacheInstructionDatasetJax(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


def _main() -> None:
	parser = argparse.ArgumentParser(description="Smoke test the JAX cache instruction dataloader.")
	parser.add_argument("--cache_dir", type=str, required=True)
	parser.add_argument("--anchor_step", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--drop_last", action="store_true")
	args = parser.parse_args()

	loader = build_cache_dataloader_jax(
		args.cache_dir,
		anchor_step=args.anchor_step,
		batch_size=args.batch_size,
		shuffle_seed=0,
		drop_last=bool(args.drop_last),
	)

	for i, batch in enumerate(loader):
		print(f"batch={i} feature_keys={sorted(batch.features.keys())[:8]}...")
		if i >= 1:
			break


if __name__ == "__main__":
	_main()
