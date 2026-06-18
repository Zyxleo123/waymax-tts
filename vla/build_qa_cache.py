from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla.preprocess import TorchPreprocessBatch, preprocess_simulator_state
from vla.qa_dataloader import QADataLoaderConfig, _build_waymax_dataset_config, _resolve_tfrecord_paths
from tqdm import tqdm
from waymax import dataloader
from viz.render import _load_scenario_state_fast


@dataclass(frozen=True)
class BuildQACacheConfig:
	tfrecord_dir: str
	output_dir: str
	file_indices: list[int] | None = None
	batch_size: int = 1
	max_num_objects: int = 64
	shuffle_seed: int = 0
	shuffle_buffer_size: int = 1024
	dataset_num_shards: int = 1
	include_sdc_paths: bool = False
	anchor_step_override: int | None = 0
	max_batches_per_shard: int | None = None
	compress: bool = True


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
	return value.detach().cpu().numpy()


def _append_batch(storage: dict[str, list[np.ndarray]], prefix: str, items: dict[str, torch.Tensor]) -> None:
	for key, value in items.items():
		storage.setdefault(f"{prefix}/{key}", []).append(_tensor_to_numpy(value))


def _concat_storage(storage: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
	result: dict[str, np.ndarray] = {}
	for key, arrays in storage.items():
		if not arrays:
			continue
		if len(arrays) == 1:
			result[key] = arrays[0]
		else:
			result[key] = np.concatenate(arrays, axis=0)
	return result


def _save_npz(output_path: Path, batch: TorchPreprocessBatch, *, compress: bool) -> None:
	arrays: dict[str, np.ndarray] = {}
	_append_batch(arrays := {}, "features", batch.features)
	_append_batch(arrays, "aux", batch.aux)

	flat_arrays = _concat_storage(arrays)
	flat_arrays["has_qa"] = np.array(batch.qa is not None, dtype=np.bool_)

	if compress:
		np.savez_compressed(output_path, **flat_arrays)
	else:
		np.savez(output_path, **flat_arrays)


def _build_cache_for_shard(
	*,
	tfrecord_path: str,
	output_path: Path,
	base_cfg: BuildQACacheConfig,
) -> None:

	qa_cfg = QADataLoaderConfig(
		tfrecord_paths=(tfrecord_path,),
		qa_dir=None,
		batch_size=base_cfg.batch_size,
		max_num_objects=base_cfg.max_num_objects,
		shuffle_seed=base_cfg.shuffle_seed,
		shuffle_buffer_size=base_cfg.shuffle_buffer_size,
		dataset_num_shards=base_cfg.dataset_num_shards,
		include_sdc_paths=base_cfg.include_sdc_paths,
		anchor_step_override=base_cfg.anchor_step_override,
	)
	waymax_cfg = _build_waymax_dataset_config(qa_cfg, tfrecord_path)

	generator = torch.Generator(device="cpu")
	generator.manual_seed(int(base_cfg.shuffle_seed))

	storage: dict[str, list[np.ndarray]] = {}
	processed_batches = 0
	pbar = tqdm(desc=f"Processing {os.path.basename(tfrecord_path)}", unit="batch")
	scenario_index = 0
	while True:
		try:
			sim_state = _load_scenario_state_fast(waymax_cfg, scenario_index)
		except Exception:
			break
		batch, _ = preprocess_simulator_state(
			sim_state,
			generator,
			qa_cfg.preprocess_cfg,
			anchor_step_override=base_cfg.anchor_step_override,
			goal_step_override=90
		)
		_append_batch(storage, "features", batch.features)
		_append_batch(storage, "aux", batch.aux)
		processed_batches += 1
		scenario_index += 1
		pbar.update(1)
		
	pbar.close()

	if processed_batches == 0:
		raise RuntimeError(f"No batches were produced for shard: {tfrecord_path}")

	flat_arrays = _concat_storage(storage)
	flat_arrays["tfrecord_path"] = np.array(os.path.basename(tfrecord_path))
	flat_arrays["num_batches"] = np.array(processed_batches, dtype=np.int32)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	if base_cfg.compress:
		np.savez_compressed(output_path, **flat_arrays)
	else:
		np.savez(output_path, **flat_arrays)


def build_qa_cache(cfg: BuildQACacheConfig) -> None:
	output_dir = Path(cfg.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tfrecord_paths = _resolve_tfrecord_paths(cfg.tfrecord_dir, cfg.file_indices)
	for tfrecord_path in tfrecord_paths:
		output_path = output_dir / f"{Path(tfrecord_path).name}.npz"
		print(f"Caching {tfrecord_path} -> {output_path}")
		_build_cache_for_shard(
			tfrecord_path=tfrecord_path,
			output_path=output_path,
			base_cfg=cfg,
		)


def _parse_args() -> BuildQACacheConfig:
	parser = argparse.ArgumentParser(description="Build NPZ caches for Waymo QA preprocessing features.")
	parser.add_argument("--tfrecord_dir", type=str, required=True)
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--max_num_objects", type=int, default=64)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--shuffle_buffer_size", type=int, default=1024)
	parser.add_argument("--dataset_num_shards", type=int, default=1)
	parser.add_argument("--include_sdc_paths", action="store_true")
	parser.add_argument("--anchor_step_override", type=int, default=0)
	parser.add_argument("--max_batches_per_shard", type=int, default=None)
	parser.add_argument("--no_compress", action="store_true")
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

	return BuildQACacheConfig(
		tfrecord_dir=args.tfrecord_dir,
		output_dir=args.output_dir,
		file_indices=file_indices,
		batch_size=args.batch_size,
		max_num_objects=args.max_num_objects,
		shuffle_seed=args.shuffle_seed,
		shuffle_buffer_size=args.shuffle_buffer_size,
		dataset_num_shards=args.dataset_num_shards,
		include_sdc_paths=args.include_sdc_paths,
		anchor_step_override=args.anchor_step_override,
		max_batches_per_shard=args.max_batches_per_shard,
		compress=not args.no_compress,
	)


def main() -> None:
	build_qa_cache(_parse_args())


if __name__ == "__main__":
	main()
