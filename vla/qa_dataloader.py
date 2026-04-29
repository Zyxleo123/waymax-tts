from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from dataclasses import dataclass
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data.types import PreprocessConfig
from vla.preprocess import TorchPreprocessBatch


@dataclass(frozen=True)
class QADataLoaderConfig:
	tfrecord_paths: tuple[str, ...]
	qa_dir: str | None = None
	batch_size: int = 1
	max_num_objects: int = 64
	shuffle_seed: int = 0
	shuffle_buffer_size: int = 1024
	dataset_num_shards: int = 1
	include_sdc_paths: bool = False
	preprocess_cfg: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)
	anchor_step_override: int | None = None


@dataclass(frozen=True)
class QACacheLoaderConfig:
	cache_paths: tuple[str, ...]
	qa_dir: str | None = None
	batch_size: int = 1
	shuffle_seed: int = 0


def _build_waymax_dataset_config(cfg: QADataLoaderConfig, tfrecord_path: str):
	from waymax import config as waymax_config

	base_cfg = waymax_config.WOD_1_3_1_TRAINING
	cfg_fields = {field.name for field in dataclasses.fields(base_cfg)}

	replace_kwargs: dict[str, object] = {
		"path": str(Path(tfrecord_path)),
		"max_num_objects": int(cfg.max_num_objects),
		"batch_dims": (int(cfg.batch_size),),
		"shuffle_seed": int(cfg.shuffle_seed),
		"shuffle_buffer_size": int(cfg.shuffle_buffer_size),
	}

	def _set_if_supported(field: str, value: object) -> None:
		if field in cfg_fields:
			replace_kwargs[field] = value

	_set_if_supported("num_shards", int(cfg.dataset_num_shards))
	_set_if_supported("include_sdc_paths", bool(cfg.include_sdc_paths))

	return dataclasses.replace(base_cfg, **replace_kwargs)


def _resolve_tfrecord_paths(tfrecord_dir: str, file_indices: list[int] | None) -> tuple[str, ...]:
	directory = Path(tfrecord_dir)
	if not directory.exists():
		raise FileNotFoundError(f"tfrecord_dir does not exist: {tfrecord_dir}")
	if not directory.is_dir():
		raise NotADirectoryError(f"tfrecord_dir is not a directory: {tfrecord_dir}")

	if file_indices is not None:
		return tuple(
			str(directory / f"training_tfexample.tfrecord-{int(file_idx):05d}-of-01000")
			for file_idx in file_indices
		)

	paths = sorted(directory.glob("training_tfexample.tfrecord-*-of-*"))
	if not paths:
		raise FileNotFoundError(
			f"No training_tfexample.tfrecord shards found in {tfrecord_dir}. "
			"Expected files like training_tfexample.tfrecord-00000-of-01000."
		)
	return tuple(str(path) for path in paths)


def _resolve_cache_paths(cache_dir: str, file_indices: list[int] | None) -> tuple[str, ...]:
	directory = Path(cache_dir)
	if not directory.exists():
		raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")
	if not directory.is_dir():
		raise NotADirectoryError(f"cache_dir is not a directory: {cache_dir}")

	paths = sorted(directory.glob("*.npz"))
	if file_indices is None:
		if not paths:
			raise FileNotFoundError(f"No npz cache files found in {cache_dir}")
		return tuple(str(path) for path in paths)

	index_to_path: dict[int, Path] = {}
	pattern = re.compile(r"-(\d{5})-of-\d{5}\.npz$")
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
def _load_qa_json(qa_json_path: str) -> dict[str, Any]:
	path = Path(qa_json_path)
	if not path.exists():
		raise FileNotFoundError(f"QA json file does not exist: {qa_json_path}")
	with path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	if not isinstance(data, dict):
		raise ValueError(f"QA json must be a dict at top level: {qa_json_path}")
	return data


def _build_yes_no_question_bank(answers: dict[str, Any]) -> list[dict[str, Any]]:
	question_specs = [
		("has_left_lane", "Is there a lane on the left of the ego vehicle?", bool(answers["has_left_lane"])),
		("has_right_lane", "Is there a lane on the right of the ego vehicle?", bool(answers["has_right_lane"])),
		(
			"num_vehicle_left",
			"Is there any vehicle on the left side of the ego vehicle? \nAnswer (yes or no):",
			int(answers["num_vehicle_left"]) > 0,
		),
		(
			"num_vehicle_right",
			"Is there any vehicle on the right side of the ego vehicle? \nAnswer (yes or no):",
			int(answers["num_vehicle_right"]) > 0,
		),
		(
			"num_vehicle_front_same_lane",
			"Is there any vehicle in front of the ego vehicle in the same lane? \nAnswer (yes or no):",
			int(answers["num_vehicle_front_same_lane"]) > 0,
		),
		(
			"num_vehicle_behind_same_lane",
			"Is there any vehicle behind the ego vehicle in the same lane? \nAnswer (yes or no):",
			int(answers["num_vehicle_behind_same_lane"]) > 0,
		),
		(
			"num_pedestrian_front",
			"Is there any pedestrian in front of the ego vehicle? \nAnswer (yes or no):",
			int(answers["num_pedestrian_front"]) > 0,
		),
		(
			"traffic_light_state",
			"Is the traffic light in front of the ego vehicle red? \nAnswer (yes or no):",
			int(answers["traffic_light_state"]) == 4,
		),
	]

	return [
		{
			"key": key,
			"question": question,
			"answer": "yes" if is_yes else "no",
			"label": bool(is_yes),
		}
		for key, question, is_yes in question_specs
	]


def _load_qas_for_scenario(qa_json_path: str, scenario_index: int) -> dict[str, Any]:
	qa_json = _load_qa_json(qa_json_path)
	scenario_key = str(int(scenario_index))
	if scenario_key not in qa_json:
		raise KeyError(f"Scenario index {scenario_index} not found in QA json: {qa_json_path}")
	entry = qa_json[scenario_key]
	if not isinstance(entry, dict):
		raise ValueError(f"Invalid QA entry for scenario {scenario_index} in {qa_json_path}")
	answers = entry.get("answers", {})
	if not isinstance(answers, dict):
		raise ValueError(f"Invalid answers field for scenario {scenario_index} in {qa_json_path}")
	return {
		"scenario_index": int(scenario_index),
		"timestep": int(entry.get("timestep", 0)),
		"qas": _build_yes_no_question_bank(answers),
	}


def _load_qas_for_batch(qa_json_path: str, start_index: int, batch_size: int) -> list[dict[str, Any]]:
	return [_load_qas_for_scenario(qa_json_path, start_index + offset) for offset in range(batch_size)]


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


class CacheQADataset(IterableDataset[TorchPreprocessBatch]):
	def __init__(self, cfg: QACacheLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def __iter__(self) -> Iterator[TorchPreprocessBatch]:
		worker_info = get_worker_info()
		cache_paths = list(self.cfg.cache_paths)
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
		if worker_info is not None:
			cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

		for cache_path in cache_paths:
			cache_file = Path(cache_path)
			qa_json_path = None
			if self.cfg.qa_dir is not None:
				qa_json_path = str(Path(self.cfg.qa_dir) / f"{cache_file.stem}.json")

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
					batch = TorchPreprocessBatch(features=features, aux=aux)
					if qa_json_path is not None:
						batch.qa = _load_qas_for_batch(qa_json_path, start_index, end_index - start_index)
					yield batch


def build_qa_dataloader(
	cache_dir: str,
	*,
	file_indices: list[int] | None = None,
	qa_dir: str | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	num_workers: int = 0,
	pin_memory: bool = False,
) -> DataLoader[TorchPreprocessBatch]:
	cache_paths = _resolve_cache_paths(cache_dir, file_indices)
	cfg = QACacheLoaderConfig(
		cache_paths=cache_paths,
		qa_dir=qa_dir,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
	)
	dataset = CacheQADataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


__all__ = [
	"QADataLoaderConfig",
	"QACacheLoaderConfig",
	"CacheQADataset",
	"build_qa_dataloader",
]


def _main() -> None:
	parser = argparse.ArgumentParser(description="Smoke test the VLA QA cache dataloader.")
	parser.add_argument("--cache_dir", type=str, help="Directory containing NPZ cache shards.",
					    default="/zfsauton/scratch/mineuih/waymax_rs/qa_cache/")
	parser.add_argument("--qa_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/qa_dataset/", help="Directory containing per-shard QA JSON files.")
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

	loader = build_qa_dataloader(
		args.cache_dir,
		file_indices=file_indices,
		qa_dir=args.qa_dir,
		batch_size=args.batch_size,
		shuffle_seed=0,
		num_workers=args.num_workers,
	)

	first_batch = next(iter(loader))
	print("feature_shapes:")
	for key, value in first_batch.features.items():
		print(f"  {key}: {tuple(value.shape)} {value.dtype}")
	print("qa:")
	if first_batch.qa is None:
		print("  None")
	else:
		for item in first_batch.qa:
			print(
				f"  scenario_index={item['scenario_index']} timestep={item['timestep']} "
				f"num_qas={len(item['qas'])}"
			)
			for qa_item in item["qas"]:
				print(f"    - {qa_item['question']} -> {qa_item['answer']}")


if __name__ == "__main__":
	_main()
