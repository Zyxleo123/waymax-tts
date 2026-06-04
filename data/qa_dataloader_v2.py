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
from data.preprocess_cached import preprocess_cached_npz
from data.utils import (
    resolve_cache_paths,
	load_offsets,
	build_instruction_paths,
	split_cache_paths,
	npz_to_torch
)

@dataclass
class QABatch:
	features: dict[str, torch.Tensor]
	aux: dict[str, torch.Tensor]
	qa: Any | None = None


@dataclass(frozen=True)
class QACacheLoaderConfig:
	cache_paths: tuple[str, ...]
	preprocess_cfg: Any
	qa_dir: str | None = None
	anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40)
	batch_size: int = 1
	shuffle_seed: int = 42



def _build_question_bank(annotations: dict[str, Any]) -> list[dict[str, Any]]:
	"""Build question-answer pairs based on available keys in the answers dictionary.
	
	For each key present in answers, generate an appropriate question with the corresponding answer.
	Handles both yes/no questions and numeric/float answers.
	"""
	qa_list = []
	# lane existence questions (yes/no)
	for lane in ["left", "right"]:
		is_yes = bool(len(annotations["lane_context"].get(f"{lane}_lane_ids", [])) > 0)
		qa_list.append({
			"key": "lane_existence",
			"question": f"Is there a lane on the {lane} of the ego vehicle? Answer with yes or no.",
			"answer": "yes" if is_yes else "no",
			"label": is_yes,
		})
	# traffic light questions
	traffic_lights = annotations["lane_context"]['traffic_lights']
	is_yes = bool(len(traffic_lights) > 0)
	qa_list.append({
		"key": "traffic_light_existence",
		"question": "Is there a traffic light in front of the ego vehicle? Answer with yes or no.",
		"answer": "yes" if is_yes else "no",
		"label": is_yes,
	})
	if is_yes and isinstance(traffic_lights, dict):
		for direction, state in traffic_lights.items():
			if "arrow" in state:
				is_yes = "stop" in state.lower()
				qa_list.append({
					"key": "traffic_light_state",
					"question": f"Is the {direction}-turn signal stop? Answer with yes or no.",
					"answer": "yes" if is_yes else "no",
					"label": is_yes,
				})
			else:
				is_yes = "stop" in state.lower()
				qa_list.append({
					"key": "traffic_light_state",
					"question": f"Is the straight through signal stop? Answer with yes or no.",
					"answer": "yes" if is_yes else "no",
					"label": is_yes,
				})

	# Vehicle questions
	for direction, vehicle in annotations['vehicles'].items():
		is_yes = bool(vehicle is not None)
		if 'front' in direction:
			dir_text = "ahead"
		elif 'rear' in direction:
			dir_text = "behind"
		if 'left' in direction:
			lane_text = "in the left lane"
		elif 'right' in direction:
			lane_text = "in the right lane"
		else:
			lane_text = "in the same lane"
		qa_list.append({
			"key": f"vehicle_existence",
			"question": f"Is there any vehicle {dir_text} of the ego vehicle {lane_text}? Answer with yes or no.",
			"answer": "yes" if is_yes else "no",
			"label": is_yes,
		})
		if is_yes:
			qa_list.append({
				"key": f"vehicle_state",
				"question": f"What is the distance to the vehicle {dir_text} of the ego vehicle {lane_text} in meters? Answer with a number only. (ex. 10.0)",
				"answer": f"{vehicle['distance']:.1f}",
				"label": float(vehicle['distance']),
			})
			qa_list.append({
				"key": f"vehicle_state",
				"question": f"What is the speed of the vehicle {dir_text} of the ego vehicle {lane_text} in m/s? Answer with a number only. (ex. 5.0)",
				"answer": f"{vehicle['speed']:.1f}",
				"label": float(vehicle['speed']),
			})
	qa_list.append({
		"key": "ego_speed",
		"question": "What is the current speed of the ego vehicle in m/s? Answer with a number only. (ex. 15.0)",
		"answer": f"{annotations['ego_motion']['start_speed']:.1f}",
		"label": float(annotations['ego_motion']['start_speed']),
	})

	# Risk questions
	risk_pedestrian, risk_vehicle = False, False
	for risk_object in annotations['risk_objects']:
		if risk_object['object_type'] == 'pedestrian':
			risk_pedestrian = True
		elif risk_object['object_type'] == 'vehicle':
			risk_vehicle = True
	qa_list.append({
		"key": "risk_pedestrian",
		"question": "Is there any pedestrian that may impede the ego vehicle's progress? Answer with yes or no.",
		"answer": "yes" if risk_pedestrian else "no",
		"label": risk_pedestrian,
	})
	qa_list.append({
		"key": "risk_vehicle",
		"question": "Is there any vehicle that may impede the ego vehicle's progress? Answer with yes or no.",
		"answer": "yes" if risk_vehicle else "no",
		"label": risk_vehicle,
	})

	goal_x, goal_y = annotations["goal_info"]["relative_position"]
	goal_x = "0.0" if f"{goal_x:.1f}" == "-0.0" else f"{goal_x:.1f}"
	goal_y = "0.0" if f"{goal_y:.1f}" == "-0.0" else f"{goal_y:.1f}"
	qa_list.append({
		"key": "goal",
		"question": "What is the relative x-position of the goal in meters? Answer with a number only. (ex. 10.0)",
		"answer": goal_x,
		"label": float(goal_x),
	})
	qa_list.append({
		"key": "goal",
		"question": "What is the relative y-position of the goal in meters? Answer with a number only. (ex. 5.0)",
		"answer": goal_y,
		"label": float(goal_y),
	})
	
	return random.sample(qa_list, k=1)  # Randomly select one question-answer pair for this scenario


def _load_qas_for_batch(qa_jsonl_path: str, qa_index_path: str, scenario_indices: list[int]) -> list[dict[str, Any]]:
	offsets = load_offsets(Path(qa_index_path))
	qas = []
	with Path(qa_jsonl_path).open("rb") as f:
		for i in scenario_indices:
			offset = offsets[i]
			f.seek(int(offset))
			raw = f.readline()
			if not raw:
				raise ValueError(f"Missing QA entry at offset {offset} in {qa_jsonl_path}")
			try:
				obj = json.loads(raw.decode("utf-8"))
			except json.JSONDecodeError as exc:
				raise ValueError(f"Failed to decode QA entry at offset {offset} in {qa_jsonl_path}") from exc
			qas.append(
				{
					"scenario_index": obj["scenario_index"],
					"timestep": obj["annotation"]["scenario_window"]["start"],
					"qas": _build_question_bank(obj['annotation'])
				}
			)
	return qas



class CacheQADataset(IterableDataset[QABatch]):
	def __init__(self, cfg: QACacheLoaderConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def __iter__(self) -> Iterator[QABatch]:
		worker_info = get_worker_info()
		cache_paths = list(self.cfg.cache_paths)
		if self.cfg.shuffle_seed:
			random.Random(int(self.cfg.shuffle_seed)).shuffle(cache_paths)
		if worker_info is not None:
			cache_paths = cache_paths[worker_info.id :: worker_info.num_workers]

		for anchor_step in self.cfg.anchor_steps:
			for cache_path in cache_paths:
				cache_file = Path(cache_path)
				qa_jsonl_path, qa_index_path = build_instruction_paths(cache_file, self.cfg.qa_dir or "", anchor_step)
				preprocessed = preprocess_cached_npz(cache_file, cfg=self.cfg.preprocess_cfg, anchor_step_override=anchor_step)
				feature_keys = sorted(preprocessed["features"].keys())
				total_examples = int(preprocessed["features"][feature_keys[0]].shape[0])
				scenario_indices = preprocessed["metadata"]["scenario_indices"]
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
					batch = QABatch(features=batch_features, aux=batch_aux)
					batch.qa = _load_qas_for_batch(
						str(qa_jsonl_path),
						str(qa_index_path),
						scenario_indices[start_index:end_index]
					)
					yield batch




def build_qa_dataloader(
	cache_dir: str,
	preprocess_cfg,
	*,
	file_indices: list[int] | None = None,
	qa_dir: str | None = None,
	batch_size: int = 1,
	shuffle_seed: int = 0,
	num_workers: int = 0,
	pin_memory: bool = False,
	cache_paths: tuple[str, ...] | None = None,
) -> DataLoader[QABatch]:
	if cache_paths is None:
		cache_paths = resolve_cache_paths(cache_dir, file_indices)
	cfg = QACacheLoaderConfig(
		cache_paths=cache_paths,
		qa_dir=qa_dir,
		batch_size=batch_size,
		shuffle_seed=shuffle_seed,
		preprocess_cfg=preprocess_cfg,
	)
	dataset = CacheQADataset(cfg)
	return DataLoader(
		dataset,
		batch_size=None,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


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
		preprocess_cfg=args,
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
