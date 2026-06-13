from __future__ import annotations

import numpy as np
import torch
import jax.numpy as jnp
import json
import re
import random
from pathlib import Path
from typing import Any, Iterator


VLA_PROMPT = """
Given a driving scenario, propose a driving instruction and a subgoal that helps the ego vehicle reach the goal.
Only output the instruction and subgoal without any additional text, in the format:
Instruction: instruction text here. Subgoal: x,y
"""

def infer_subgoal_from_features(features: dict[str, torch.Tensor], ego_range=100.0) -> list[str]:
	subgoal_xy = features['subgoal_xy'].cpu().detach().numpy() * ego_range
	return [f"{x:.1f},{y:.1f}" for x, y in subgoal_xy.tolist()]


def build_question_bank(annotations: dict[str, Any]) -> list[dict[str, Any]]:
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


def build_annotation_paths(cache_file: Path, annotation_dir: str, anchor_step: int) -> tuple[Path, Path]:
	annotation_root = Path(annotation_dir)
	stem = cache_file.name.removesuffix(".sim_state_cache.npz")
	return (
		annotation_root / f"{stem}_t{anchor_step}.jsonl",
		annotation_root / f"{stem}_t{anchor_step}.idx.json",
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

def npz_to_jax(value: np.ndarray):
	return jnp.asarray(np.asarray(value))