from __future__ import annotations

import argparse
import gc
import io
import json
import os
import multiprocessing as mp
import signal
import sys
import zipfile
from glob import glob
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from lane_graph.lane_graph_utils import LaneGraphData

# Allow forcing CPU-only mode via CLI flag `--cpu` or `--no-gpu`, or env var WAYMAX_FORCE_CPU.
# This must be set before importing modules that may initialize JAX/TensorFlow GPU devices.
if any(arg in ("--cpu", "--no-gpu") for arg in sys.argv) or os.environ.get("WAYMAX_FORCE_CPU", "").lower() in (
	"1",
	"true",
	"yes",
):
	os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _find_repo_root(start: Path) -> Path:
	for candidate in [start, *start.parents]:
		if (candidate / "README.md").exists() and (candidate / "data").is_dir():
			return candidate
	raise FileNotFoundError("Could not find the repository root.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from annotation.helpers import (
	check_direction,
	check_risk,
	check_speed,
	check_traffic_light,
	check_turn,
	extend_lane_id,
	get_current_lane_id,
	get_ego_idx,
	get_goal_info,
	get_left_lane_id,
	get_right_lane_id,
	get_vehicle_lane_id,
	get_vehicle_on_target_lane,
)


SIM_STATE_CACHE_FIELDS = {
	"object_metadata": {
		"is_sdc": "object_metadata_is_sdc",
		"object_types": "object_metadata_object_types",
	},
	"log_trajectory": {
		"x": "log_trajectory_x",
		"y": "log_trajectory_y",
		"yaw": "log_trajectory_yaw",
		"valid": "log_trajectory_valid",
		"vel_x": "log_trajectory_vel_x",
		"vel_y": "log_trajectory_vel_y",
		"speed": "log_trajectory_speed",
	},
	"log_traffic_light": {
		"state": "log_traffic_light_state",
		"lane_ids": "log_traffic_light_lane_ids",
		"valid": "log_traffic_light_valid",
	},
	"roadgraph_points": {
		"ids": "roadgraph_points_ids",
		"types": "roadgraph_points_types",
		"x": "roadgraph_points_x",
		"y": "roadgraph_points_y",
		"dir_x": "roadgraph_points_dir_x",
		"dir_y": "roadgraph_points_dir_y",
	},
}


class _Group:
	pass


def _to_world_first(array: np.ndarray) -> np.ndarray:
	return np.expand_dims(np.asarray(array), axis=0)


def _scenario_from_cache(npz_data: np.lib.npyio.NpzFile, scenario_pos: int):
	sim_state = _Group()
	for group_name, attributes in SIM_STATE_CACHE_FIELDS.items():
		group_obj = _Group()
		for attr_name, field_name in attributes.items():
			setattr(group_obj, attr_name, _to_world_first(npz_data[field_name][scenario_pos]))
		setattr(sim_state, group_name, group_obj)
	return sim_state


def _lane_graph_zip_path_for_tfrecord(tfrecord_path: str, lane_graph_dir: str) -> Path:
	base = os.path.basename(tfrecord_path)
	return Path(lane_graph_dir) / f"{base}.lanegraph.zip"


def _sim_state_cache_path_for_tfrecord(tfrecord_path: str, cache_dir: str) -> Path:
	base = os.path.basename(tfrecord_path)
	return Path(cache_dir) / f"{base}.sim_state_cache.npz"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run manual annotation from cached sim_state npz and lane graph zip."
	)
	parser.add_argument("--tfrecord_dir", type=str, default="/zfsauton/scratch/eshau/womd/tf_example/training/", help="Directory containing TFRecord files.")
	parser.add_argument("--lane_graph_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs/", help="Directory containing matching lane graph zip files.")
	parser.add_argument("--cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/", help="Directory containing cached sim_state npz files.")
	parser.add_argument("--output_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/annotations/", help="Directory where JSONL files will be written.")
	parser.add_argument("--start_timestep", type=int, default=0, help="Start timestep for the annotation window.")
	parser.add_argument(
		"--trajectory_length",
		type=int,
		default=50,
		help="Length of the annotation window. end_timestep = start_timestep + trajectory_length.",
	)
	parser.add_argument("--max_tfrecords", type=int, default=None, help="Optional limit on how many TFRecord files to process.")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files.")
	parser.add_argument(
		"--num_workers",
		type=int,
		default=None,
		help="Number of parallel worker processes to use. Defaults to the number of CPU cores or TFRecord files, whichever is smaller.",
	)
	parser.add_argument("--cpu", "--no-gpu", action="store_true", help="Force CPU-only execution for this script (do not use GPU).")
	return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, dict):
		return {str(key): _to_jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(item) for item in value]
	return str(value)


def extract_annotation_for_scenario(
	sim_state,
	lane_graph,
	start_timestep: int,
	trajectory_length: int,
	world_idx: int = 0,
) -> dict[str, Any]:
	end_timestep = int(start_timestep) + int(trajectory_length)
	num_timesteps = int(sim_state.log_trajectory.x[world_idx].shape[1])
	end_timestep = min(end_timestep, num_timesteps)

	ego_idx = get_ego_idx(sim_state, world_idx)
	start_lane_id = int(get_current_lane_id(sim_state, start_timestep, world_idx=world_idx))
	end_lane_id = int(get_current_lane_id(sim_state, end_timestep - 1, world_idx=world_idx))
	direction = check_direction(sim_state, start_timestep, end_timestep, world_idx)
	start_speed, end_speed, avg_speed = check_speed(sim_state, start_timestep, end_timestep, world_idx)
	risk_objects = check_risk(sim_state, start_timestep, end_timestep, world_idx)
	goal_info = get_goal_info(sim_state, start_timestep, world_idx)

	turn_label = "straight"
	if lane_graph is not None and start_lane_id >= 0 and end_lane_id >= 0 and direction != "straight":
		turn_label = check_turn(
			sim_state,
			lane_graph,
			start_timestep,
			end_timestep,
			start_lane_id,
			end_lane_id,
			direction,
			world_idx,
		)

	traffic_lights = []
	if lane_graph is not None and start_lane_id >= 0:
		traffic_lights = check_traffic_light(
			sim_state,
			lane_graph,
			start_timestep,
			start_lane_id,
			world_idx,
		)
	lane_change = "unknown"
	left_lane_ids = []
	right_lane_ids = []
	front_vehicles, rear_vehicles, front_left_vehicles, rear_left_vehicles, front_right_vehicles, rear_right_vehicles = None, None, None, None, None, None
	if lane_graph is not None and start_lane_id >= 0:
		extended_lane_ids = extend_lane_id(lane_graph, start_lane_id)
		left_lane_ids = get_left_lane_id(lane_graph, start_lane_id)
		right_lane_ids = get_right_lane_id(lane_graph, start_lane_id)
		extended_left_lane_ids = []
		for left_lane_id in left_lane_ids:
			extended_left_lane_ids.extend(extend_lane_id(lane_graph, left_lane_id))
		extended_right_lane_ids = []
		for right_lane_id in right_lane_ids:
			extended_right_lane_ids.extend(extend_lane_id(lane_graph, right_lane_id))

		if end_lane_id in extended_lane_ids:
			lane_change = "none"
		elif end_lane_id in extended_left_lane_ids:
			lane_change = "left"
		elif end_lane_id in extended_right_lane_ids:
			lane_change = "right"
		else:
			lane_change = "unknown"

		vehicle_lane_ids = get_vehicle_lane_id(sim_state, start_timestep, extended_lane_ids + extended_left_lane_ids + extended_right_lane_ids, world_idx)

		front_vehicles, rear_vehicles = get_vehicle_on_target_lane(
			sim_state, start_timestep, extended_lane_ids, vehicle_lane_ids, world_idx
		)
		front_left_vehicles, rear_left_vehicles = get_vehicle_on_target_lane(
			sim_state, start_timestep, extended_left_lane_ids, vehicle_lane_ids, world_idx
		)
		front_right_vehicles, rear_right_vehicles = get_vehicle_on_target_lane(
			sim_state, start_timestep, extended_right_lane_ids, vehicle_lane_ids, world_idx
		)
		goal_lane_id = goal_info["lane_id"]
		if goal_lane_id in extended_lane_ids:
			goal_relation = "current_lane"
		elif goal_lane_id in extended_left_lane_ids:
			goal_relation = "left_lane"
		elif goal_lane_id in extended_right_lane_ids:
			goal_relation = "right_lane"
		else:
			goal_relation = "unknown"
		goal_info["goal_relation"] = goal_relation

	start_x = np.array(sim_state.log_trajectory.x[world_idx][ego_idx, start_timestep])
	start_y = np.array(sim_state.log_trajectory.y[world_idx][ego_idx, start_timestep])
	start_xy = np.stack([start_x, start_y], axis=-1)
	end_x = np.array(sim_state.log_trajectory.x[world_idx][ego_idx, end_timestep - 1])
	end_y = np.array(sim_state.log_trajectory.y[world_idx][ego_idx, end_timestep - 1])
	end_xy = np.stack([end_x, end_y], axis=-1)
	start_yaw = np.array(sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep])
	end_yaw = np.array(sim_state.log_trajectory.yaw[world_idx][ego_idx, end_timestep - 1])
	relative_end_position = end_xy - start_xy
	relative_end_position_rotated = np.array(
		[
			relative_end_position[0] * np.cos(-start_yaw) - relative_end_position[1] * np.sin(-start_yaw),
			relative_end_position[0] * np.sin(-start_yaw) + relative_end_position[1] * np.cos(-start_yaw),
		]
	)
	relative_end_yaw = end_yaw - start_yaw

	return {
		"scenario_window": {
			"start": int(start_timestep),
			"end": int(end_timestep),
		},
		"ego_motion": {
			"end_xy": _to_jsonable(relative_end_position_rotated),
			"end_yaw": _to_jsonable(relative_end_yaw),
			"start_speed": _to_jsonable(start_speed),
			"end_speed": _to_jsonable(end_speed),
			"avg_speed": _to_jsonable(avg_speed),
			"direction": direction,
			"turn_label": turn_label,
			"lane_change": lane_change,
		},
		"lane_context": {
			"start_lane_id": int(start_lane_id),
			"end_lane_id": int(end_lane_id),
			"left_lane_ids": _to_jsonable(left_lane_ids),
			"right_lane_ids": _to_jsonable(right_lane_ids),
			"traffic_lights": _to_jsonable(traffic_lights),
		},
		"vehicles": {
			"front": _to_jsonable(front_vehicles),
			"rear": _to_jsonable(rear_vehicles),
			"front_left": _to_jsonable(front_left_vehicles),
			"rear_left": _to_jsonable(rear_left_vehicles),
			"front_right": _to_jsonable(front_right_vehicles),
			"rear_right": _to_jsonable(rear_right_vehicles),
		},
		"risk_objects": _to_jsonable(risk_objects),
		"goal_info": _to_jsonable(goal_info),
	}

def make_instruction(annotation: dict) -> str:
	annot = annotation['ego_motion']
	# speed instruction
	if float(annot['start_speed']) < 0.1:
		if float(annot['end_speed']) < 0.1:
			speed_inst = 'stop'
		elif float(annot['end_speed']) > 0.1:
			speed_inst = 'stop and go'
	elif np.abs(float(annot['start_speed']) - float(annot['end_speed'])) / float(annot['start_speed']) < 0.1:
		speed_inst = 'maintain'
	elif float(annot['start_speed']) < float(annot['end_speed']):
		speed_inst = 'accelerate'
	else:
		speed_inst = 'decelerate'
	lane_is_known = True
	if speed_inst == 'stop':
		instruction = 'stop'
	else:
		instruction = ''
		if speed_inst == 'stop and go':
			instruction += 'stop for a while, and then '

		if annot['direction'] == 'straight':
			instruction += 'go straight '
		elif annot['direction'] == 'left':
			instruction += 'go left '
		elif annot['direction'] == 'right':
			instruction += 'go right '
		elif annot['direction'] == 'slight left':
			instruction += 'go slightly left '
		elif annot['direction'] == 'slight right':
			instruction += 'go slightly right '
		else:
			instruction += 'go '

		if 'left' in annot['turn_label']:
			instruction += 'to turn left '
		elif 'right' in annot['turn_label']:
			instruction += 'to turn right '
		elif 'u-turn' in annot['turn_label']:
			instruction += 'to make a U-turn '
		
		if 'left' in annot['lane_change']:
			instruction += 'while changing lane to the left '
		elif 'right' in annot['lane_change']:
			instruction += 'while changing lane to the right '
		elif 'none' in annot['lane_change']:
			instruction += 'while following current lane '
		else:
			lane_is_known = False
		
		if speed_inst == 'accelerate':
			instruction += 'and accelerating' if lane_is_known else 'while accelerating'
		elif speed_inst == 'decelerate':
			instruction += 'and slowing down' if lane_is_known else 'while slowing down'

	return instruction


def _iter_tfrecord_paths(tfrecord_dir: str) -> list[Path]:
	return [Path(path) for path in sorted(glob(str(Path(tfrecord_dir) / "*"))) if Path(path).is_file()]


def _build_lane_graph_from_zip(lane_graph_zf: zipfile.ZipFile, scenario_index: int) -> LaneGraphData:
	lane_graph_name = [
		name
		for name in lane_graph_zf.namelist()
		if name.startswith(f"{scenario_index:06d}") and name.endswith(".npz")
	][0]
	lane_graph_data = lane_graph_zf.read(lane_graph_name)
	with np.load(io.BytesIO(lane_graph_data)) as data:
		scenario_id = str(data["scenario_id"][0])
		lane_ids = data["lane_ids"].astype(np.int64)
		lane_starts = data["lane_starts"].astype(np.int64)
		lane_ends = data["lane_ends"].astype(np.int64)
		lane_id_to_node_range = {
			int(lid): (int(start), int(end))
			for lid, start, end in zip(lane_ids, lane_starts, lane_ends)
		}
		return LaneGraphData(
			scenario_id=scenario_id,
			lane_ids=lane_ids,
			nodes_xyz=data["nodes_xyz"].astype(np.float32),
			node_lane_ids=data["node_lane_ids"].astype(np.int64),
			node_point_indices=data["node_point_indices"].astype(np.int64),
			polyline_edges=data["polyline_edges"].astype(np.int64),
			successor_edges=data["successor_edges"].astype(np.int64),
			predecessor_edges=data["predecessor_edges"].astype(np.int64),
			left_neighbor_edges=data["left_neighbor_edges"].astype(np.int64),
			right_neighbor_edges=data["right_neighbor_edges"].astype(np.int64),
			lane_id_to_node_range=lane_id_to_node_range,
		)


def _process_single_tfrecord(
	tfrecord_path: str,
	lane_graph_dir: str,
	cache_dir: str,
	output_dir: str,
	start_timestep: int,
	trajectory_length: int,
	overwrite: bool,
	show_progress: bool,
) -> Path:
	signal.signal(signal.SIGINT, signal.SIG_IGN)
	tfrecord_path = str(tfrecord_path)
	output_path = Path(output_dir) / f"{Path(tfrecord_path).name}_t{int(start_timestep)}.jsonl"
	if output_path.exists() and not overwrite:
		return output_path

	cache_path = _sim_state_cache_path_for_tfrecord(tfrecord_path, cache_dir)
	if not cache_path.exists():
		raise FileNotFoundError(f"Missing sim_state cache file: {cache_path}")

	lane_graph_filename = _lane_graph_zip_path_for_tfrecord(tfrecord_path, lane_graph_dir)
	if not lane_graph_filename.exists():
		raise FileNotFoundError(f"Missing lane graph zip file: {lane_graph_filename}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with np.load(cache_path, allow_pickle=True) as cache_npz:
		scenario_indices = np.asarray(cache_npz["scenario_indices"]).astype(np.int32)
		with zipfile.ZipFile(lane_graph_filename, "r") as lane_graph_zf:
			with output_path.open("w", encoding="utf-8") as handle:
				pbar = (
					tqdm(total=len(scenario_indices), desc=f"Processing {Path(tfrecord_path).name}", unit="scenario")
					if show_progress
					else None
				)
				for pos, scenario_index in enumerate(scenario_indices.tolist()):
					sim_state = _scenario_from_cache(cache_npz, pos)
					lane_graph = _build_lane_graph_from_zip(lane_graph_zf, int(scenario_index))
					annotation = extract_annotation_for_scenario(
						sim_state,
						lane_graph,
						start_timestep=int(start_timestep),
						trajectory_length=int(trajectory_length),
						world_idx=0,
					)
					record: dict[str, Any] = {
						"tfrecord_path": Path(tfrecord_path).name,
						"scenario_index": int(scenario_index),
						"annotation": annotation,
						"instruction": make_instruction(annotation),
					}
					handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
					handle.write("\n")
					if pbar is not None:
						pbar.update(1)
				if pbar is not None:
					pbar.close()

	gc.collect()
	return output_path


def _terminate_pool_executor(executor: ProcessPoolExecutor) -> None:
	processes = getattr(executor, "_processes", None)
	if processes:
		for process in processes.values():
			try:
				process.terminate()
			except Exception:
				pass
	try:
		executor.shutdown(wait=False, cancel_futures=True)
	except Exception:
		pass


def run(args: argparse.Namespace) -> list[Path]:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tfrecord_paths = _iter_tfrecord_paths(args.tfrecord_dir)
	if not tfrecord_paths:
		raise FileNotFoundError(f"No TFRecord files found in {args.tfrecord_dir}.")

	selected_paths = tfrecord_paths if args.max_tfrecords is None else tfrecord_paths[: int(args.max_tfrecords)]
	max_workers = int(args.num_workers) if args.num_workers is not None else min(len(selected_paths), os.cpu_count() or 1)
	max_workers = max(1, min(max_workers, len(selected_paths)))
	use_multiprocessing = max_workers > 1

	written_paths: list[Path] = []
	if not use_multiprocessing:
		for tfrecord_path in selected_paths:
			written_paths.append(
				_process_single_tfrecord(
					tfrecord_path=str(tfrecord_path),
					lane_graph_dir=args.lane_graph_dir,
					cache_dir=args.cache_dir,
					output_dir=str(output_dir),
					start_timestep=int(args.start_timestep),
					trajectory_length=int(args.trajectory_length),
					overwrite=bool(args.overwrite),
					show_progress=True,
				)
			)
		return written_paths

	executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
	try:
		futures = [
			executor.submit(
				_process_single_tfrecord,
				str(tfrecord_path),
				args.lane_graph_dir,
				args.cache_dir,
				str(output_dir),
				int(args.start_timestep),
				int(args.trajectory_length),
				bool(args.overwrite),
				i % max_workers == 0,
			)
			for i, tfrecord_path in enumerate(selected_paths)
		]
		for future in as_completed(futures):
			written_paths.append(future.result())
	except KeyboardInterrupt:
		_terminate_pool_executor(executor)
		raise
	finally:
		executor.shutdown(wait=True, cancel_futures=True)
	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()