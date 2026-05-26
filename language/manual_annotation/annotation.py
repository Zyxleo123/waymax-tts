from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import os

# Allow forcing CPU-only mode via CLI flag `--cpu` or `--no-gpu`, or env var WAYMAX_FORCE_CPU.
# This must be set before importing modules that may initialize JAX/TensorFlow GPU devices.
if any(arg in ("--cpu", "--no-gpu") for arg in sys.argv) or os.environ.get("WAYMAX_FORCE_CPU", "").lower() in (
	"1",
	"true",
	"yes",
):
	# Prefer JAX explicit platform and hide CUDA devices from CUDA-enabled libraries.
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

from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_fast
from lane_graph.lane_graph_utils import LaneGraphLoader
from language.manual_annotation.helpers import (
	check_direction,
	check_risk,
	check_speed,
	check_traffic_light,
	check_turn,
	extend_lane_id,
	get_current_lane_id,
	get_ego_idx,
	get_left_lane_id,
	get_right_lane_id,
)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Load Waymax scenarios, extract annotation dictionaries, and save one JSONL per TFRecord."
	)
	parser.add_argument("--tfrecord_dir", type=str, default="/zfsauton/scratch/eshau/womd/tf_example/training/", help="Directory containing TFRecord files.")
	parser.add_argument("--lane_graph_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs/", help="Directory containing matching lane graph zip files.")
	parser.add_argument("--output_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/annotations/", help="Directory where JSONL files will be written.")
	parser.add_argument("--start_timestep", type=int, default=10, help="Start timestep for the annotation window.")
	parser.add_argument(
		"--trajectory_length",
		type=int,
		default=50,
		help="Length of the annotation window. end_timestep = start_timestep + trajectory_length.",
	)
	parser.add_argument("--max_num_objects", type=int, default=128, help="Maximum number of objects to load per scenario.")
	parser.add_argument("--max_tfrecords", type=int, default=None, help="Optional limit on how many TFRecord files to process.")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files.")
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


def _lane_graph_summary(lane_graph: Any | None) -> dict[str, Any]:
	if lane_graph is None:
		return {
			"available": False,
			"scenario_id": None,
			"num_nodes": None,
			"num_lanes": None,
		}

	return {
		"available": True,
		"scenario_id": lane_graph.scenario_id,
		"num_nodes": int(lane_graph.nodes_xyz.shape[0]),
		"num_lanes": int(len(lane_graph.lane_id_to_node_range)),
	}


def extract_annotation_for_scenario(
	sim_state,
	lane_graph,
	start_timestep: int,
	trajectory_length: int,
	world_idx: int = 0,
) -> dict[str, Any]:
	end_timestep = int(start_timestep) + int(trajectory_length)
	num_timesteps = int(sim_state.log_trajectory.xy[world_idx].shape[0])

	ego_idx = get_ego_idx(sim_state, world_idx)
	start_lane_id = int(get_current_lane_id(sim_state, start_timestep, world_idx=world_idx))
	end_lane_id = int(get_current_lane_id(sim_state, end_timestep - 1, world_idx=world_idx))
	direction = check_direction(sim_state, start_timestep, end_timestep, world_idx)
	start_speed, end_speed, avg_speed = check_speed(sim_state, start_timestep, end_timestep, world_idx)
	risk_objects = check_risk(sim_state, start_timestep, end_timestep, world_idx)

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
	if lane_graph is not None and start_lane_id >= 0:
		extended_lane_ids = extend_lane_id(lane_graph, start_lane_id)
		if end_lane_id in extended_lane_ids:
			lane_change = "none"
		else:
			left_lane_ids = get_left_lane_id(lane_graph, start_lane_id)
			right_lane_ids = get_right_lane_id(lane_graph, start_lane_id)
			ptr = 0
			while ptr < len(left_lane_ids) or ptr < len(right_lane_ids):
				if ptr < len(left_lane_ids):
					extended_left_lane_ids = extend_lane_id(lane_graph, left_lane_ids[ptr])
					if end_lane_id in extended_left_lane_ids:
						lane_change = f"left_{ptr+1}"
						break
				if ptr < len(right_lane_ids):
					extended_right_lane_ids = extend_lane_id(lane_graph, right_lane_ids[ptr])
					if end_lane_id in extended_right_lane_ids:
						lane_change = f"right_{ptr+1}"
						break
				ptr += 1

	left_lane_ids = []
	right_lane_ids = []
	extended_lane_ids = []
	if lane_graph is not None and start_lane_id >= 0:
		left_lane_ids = get_left_lane_id(lane_graph, start_lane_id)
		right_lane_ids = get_right_lane_id(lane_graph, start_lane_id)
		extended_lane_ids = extend_lane_id(lane_graph, start_lane_id)

	start_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, start_timestep]
	end_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, end_timestep - 1]
	start_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep]
	end_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, end_timestep - 1]

	return {
		"scenario_window": {
			"start": int(start_timestep),
			"end": int(end_timestep),
		},
		"ego_motion": {
			"start_xy": _to_jsonable(start_xy),
			"end_xy": _to_jsonable(end_xy),
			"start_yaw": _to_jsonable(start_yaw),
			"end_yaw": _to_jsonable(end_yaw),
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
		"risk_objects": _to_jsonable(risk_objects),
	}


def _iter_tfrecord_paths(tfrecord_dir: str) -> list[Path]:
	return [Path(path) for path in sorted(glob(str(Path(tfrecord_dir) / "*"))) if Path(path).is_file()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
			handle.write("\n")


def run(args: argparse.Namespace) -> list[Path]:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tfrecord_paths = _iter_tfrecord_paths(args.tfrecord_dir)
	if not tfrecord_paths:
		raise FileNotFoundError(f"No TFRecord files found in {args.tfrecord_dir}.")

	lane_graph_loader = LaneGraphLoader(args.lane_graph_dir)
	written_paths: list[Path] = []
	num_processed = 0
	for tfrecord_path in tfrecord_paths:
		output_path = output_dir / f"{tfrecord_path.name}_t{args.start_timestep}.jsonl"
		if output_path.exists() and not args.overwrite:
			written_paths.append(output_path)
			continue
		# touch the file to reserve the name while we process
		output_path.touch(exist_ok=True)

		ds_cfg = dataclasses.replace(
			waymax_config.WOD_1_3_1_TRAINING,
			path=str(tfrecord_path),
			max_num_objects=int(args.max_num_objects),
			batch_dims=(1,),
			shuffle_seed=0,
		)

		records: list[dict[str, Any]] = []
		scenario_index = 0
		consecutive_failures = 0
		pbar = tqdm(total=None, desc=f"Processing {tfrecord_path.name}", unit="scenario")

		while True:
			try:
				sim_state = load_scenario_state_fast(ds_cfg, scenario_index)
			except Exception as exc:
				message = str(exc)
				if "out of range" in message:
					break
				consecutive_failures += 1
				if consecutive_failures >= 3:
					break
				scenario_index += 1
				continue

			consecutive_failures = 0
			lane_graph = lane_graph_loader.get_lane_graph_for_scenario(str(tfrecord_path), scenario_index)

			record: dict[str, Any] = {
				"tfrecord_path": str(tfrecord_path).split("/")[-1],
				"scenario_index": int(scenario_index),
				"annotation": extract_annotation_for_scenario(
					sim_state,
					lane_graph,
					start_timestep=int(args.start_timestep),
					trajectory_length=int(args.trajectory_length),
					world_idx=0,
				),
			}
			records.append(record)
			scenario_index += 1
			pbar.update(1)
		pbar.close()
		_write_jsonl(output_path, records)
		written_paths.append(output_path)
		num_processed += 1

		if args.max_tfrecords is not None and num_processed >= int(args.max_tfrecords):
			break

	return written_paths


def main() -> None:
	args = _parse_args()
	run(args)


if __name__ == "__main__":
	main()
