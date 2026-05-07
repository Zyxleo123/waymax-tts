from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from vla.qa_dataloader import QADataLoaderConfig, _build_waymax_dataset_config, _resolve_tfrecord_paths
from viz.render import _load_scenario_state_fast


def _parse_int_csv(raw: str | None) -> list[int] | None:
	if raw is None:
		return None
	values = [part.strip() for part in raw.split(",") if part.strip()]
	if not values:
		return None
	return [int(value) for value in values]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Load Waymax scenarios directly and summarize lane segment counts from roadgraph_points ids."
	)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--tfrecord_path", type=str, default=None, help="Single Waymax tfrecord shard path.")
	group.add_argument("--tfrecord_dir", type=str, default=None, help="Directory containing Waymax tfrecord shards.")
	parser.add_argument("--file_indices", type=str, default=None, help="Optional comma-separated shard indices when using --tfrecord_dir.")
	parser.add_argument("--scenario_start", type=int, default=0, help="First scenario index to inspect in each shard.")
	parser.add_argument("--num_scenarios", type=int, default=100, help="How many sequential scenarios to inspect per shard.")
	parser.add_argument("--batch_size", type=int, default=1, help="Waymax batch size used to build the dataset config.")
	parser.add_argument("--max_num_objects", type=int, default=64, help="Waymax max_num_objects used to build the dataset config.")
	parser.add_argument("--shuffle_seed", type=int, default=0, help="Waymax shuffle seed used to build the dataset config.")
	parser.add_argument("--shuffle_buffer_size", type=int, default=1024, help="Waymax shuffle buffer size used to build the dataset config.")
	parser.add_argument("--dataset_num_shards", type=int, default=1, help="Waymax dataset shard count used to build the dataset config.")
	parser.add_argument("--include_sdc_paths", action="store_true", help="Preserve SDC path fields if supported by the Waymax config.")
	parser.add_argument("--output_json", type=str, default=None, help="Optional path to save the summary as JSON.")
	return parser.parse_args()


def _summarize(values: list[int], *, name: str) -> dict[str, Any]:
	array = np.asarray(values, dtype=np.float64)
	if array.size == 0:
		return {"name": name, "count": 0}
	return {
		"name": name,
		"count": int(array.size),
		"mean": float(array.mean()),
		"std": float(array.std(ddof=0)),
		"min": float(array.min()),
		"p25": float(np.percentile(array, 25)),
		"median": float(np.percentile(array, 50)),
		"p75": float(np.percentile(array, 75)),
		"p90": float(np.percentile(array, 90)),
		"p95": float(np.percentile(array, 95)),
		"max": float(array.max()),
	}


def _roadgraph_segment_stats(state: Any) -> tuple[int, list[int]]:
	roadgraph_points = getattr(state, "roadgraph_points", None)
	if roadgraph_points is None:
		raise ValueError("Loaded scenario does not contain roadgraph_points.")

	ids = np.asarray(roadgraph_points.ids)
	valid = np.asarray(roadgraph_points.valid).astype(bool)
	if ids.ndim == 2:
		ids = ids[0]
	if valid.ndim == 2:
		valid = valid[0]

	# get roadgraph coordinates
	rx = np.asarray(roadgraph_points.x)
	ry = np.asarray(roadgraph_points.y)
	if rx.ndim == 2:
		rx = rx[0]
	if ry.ndim == 2:
		ry = ry[0]

	# determine ego position at timestep 0 (if available)
	traj_x = getattr(state.log_trajectory, "x", None)
	traj_y = getattr(state.log_trajectory, "y", None)
	if traj_x is None or traj_y is None:
		# no trajectory info; fallback to distance-unfiltered valid points
		valid_ids = ids[valid]
	else:
		x_arr = np.asarray(traj_x)
		y_arr = np.asarray(traj_y)
		# extract ego index
		is_sdc = np.asarray(state.object_metadata.is_sdc)
		if is_sdc.ndim == 2:
			is_sdc = is_sdc[0]
		if is_sdc.size == 0:
			valid_ids = ids[valid]
		else:
			ego_idx = int(np.argmax(is_sdc))
			# get ego position at timestep 0
			if x_arr.ndim == 3:
				ego_x = float(x_arr[0, ego_idx, 0])
				ego_y = float(y_arr[0, ego_idx, 0])
			else:
				ego_x = float(x_arr[ego_idx, 0])
				ego_y = float(y_arr[ego_idx, 0])
			# compute distances to roadgraph points and filter within 30m
			dist = np.sqrt((rx - ego_x) ** 2 + (ry - ego_y) ** 2)
			range_mask = dist <= 30.0
			valid_ids = ids[valid & range_mask]

	if valid_ids.size == 0:
		return 0, []

	unique_ids, counts = np.unique(valid_ids, return_counts=True)
	segment_count = int(unique_ids.size)
	point_counts = [int(value) for value in counts.tolist()]
	return segment_count, point_counts


def _build_dataset_cfg(args: argparse.Namespace, tfrecord_path: str):
	loader_cfg = QADataLoaderConfig(
		tfrecord_paths=(tfrecord_path,),
		qa_dir=None,
		batch_size=int(args.batch_size),
		max_num_objects=int(args.max_num_objects),
		shuffle_seed=int(args.shuffle_seed),
		shuffle_buffer_size=int(args.shuffle_buffer_size),
		dataset_num_shards=int(args.dataset_num_shards),
		include_sdc_paths=bool(args.include_sdc_paths),
	)
	return _build_waymax_dataset_config(loader_cfg, tfrecord_path)


def _resolve_shards(args: argparse.Namespace) -> tuple[str, ...]:
	file_indices = _parse_int_csv(args.file_indices)
	if args.tfrecord_path is not None:
		return (str(Path(args.tfrecord_path)),)
	return _resolve_tfrecord_paths(str(args.tfrecord_dir), file_indices)


def _inspect_shard(args: argparse.Namespace, tfrecord_path: str) -> dict[str, Any]:
	ds_cfg = _build_dataset_cfg(args, tfrecord_path)
	scenarios: list[dict[str, Any]] = []
	segment_counts: list[int] = []
	point_counts_all: list[int] = []

	for scenario_index in range(int(args.scenario_start), int(args.scenario_start) + int(args.num_scenarios)):
		try:
			state = _load_scenario_state_fast(ds_cfg, scenario_index)
		except Exception:
			break

		segment_count, point_counts = _roadgraph_segment_stats(state)
		segment_counts.append(segment_count)
		point_counts_all.extend(point_counts)
		scenarios.append(
			{
				"scenario_index": int(scenario_index),
				"segment_count": int(segment_count),
				"point_counts_per_segment": point_counts,
			}
		)

	return {
		"tfrecord_path": tfrecord_path,
		"num_scenarios": int(len(scenarios)),
		"segment_counts": _summarize(segment_counts, name="segment_counts"),
		"point_counts_per_segment": _summarize(point_counts_all, name="point_counts_per_segment"),
		"scenarios": scenarios,
	}


def main() -> None:
	args = _parse_args()
	shards = _resolve_shards(args)
	shard_results = [_inspect_shard(args, shard_path) for shard_path in shards]

	all_segment_counts = [scenario["segment_count"] for shard in shard_results for scenario in shard["scenarios"]]
	all_point_counts = [point for shard in shard_results for scenario in shard["scenarios"] for point in scenario["point_counts_per_segment"]]

	summary = {
		"config": {
			"tfrecord_path": args.tfrecord_path,
			"tfrecord_dir": args.tfrecord_dir,
			"file_indices": _parse_int_csv(args.file_indices),
			"scenario_start": int(args.scenario_start),
			"num_scenarios": int(args.num_scenarios),
			"batch_size": int(args.batch_size),
			"max_num_objects": int(args.max_num_objects),
			"shuffle_seed": int(args.shuffle_seed),
			"shuffle_buffer_size": int(args.shuffle_buffer_size),
			"dataset_num_shards": int(args.dataset_num_shards),
			"include_sdc_paths": bool(args.include_sdc_paths),
		},
		"overall": {
			"num_tfrecord_shards": int(len(shards)),
			"num_scenarios": int(len(all_segment_counts)),
			"segment_counts": _summarize(all_segment_counts, name="segment_counts"),
			"point_counts_per_segment": _summarize(all_point_counts, name="point_counts_per_segment"),
		},
		"shards": shard_results,
	}

	print(json.dumps(summary["overall"], indent=2, sort_keys=True))

	if args.output_json is not None:
		output_path = Path(args.output_json)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
	main()
