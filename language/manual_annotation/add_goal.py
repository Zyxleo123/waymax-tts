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
	get_goal,
	check_goal_is_behind
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
	parser.add_argument("--start_from", type=int, default=0)
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
	tfrecord_paths = tfrecord_paths[args.start_from:]

	written_paths: list[Path] = []
	num_processed = 0
	for tfrecord_path in tfrecord_paths:
		output_path = output_dir / f"{tfrecord_path.name}_t{args.start_timestep}.jsonl"
		ds_cfg = dataclasses.replace(
			waymax_config.WOD_1_3_1_TRAINING,
			path=str(tfrecord_path),
			max_num_objects=int(args.max_num_objects),
			batch_dims=(1,),
			shuffle_seed=0,
		)

		records: list[dict[str, Any]] = []
		with output_path.open("r", encoding="utf-8") as f:
				for i, item in enumerate(f):
					record = json.loads(item)
					records.append(record)	
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
			goal = get_goal(sim_state, world_idx=0)
			goal_is_behind = check_goal_is_behind(sim_state, goal, args.start_timestep, world_idx=0)
			
			records[scenario_index]["annotation"]["goal"] = {
				"position": _to_jsonable(goal),
                "is_behind": _to_jsonable(goal_is_behind)
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
