from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow forcing CPU-only mode via CLI flag `--cpu` / `--no-gpu`, or env var WAYMAX_FORCE_CPU.
# This must be set before importing modules that may initialize GPU backends.
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

from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_fast


# Fields explicitly used by language/manual_annotation/annotation.py and helpers.py.
SIM_STATE_FIELD_EXTRACTORS = {
	"object_metadata_is_sdc": lambda sim_state, world_idx: np.asarray(sim_state.object_metadata.is_sdc[world_idx]),
	"object_metadata_object_types": lambda sim_state, world_idx: np.asarray(sim_state.object_metadata.object_types[world_idx]),
	"log_trajectory_timestamp_micros": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.timestamp_micros[world_idx]),
	"log_trajectory_x": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.x[world_idx]),
	"log_trajectory_y": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.y[world_idx]),
	"log_trajectory_yaw": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.yaw[world_idx]),
	"log_trajectory_valid": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.valid[world_idx]),
	"log_trajectory_vel_x": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.vel_x[world_idx]),
	"log_trajectory_vel_y": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.vel_y[world_idx]),
	"log_trajectory_speed": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.speed[world_idx]),
	"log_trajectory_length": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.length[world_idx]),
	"log_trajectory_width": lambda sim_state, world_idx: np.asarray(sim_state.log_trajectory.width[world_idx]),
	"log_traffic_light_state": lambda sim_state, world_idx: np.asarray(sim_state.log_traffic_light.state[world_idx]),
	"log_traffic_light_x": lambda sim_state, world_idx: np.asarray(sim_state.log_traffic_light.x[world_idx]),
	"log_traffic_light_y": lambda sim_state, world_idx: np.asarray(sim_state.log_traffic_light.y[world_idx]),
	"log_traffic_light_lane_ids": lambda sim_state, world_idx: np.asarray(sim_state.log_traffic_light.lane_ids[world_idx]),
	"log_traffic_light_valid": lambda sim_state, world_idx: np.asarray(sim_state.log_traffic_light.valid[world_idx]),
	"roadgraph_points_x": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.x[world_idx]),
	"roadgraph_points_y": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.y[world_idx]),
	"roadgraph_points_ids": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.ids[world_idx]),
	"roadgraph_points_types": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.types[world_idx]),
	"roadgraph_points_valid": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.valid[world_idx]),
	"roadgraph_points_dir_x": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.dir_x[world_idx]),
	"roadgraph_points_dir_y": lambda sim_state, world_idx: np.asarray(sim_state.roadgraph_points.dir_y[world_idx]),
}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Cache sim_state attributes used by manual annotation into one npz per TFRecord."
	)
	parser.add_argument(
		"--tfrecord_dir",
		type=str,
		default="/zfsauton/scratch/eshau/womd/tf_example/training/",
		help="Directory containing TFRecord files.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/",
		help="Directory where cached npz files will be written.",
	)
	parser.add_argument(
		"--max_num_objects",
		type=int,
		default=128,
		help="Maximum number of objects to load per scenario while reading TFRecord.",
	)
	parser.add_argument(
		"--max_tfrecords",
		type=int,
		default=None,
		help="Optional limit on how many TFRecord files to process.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npz files.")
	parser.add_argument(
		"--world_idx",
		type=int,
		default=0,
		help="World index to cache (default: 0).",
	)
	parser.add_argument(
		"--cpu",
		"--no-gpu",
		action="store_true",
		help="Force CPU-only execution for this script (do not use GPU).",
	)
	parser.add_argument(
		"--start_from",
		type=int,
		default=0,
    )
	return parser.parse_args()


def _iter_tfrecord_paths(tfrecord_dir: str) -> list[Path]:
	return [Path(path) for path in sorted(glob(str(Path(tfrecord_dir) / "*"))) if Path(path).is_file()]


def _pack_per_scenario_arrays(arrays: list[np.ndarray]) -> np.ndarray:
	if not arrays:
		return np.empty((0,), dtype=np.float32)
	try:
		return np.stack(arrays, axis=0)
	except ValueError:
		packed = np.empty((len(arrays),), dtype=object)
		for i, item in enumerate(arrays):
			packed[i] = item
		return packed


def _extract_used_sim_state_fields(sim_state, world_idx: int) -> dict[str, np.ndarray]:
	return {
		field_name: extractor(sim_state, world_idx)
		for field_name, extractor in SIM_STATE_FIELD_EXTRACTORS.items()
	}


def cache_single_tfrecord(
	tfrecord_path: Path,
	output_dir: Path,
	max_num_objects: int,
	world_idx: int,
	overwrite: bool,
) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{tfrecord_path.name}.sim_state_cache.npz"
	if output_path.exists() and not overwrite:
		return output_path

	ds_cfg = dataclasses.replace(
		waymax_config.WOD_1_3_1_TRAINING,
		path=str(tfrecord_path),
		max_num_objects=int(max_num_objects),
		batch_dims=(1,),
		shuffle_seed=0,
	)

	per_field: dict[str, list[np.ndarray]] = {
		field_name: [] for field_name in SIM_STATE_FIELD_EXTRACTORS
	}
	scenario_indices: list[int] = []

	scenario_index = 0
	consecutive_failures = 0
	pbar = tqdm(total=None, desc=f"Caching {tfrecord_path.name}", unit="scenario")
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
		extracted = _extract_used_sim_state_fields(sim_state, world_idx=world_idx)
		for field_name, value in extracted.items():
			per_field[field_name].append(value)
		scenario_indices.append(scenario_index)
		scenario_index += 1
		pbar.update(1)

	pbar.close()

	packed: dict[str, np.ndarray] = {
		"tfrecord_path": np.asarray(str(tfrecord_path), dtype=np.str_),
		"tfrecord_name": np.asarray(tfrecord_path.name, dtype=np.str_),
		"world_idx": np.asarray(int(world_idx), dtype=np.int32),
		"num_scenarios": np.asarray(len(scenario_indices), dtype=np.int32),
		"scenario_indices": np.asarray(scenario_indices, dtype=np.int32),
		"cached_field_names": np.asarray(sorted(SIM_STATE_FIELD_EXTRACTORS.keys()), dtype=np.str_),
	}
	for field_name, values in per_field.items():
		packed[field_name] = _pack_per_scenario_arrays(values)

	np.savez_compressed(output_path, **packed)
	return output_path


def run(args: argparse.Namespace) -> list[Path]:
	tfrecord_paths = _iter_tfrecord_paths(args.tfrecord_dir)
	if not tfrecord_paths:
		raise FileNotFoundError(f"No TFRecord files found in {args.tfrecord_dir}.")
	tfrecord_paths = tfrecord_paths[args.start_from :]

	selected_paths = tfrecord_paths if args.max_tfrecords is None else tfrecord_paths[: int(args.max_tfrecords)]
	output_dir = Path(args.output_dir)

	written_paths: list[Path] = []
	for tfrecord_path in selected_paths:
		written_paths.append(
			cache_single_tfrecord(
				tfrecord_path=tfrecord_path,
				output_dir=output_dir,
				max_num_objects=int(args.max_num_objects),
				world_idx=int(args.world_idx),
				overwrite=bool(args.overwrite),
			)
		)
	return written_paths


def main() -> None:
	args = _parse_args()
	written_paths = run(args)
	for path in written_paths:
		print(path)


if __name__ == "__main__":
	main()
	