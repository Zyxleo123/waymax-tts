from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Iterable


# Reduce TensorFlow/XLA startup noise and keep TF from competing for GPU memory.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import numpy as np

from data.preprocess import preprocess_simulator_state
from data.types import PreprocessConfig


def _parse_int_list(tokens: Iterable[str] | None) -> list[int] | None:
	if tokens is None:
		return None
	parsed: list[int] = []
	for token in tokens:
		for part in token.split(","):
			part = part.strip()
			if part:
				parsed.append(int(part))
	return parsed or None


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


def _build_waymax_dataset_config(tfrecord_path: str, args) -> object:
	from waymax import config as waymax_config

	base_cfg = waymax_config.WOD_1_3_1_TRAINING
	cfg_fields = {field.name for field in dataclasses.fields(base_cfg)}

	replace_kwargs: dict[str, object] = {
		"path": str(tfrecord_path),
		"max_num_objects": int(args.max_num_objects),
		"batch_dims": (int(args.batch_size),),
		"shuffle_seed": int(args.shuffle_seed),
		"shuffle_buffer_size": int(args.shuffle_buffer_size),
	}

	def _set_if_supported(field: str, value: object) -> None:
		if field in cfg_fields:
			replace_kwargs[field] = value

	_set_if_supported("num_shards", int(args.dataset_num_shards))
	_set_if_supported("include_sdc_paths", bool(args.dataset_include_sdc_paths))

	max_num_rg_points = int(args.dataset_max_num_rg_points)
	if "max_num_rg_points" in cfg_fields:
		replace_kwargs["max_num_rg_points"] = max_num_rg_points

	return dataclasses.replace(base_cfg, **replace_kwargs)


def _build_preprocess_config(args) -> PreprocessConfig:
	return PreprocessConfig(
		model_dt=float(args.model_dt),
		world_dt_fallback=float(args.world_dt_fallback),
		max_range=float(args.max_range),
		ego_range=float(args.ego_range),
		max_velocity=float(args.max_velocity),
		max_width=float(args.max_width),
		max_tl_points=int(args.max_tl_points),
		num_map_type_classes=int(args.num_map_type_classes),
		predict_horizon=int(args.predict_horizon),
		max_segments=int(args.max_segments),
		max_points_per_segment=int(args.max_points_per_segment),
		num_object_types=int(args.num_object_types),
		inst_dim=int(args.inst_dim),
	)


def _iter_simulator_state_batches(cfg) -> Iterable[object]:
	import tensorflow as tf
	from waymax import dataloader
	from waymax.dataloader import womd_factories

	tf.get_logger().setLevel("ERROR")
	try:
		tf.autograph.set_verbosity(0)
	except Exception:
		pass
	try:
		tf.config.set_visible_devices([], "GPU")
	except Exception:
		pass

	raw_ds = tf.data.TFRecordDataset([cfg.path]).batch(int(cfg.batch_dims[0]), drop_remainder=False)
	for serialized_batch in raw_ds:
		processed = dataloader.preprocess_serialized_womd_data(serialized_batch, cfg)
		yield womd_factories.simulator_state_from_womd_dict(
			processed, include_sdc_paths=cfg.include_sdc_paths
		)


def _append_tree(accum: dict[str, list[np.ndarray]], tree: dict[str, jax.Array]) -> None:
	for key, value in tree.items():
		if key not in accum:
			accum[key] = []
		accum[key].append(np.asarray(jax.device_get(value)))


def _concat_tree(accum: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
	merged: dict[str, np.ndarray] = {}
	for key, chunks in accum.items():
		if not chunks:
			continue
		merged[key] = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
	return merged


def _save_cache_npz(
	output_path: Path,
	features: dict[str, np.ndarray],
	aux: dict[str, np.ndarray],
	metadata: dict[str, np.ndarray],
) -> None:
	payload: dict[str, np.ndarray] = {}
	payload.update({f"features/{k}": v for k, v in features.items()})
	payload.update({f"aux/{k}": v for k, v in aux.items()})
	payload.update(metadata)
	np.savez(output_path, **payload)


def build_cache_for_tfrecord(
	*,
	tfrecord_path: str,
	output_dir: Path,
	preprocess_cfg: PreprocessConfig,
	target_timestep: int,
	goal_step_override: int | None,
	seed: int,
	args,
) -> Path:
	cfg = _build_waymax_dataset_config(tfrecord_path, args)
	rng = jax.random.PRNGKey(int(seed))
	
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{Path(tfrecord_path).name}_t{target_timestep}.npz"
	if output_path.exists():
		print(f"Cache already exists, skipping: {output_path}")
		return output_path

	feature_chunks: dict[str, list[np.ndarray]] = {}
	aux_chunks: dict[str, list[np.ndarray]] = {}
	scenario_indices: list[np.ndarray] = []

	scenario_counter = 0
	for state_batch in _iter_simulator_state_batches(cfg):
		batch_size = int(state_batch.log_trajectory.x.shape[0])
		batch_indices = np.arange(scenario_counter, scenario_counter + batch_size, dtype=np.int32)
		scenario_indices.append(batch_indices)
		scenario_counter += batch_size

		rng, rng_pre = jax.random.split(rng)
		pre_batch, _ = preprocess_simulator_state(
			state_batch,
			rng_pre,
			preprocess_cfg,
			anchor_step_override=int(target_timestep),
			goal_step_override=goal_step_override,
		)

		_append_tree(feature_chunks, pre_batch.features)
		_append_tree(aux_chunks, pre_batch.aux)

	features = _concat_tree(feature_chunks)
	aux = _concat_tree(aux_chunks)
	if not features:
		raise ValueError(f"No features produced for tfrecord: {tfrecord_path}")

	scenario_index = (
		scenario_indices[0]
		if len(scenario_indices) == 1
		else np.concatenate(scenario_indices, axis=0)
	)
	metadata = {
		"scenario_index": scenario_index,
		"timestep": np.full_like(scenario_index, int(target_timestep)),
		"tfrecord_path": np.asarray(str(tfrecord_path), dtype=np.bytes_),
	}

	_save_cache_npz(output_path, features, aux, metadata)
	return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Build Waymo cache .npz files per TFRecord shard.")
	parser.add_argument("--tfrecord_dir", type=str, default=None, help="Directory with TFRecord shards.")
	parser.add_argument(
		"--tfrecord_paths",
		type=str,
		nargs="*",
		default=None,
		help="Optional explicit TFRecord shard paths; overrides --tfrecord_dir.",
	)
	parser.add_argument(
		"--file_indices",
		type=str,
		nargs="*",
		default=None,
		help="Optional shard indices, e.g. --file_indices 0 1 2 or --file_indices 0,1,2.",
	)
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to write .npz cache files.")
	parser.add_argument("--target_timestep", type=int, required=True, help="Anchor timestep to cache.")
	parser.add_argument("--goal_step", type=int, default=None, help="Optional goal timestep override.")
	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--max_num_objects", type=int, default=128)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--shuffle_buffer_size", type=int, default=0)
	parser.add_argument("--dataset_num_shards", type=int, default=1)
	parser.add_argument("--dataset_max_num_rg_points", type=int, default=30000)
	parser.add_argument("--dataset_include_sdc_paths", action="store_true")

	parser.add_argument("--model_dt", type=float, default=0.2)
	parser.add_argument("--world_dt_fallback", type=float, default=0.1)
	parser.add_argument("--predict_horizon", type=int, default=25)
	parser.add_argument("--max_range", type=float, default=100.0)
	parser.add_argument("--ego_range", type=float, default=100.0)
	parser.add_argument("--max_velocity", type=float, default=25.0)
	parser.add_argument("--max_width", type=float, default=10.0)
	parser.add_argument("--max_tl_points", type=int, default=16)
	parser.add_argument("--num_map_type_classes", type=int, default=21)
	parser.add_argument("--max_segments", type=int, default=128)
	parser.add_argument("--max_points_per_segment", type=int, default=128)
	parser.add_argument("--num_object_types", type=int, default=8)
	parser.add_argument("--inst_dim", type=int, default=256)
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	file_indices = _parse_int_list(args.file_indices)
	if args.tfrecord_paths:
		tfrecord_paths = tuple(args.tfrecord_paths)
	else:
		if not args.tfrecord_dir:
			raise ValueError("Provide --tfrecord_dir or --tfrecord_paths.")
		tfrecord_paths = _resolve_tfrecord_paths(args.tfrecord_dir, file_indices)

	preprocess_cfg = _build_preprocess_config(args)
	output_dir = Path(args.output_dir)

	for tfrecord_path in tfrecord_paths:
		output_path = build_cache_for_tfrecord(
			tfrecord_path=tfrecord_path,
			output_dir=output_dir,
			preprocess_cfg=preprocess_cfg,
			target_timestep=int(args.target_timestep),
			goal_step_override=args.goal_step,
			seed=int(args.seed),
			args=args,
		)
		print(f"saved: {output_path}")


if __name__ == "__main__":
	main()
