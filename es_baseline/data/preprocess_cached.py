from __future__ import annotations

"""NumPy preprocessing for cached Waymo scenario fields.

This mirrors :mod:`data.preprocess` but reads the per-scenario arrays stored in
the cached ``.npz`` files produced by
``language/manual_annotation/cached_annotation/caching.py``.

The public API returns plain NumPy dictionaries instead of a flax dataclass.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from waymax.datatypes.roadgraph import MapElementIds

from data.types import PreprocessConfig


LANE_MAP_ELEMENT_TYPES = (
	int(MapElementIds.LANE_FREEWAY),
	int(MapElementIds.LANE_SURFACE_STREET),
)


_CACHE_FIELD_NAMES = (
	"object_metadata_is_sdc",
	"object_metadata_object_types",
	"log_trajectory_timestamp_micros",
	"log_trajectory_x",
	"log_trajectory_y",
	"log_trajectory_yaw",
	"log_trajectory_valid",
	"log_trajectory_vel_x",
	"log_trajectory_vel_y",
	"log_trajectory_speed",
	"log_trajectory_length",
	"log_trajectory_width",
	"log_traffic_light_state",
	"log_traffic_light_x",
	"log_traffic_light_y",
	"log_traffic_light_lane_ids",
	"log_traffic_light_valid",
	"roadgraph_points_x",
	"roadgraph_points_y",
	"roadgraph_points_ids",
	"roadgraph_points_types",
	"roadgraph_points_valid",
	"roadgraph_points_dir_x",
	"roadgraph_points_dir_y",
	"inst_features",
	"inst_valid"
)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
	return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_xy(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
	c = np.cos(-heading)[..., None]
	s = np.sin(-heading)[..., None]
	x = xy[..., 0]
	y = xy[..., 1]
	xr = x * c[..., 0] - y * s[..., 0]
	yr = x * s[..., 0] + y * c[..., 0]
	return np.stack([xr, yr], axis=-1)


def _gather_time(arr_bnt: np.ndarray, t_b: np.ndarray) -> np.ndarray:
	idx = np.broadcast_to(t_b[:, None, None], (arr_bnt.shape[0], arr_bnt.shape[1], 1))
	return np.take_along_axis(arr_bnt, idx, axis=2)[:, :, 0]


def _gather_ego(arr_bnt: np.ndarray, ego_idx_b: np.ndarray) -> np.ndarray:
	idx = np.broadcast_to(ego_idx_b[:, None, None], (arr_bnt.shape[0], 1, arr_bnt.shape[2]))
	return np.take_along_axis(arr_bnt, idx, axis=1)[:, 0, :]


def _first_true_index(mask_bk: np.ndarray) -> np.ndarray:
	return np.argmax(mask_bk.astype(np.int32), axis=-1).astype(np.int32)


def _ensure_rng(seed_or_rng: int | np.random.Generator | None) -> np.random.Generator:
	if isinstance(seed_or_rng, np.random.Generator):
		return seed_or_rng
	return np.random.default_rng(0 if seed_or_rng is None else int(seed_or_rng))


def _load_npz(cache_path: str | Path) -> dict[str, np.ndarray]:
	path = Path(cache_path)
	with np.load(path, allow_pickle=True) as data:
		payload = {key: data[key] for key in data.files}

	missing = [key for key in _CACHE_FIELD_NAMES if key not in payload]
	if missing:
		raise KeyError(f"Cache file {path} is missing required fields: {missing}")
	return payload


def _select_scenarios(
	payload: Mapping[str, np.ndarray],
	scenario_indices: Sequence[int] | int | None,
) -> list[int]:
	if scenario_indices is None:
		return list(range(int(np.asarray(payload["object_metadata_is_sdc"]).shape[0])))
	if isinstance(scenario_indices, (int, np.integer)):
		return [int(scenario_indices)]
	return [int(index) for index in scenario_indices]


def _subset_payload(
	payload: Mapping[str, np.ndarray],
	scenario_indices: Sequence[int] | int | None,
) -> tuple[dict[str, np.ndarray], list[int]]:
	selected_indices = _select_scenarios(payload, scenario_indices)
	subset: dict[str, np.ndarray] = {}
	for key in _CACHE_FIELD_NAMES:
		subset[key] = np.asarray(payload[key])[selected_indices]
	return subset, selected_indices


def extract_ego_index(state: Mapping[str, np.ndarray]) -> np.ndarray:
	is_sdc = np.asarray(state["object_metadata_is_sdc"])
	if is_sdc.ndim != 2:
		raise ValueError("Expected batched object_metadata_is_sdc with shape [B, N].")
	return np.argmax(is_sdc.astype(np.int32), axis=-1).astype(np.int32)


def compute_world_dt_seconds(state: Mapping[str, np.ndarray], fallback_dt: float = 0.1) -> np.ndarray:
	ego_idx = extract_ego_index(state)
	ego_ts_us = _gather_ego(np.asarray(state["log_trajectory_timestamp_micros"]), ego_idx).astype(np.float32)
	ego_valid = _gather_ego(np.asarray(state["log_trajectory_valid"]), ego_idx).astype(bool)

	dt = (ego_ts_us[:, 1:] - ego_ts_us[:, :-1]) * 1e-6
	valid_pair = ego_valid[:, 1:] & ego_valid[:, :-1] & (dt > 0.0)
	any_valid = np.any(valid_pair, axis=-1)

	first_idx = _first_true_index(valid_pair)
	first_dt = np.take_along_axis(dt, first_idx[:, None], axis=1)[:, 0]
	return np.where(any_valid, first_dt, np.full_like(first_dt, fallback_dt, dtype=np.float32)).astype(np.float32)


def sample_anchor_step(
	state: Mapping[str, np.ndarray],
	rng: np.random.Generator,
	horizon_world_steps_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	ego_idx = extract_ego_index(state)
	ego_valid = _gather_ego(np.asarray(state["log_trajectory_valid"]), ego_idx).astype(bool)
	num_steps = int(np.asarray(state["log_trajectory_valid"]).shape[-1])
	arange_t = np.arange(num_steps, dtype=np.int32)[None, :]

	max_anchor = np.clip(num_steps - horizon_world_steps_b - 1, a_min=0, a_max=None)
	end_idx = np.clip(arange_t + horizon_world_steps_b[:, None], a_min=0, a_max=num_steps - 1)
	end_valid = np.take_along_axis(ego_valid, end_idx, axis=1)
	valid_mask = (arange_t <= max_anchor[:, None]) & ego_valid & end_valid

	anchors = np.zeros(valid_mask.shape[0], dtype=np.int32)
	has_any = np.any(valid_mask, axis=-1)
	for batch_idx, mask_t in enumerate(valid_mask):
		count = int(mask_t.sum())
		if count <= 0:
			continue
		target_rank = int(rng.integers(0, count))
		rank = np.cumsum(mask_t.astype(np.int32)) - 1
		anchors[batch_idx] = int(np.argmax(((rank == target_rank) & mask_t).astype(np.int32)))
	return anchors.astype(np.int32), has_any.astype(bool)


def sample_future_goal_step(
	ego_valid_bt: np.ndarray,
	anchor_step_b: np.ndarray,
	rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
	num_steps = int(ego_valid_bt.shape[1])
	arange_t = np.arange(num_steps, dtype=np.int32)[None, :]
	valid_mask = ego_valid_bt.astype(bool) & (arange_t > anchor_step_b[:, None])

	goals = np.zeros(valid_mask.shape[0], dtype=np.int32)
	has_any = np.any(valid_mask, axis=-1)
	for batch_idx, mask_t in enumerate(valid_mask):
		count = int(mask_t.sum())
		if count <= 0:
			goals[batch_idx] = int(np.clip(anchor_step_b[batch_idx], 0, num_steps - 1))
			continue
		target_rank = int(rng.integers(0, count))
		rank = np.cumsum(mask_t.astype(np.int32)) - 1
		goals[batch_idx] = int(np.argmax(((rank == target_rank) & mask_t).astype(np.int32)))
	return goals.astype(np.int32), has_any.astype(bool)


def resample_trajectory_timebase_numpy(
	traj_btd: np.ndarray,
	src_t_s: np.ndarray,
	dst_t_s: np.ndarray,
	yaw_index: int = 4,
) -> np.ndarray:
	if traj_btd.ndim != 3:
		raise ValueError(f"traj_btd must be [B, T, D], got {traj_btd.shape}")

	bsz, _, _ = traj_btd.shape
	src_t = np.asarray(src_t_s)
	dst_t = np.asarray(dst_t_s)
	if src_t.ndim == 1:
		src_t = np.broadcast_to(src_t[None, :], (bsz, src_t.shape[0]))
	elif src_t.ndim == 2 and src_t.shape[0] == 1 and bsz > 1:
		src_t = np.broadcast_to(src_t, (bsz, src_t.shape[1]))
	if dst_t.ndim == 1:
		dst_t = np.broadcast_to(dst_t[None, :], (bsz, dst_t.shape[0]))
	elif dst_t.ndim == 2 and dst_t.shape[0] == 1 and bsz > 1:
		dst_t = np.broadcast_to(dst_t, (bsz, dst_t.shape[1]))

	outputs: list[np.ndarray] = []
	for batch_idx in range(bsz):
		traj_td = np.asarray(traj_btd[batch_idx])
		src_t_t = np.asarray(src_t[batch_idx])
		dst_t_k = np.asarray(dst_t[batch_idx])
		n = int(src_t_t.shape[0])
		if n < 2:
			outputs.append(np.broadcast_to(traj_td[:1], (dst_t_k.shape[0], traj_td.shape[-1])).astype(np.float32))
			continue

		hi = np.searchsorted(src_t_t, dst_t_k, side="left")
		hi = np.clip(hi, 1, n - 1)
		lo = hi - 1

		t0 = src_t_t[lo]
		t1 = src_t_t[hi]
		denom = np.maximum(t1 - t0, 1e-6)
		w = (dst_t_k - t0) / denom

		y0 = traj_td[lo, :]
		y1 = traj_td[hi, :]
		out = y0 + w[:, None] * (y1 - y0)

		if 0 <= yaw_index < traj_td.shape[-1]:
			yaw0 = traj_td[lo, yaw_index]
			yaw1 = traj_td[hi, yaw_index]
			sin_interp = np.sin(yaw0) + w * (np.sin(yaw1) - np.sin(yaw0))
			cos_interp = np.cos(yaw0) + w * (np.cos(yaw1) - np.cos(yaw0))
			out[:, yaw_index] = np.arctan2(sin_interp, cos_interp)

		outputs.append(out.astype(np.float32))

	return np.stack(outputs, axis=0)


def _build_map_features(
	state: Mapping[str, np.ndarray],
	origin_xy_b2: np.ndarray,
	heading_b: np.ndarray,
	cfg: PreprocessConfig,
	allowed_types: tuple[int, ...] | None = None,
	max_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	x_bm = np.asarray(state["roadgraph_points_x"])
	y_bm = np.asarray(state["roadgraph_points_y"])
	dir_x_bm = np.asarray(state["roadgraph_points_dir_x"])
	dir_y_bm = np.asarray(state["roadgraph_points_dir_y"])
	types_bm = np.asarray(state["roadgraph_points_types"]).astype(np.int32)
	valid_bm = np.asarray(state["roadgraph_points_valid"]).astype(bool)
	ids_bm = np.asarray(state["roadgraph_points_ids"]).astype(np.int32)

	max_segments = int(cfg.max_segments) if max_segments is None else int(max_segments)
	max_points_per_segment = int(cfg.max_points_per_segment)
	feat_dim = 2 + 2 + int(cfg.num_map_type_classes)
	sentinel_id = np.iinfo(np.int32).max

	feats = np.zeros((x_bm.shape[0], max_segments, max_points_per_segment, feat_dim), dtype=np.float32)
	valids = np.zeros((x_bm.shape[0], max_segments, max_points_per_segment), dtype=bool)
	ids = np.zeros((x_bm.shape[0], max_segments), dtype=np.int32)

	for batch_idx in range(x_bm.shape[0]):
		xy = np.stack([x_bm[batch_idx], y_bm[batch_idx]], axis=-1)
		dirs = np.stack([dir_x_bm[batch_idx], dir_y_bm[batch_idx]], axis=-1)
		types = types_bm[batch_idx]
		point_ids = ids_bm[batch_idx]
		valid = valid_bm[batch_idx]

		centered = xy - origin_xy_b2[batch_idx][None, :]
		dist = np.linalg.norm(centered, axis=-1)
		valid_and_range = valid & (dist < float(cfg.max_range))
		if allowed_types is not None:
			valid_and_range = valid_and_range & np.isin(types, np.asarray(allowed_types, dtype=np.int32))
		if not np.any(valid_and_range):
			continue

		sort_ids = np.where(valid_and_range, point_ids, sentinel_id)
		sort_dist = np.where(valid_and_range, dist, np.inf)
		order = np.lexsort((sort_dist, sort_ids))

		sort_ids = sort_ids[order]
		sort_xy = xy[order]
		sort_dirs = dirs[order]
		sort_types = types[order]

		unique_ids, counts = np.unique(sort_ids, return_counts=True)
		keep = unique_ids != sentinel_id
		unique_ids = unique_ids[keep][:max_segments]
		counts = counts[keep][:max_segments]
		if unique_ids.size == 0:
			continue

		starts = np.concatenate([
			np.array([0], dtype=np.int32),
			np.cumsum(counts[:-1], dtype=np.int32),
		], axis=0)
		point_rank = np.arange(max_points_per_segment, dtype=np.int32)

		for seg_idx, (seg_id, count, start) in enumerate(zip(unique_ids, counts, starts, strict=False)):
			if seg_id == sentinel_id or count <= 0:
				continue
			start = int(start)
			end = min(start + max_points_per_segment, sort_xy.shape[0])

			seg_xy = np.zeros((max_points_per_segment, 2), dtype=np.float32)
			seg_dirs = np.zeros((max_points_per_segment, 2), dtype=np.float32)
			seg_types = np.zeros((max_points_per_segment,), dtype=np.int32)
			span = end - start
			if span > 0:
				seg_xy[:span] = sort_xy[start:end]
				seg_dirs[:span] = sort_dirs[start:end]
				seg_types[:span] = sort_types[start:end]

			xy_rel = _rotate_xy(seg_xy - origin_xy_b2[batch_idx][None, :], heading_b[batch_idx]) / float(cfg.max_range)
			dir_rel = _rotate_xy(seg_dirs, heading_b[batch_idx])
			dir_norm = np.linalg.norm(dir_rel, axis=-1, keepdims=True)
			dir_rel = dir_rel / np.maximum(dir_norm, 1e-6)

			ty_safe = np.where(
				(seg_types >= 0) & (seg_types < int(cfg.num_map_type_classes)),
				seg_types,
				np.full_like(seg_types, int(cfg.map_unknown_type_index)),
			)
			type_oh = np.eye(int(cfg.num_map_type_classes), dtype=np.float32)[ty_safe]
			seg_feat = np.concatenate([xy_rel.astype(np.float32), dir_rel.astype(np.float32), type_oh], axis=-1)
			seg_valid = point_rank < int(count)
			seg_feat = np.where(seg_valid[:, None], seg_feat, 0.0)

			feats[batch_idx, seg_idx] = seg_feat
			valids[batch_idx, seg_idx] = seg_valid
			ids[batch_idx, seg_idx] = seg_id

	return feats, valids, ids


def _build_tl_features(
	state: Mapping[str, np.ndarray],
	anchor_step_b: np.ndarray,
	origin_xy_b2: np.ndarray,
	heading_b: np.ndarray,
	cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
	x_bl = _gather_time(np.asarray(state["log_traffic_light_x"]), anchor_step_b)
	y_bl = _gather_time(np.asarray(state["log_traffic_light_y"]), anchor_step_b)
	st_bl = _gather_time(np.asarray(state["log_traffic_light_state"]), anchor_step_b).astype(np.int32)
	va_bl = _gather_time(np.asarray(state["log_traffic_light_valid"]), anchor_step_b).astype(bool)

	l_total = int(x_bl.shape[1])
	l = min(int(cfg.max_tl_points), l_total)
	x_bl = x_bl[:, :l]
	y_bl = y_bl[:, :l]
	st_bl = st_bl[:, :l]
	va_bl = va_bl[:, :l]

	xy = np.stack([x_bl, y_bl], axis=-1)
	xy_rel = _rotate_xy(xy - origin_xy_b2[:, None, :], heading_b[:, None]) / float(cfg.max_range)
	st_safe = np.clip(st_bl, 0, 8)
	st_oh = np.eye(9, dtype=np.float32)[st_safe]
	# st_safe = np.clip(st_bl, 0, 6)
	# st_oh = np.eye(7, dtype=np.float32)[st_safe]
	feat = np.concatenate([xy_rel.astype(np.float32), st_oh], axis=-1)
	feat = np.where(va_bl[:, :, None], feat, 0.0)

	pad = int(cfg.max_tl_points) - l
	if pad > 0:
		feat = np.pad(feat, ((0, 0), (0, pad), (0, 0)))
		va_bl = np.pad(va_bl, ((0, 0), (0, pad)), constant_values=False)

	return feat.astype(np.float32), va_bl.astype(bool)


def world_to_ego_normalized(
	ego_world_btd: np.ndarray,
	other_world_bnd: np.ndarray,
	other_valid_bn: np.ndarray,
	cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	origin = ego_world_btd[:, 0, :2]
	heading = ego_world_btd[:, 0, 4]

	pos = _rotate_xy(ego_world_btd[:, :, :2] - origin[:, None, :], heading[:, None]) / float(cfg.ego_range)
	vel = _rotate_xy(ego_world_btd[:, :, 2:4], heading[:, None]) / float(cfg.max_velocity)
	yaw = _wrap_to_pi(ego_world_btd[:, :, 4] - heading[:, None]) / np.pi
	ego_norm = np.concatenate([pos, vel, yaw[:, :, None]], axis=-1)

	other_pos = _rotate_xy(other_world_bnd[:, :, :2] - origin[:, None, :], heading[:, None]) / float(cfg.max_range)
	other_vel = _rotate_xy(other_world_bnd[:, :, 2:4], heading[:, None]) / float(cfg.max_velocity)
	other_yaw = _wrap_to_pi(other_world_bnd[:, :, 4] - heading[:, None]) / np.pi
	other_size = other_world_bnd[:, :, 5:7] / float(cfg.max_width)
	other_norm = np.concatenate([other_pos, other_vel, other_yaw[:, :, None], other_size], axis=-1)
	other_norm = np.where(other_valid_bn[:, :, None], other_norm, 0.0)

	return ego_norm[:, 1:, :], ego_norm[:, 0, :], other_norm, other_valid_bn.astype(bool)


def _prepare_batch_state(cache: Mapping[str, np.ndarray], scenario_idx: int) -> dict[str, np.ndarray]:
	state: dict[str, np.ndarray] = {}
	for key in _CACHE_FIELD_NAMES:
		state[key] = np.asarray(cache[key])[scenario_idx][None, ...]
	return state


def _select_override_value(
	override: Any,
	selected_count: int,
	selected_pos: int,
) -> Any:
	if override is None:
		return None
	value = np.asarray(override)
	if value.ndim == 0:
		return value.item()
	if value.shape[0] == selected_count:
		return value[selected_pos]
	return value


def _preprocess_single_cached_scenario(
	state: Mapping[str, np.ndarray],
	rng: np.random.Generator,
	cfg: PreprocessConfig,
	anchor_step_override: Any = None,
	goal_step_override: Any = None,
	goal_xy_override: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	bsz = int(np.asarray(state["log_trajectory_x"]).shape[0])
	ego_idx = extract_ego_index(state)
	world_dt_b = compute_world_dt_seconds(state, cfg.world_dt_fallback)

	total_horizon_s = float(cfg.predict_horizon * cfg.model_dt)
	horizon_world_steps_b = np.ceil(total_horizon_s / np.maximum(world_dt_b, 1e-3)).astype(np.int32)

	if anchor_step_override is not None:
		anchor_step_b = np.full((bsz,), int(np.asarray(anchor_step_override)), dtype=np.int32)
	else:
		anchor_step_b, _ = sample_anchor_step(state, rng, horizon_world_steps_b)

	x_bt = _gather_ego(np.asarray(state["log_trajectory_x"]), ego_idx)
	y_bt = _gather_ego(np.asarray(state["log_trajectory_y"]), ego_idx)
	vx_bt = _gather_ego(np.asarray(state["log_trajectory_vel_x"]), ego_idx)
	vy_bt = _gather_ego(np.asarray(state["log_trajectory_vel_y"]), ego_idx)
	yaw_bt = _gather_ego(np.asarray(state["log_trajectory_yaw"]), ego_idx)
	ego_valid_bt = _gather_ego(np.asarray(state["log_trajectory_valid"]), ego_idx).astype(bool)
	ts_bt = _gather_ego(np.asarray(state["log_trajectory_timestamp_micros"]), ego_idx).astype(np.float32) * 1e-6

	if goal_step_override is not None:
		if isinstance(goal_step_override, (int, np.integer)):
			goal_step_b = np.full((bsz,), int(np.asarray(goal_step_override)), dtype=np.int32)
		elif isinstance(goal_step_override, (list, np.ndarray)):
			goal_step_b = np.asarray(goal_step_override, dtype=np.int32)
			if goal_step_b.ndim == 1 and goal_step_b.shape[0] == bsz:
				goal_step_b = goal_step_b.astype(np.int32)
			else:
				raise ValueError(f"goal_step_override must be a scalar or a 1D array of length {bsz}, got {goal_step_b.shape}")
	else:
		goal_step_b, _ = sample_future_goal_step(ego_valid_bt, anchor_step_b, rng)

	if goal_xy_override is not None:
		goal_xy_world = np.asarray(goal_xy_override)
		if goal_xy_world.ndim == 1:
			goal_xy_world = np.broadcast_to(goal_xy_world[None, :], (bsz, goal_xy_world.shape[0]))
	else:
		goal_x_b = np.take_along_axis(x_bt, goal_step_b[:, None], axis=1)[:, 0]
		goal_y_b = np.take_along_axis(y_bt, goal_step_b[:, None], axis=1)[:, 0]
		goal_xy_world = np.stack([goal_x_b, goal_y_b], axis=-1)

	ego_world_btd_full = np.stack([x_bt, y_bt, vx_bt, vy_bt, yaw_bt], axis=-1)

	anchor_ts_b = np.take_along_axis(ts_bt, anchor_step_b[:, None], axis=1)[:, 0]
	src_t_b = ts_bt - anchor_ts_b[:, None]
	model_t = np.arange(cfg.predict_horizon + 1, dtype=np.float32) * float(cfg.model_dt)
	ego_resampled_world = resample_trajectory_timebase_numpy(ego_world_btd_full, src_t_b, model_t, yaw_index=4)

	x_bn = _gather_time(np.asarray(state["log_trajectory_x"]), anchor_step_b)
	y_bn = _gather_time(np.asarray(state["log_trajectory_y"]), anchor_step_b)
	vx_bn = _gather_time(np.asarray(state["log_trajectory_vel_x"]), anchor_step_b)
	vy_bn = _gather_time(np.asarray(state["log_trajectory_vel_y"]), anchor_step_b)
	yaw_bn = _gather_time(np.asarray(state["log_trajectory_yaw"]), anchor_step_b)
	length_bn = _gather_time(np.asarray(state["log_trajectory_length"]), anchor_step_b)
	width_bn = _gather_time(np.asarray(state["log_trajectory_width"]), anchor_step_b)
	valid_bn = _gather_time(np.asarray(state["log_trajectory_valid"]), anchor_step_b).astype(bool)
	is_sdc_bn = np.asarray(state["object_metadata_is_sdc"]).astype(bool)
	other_valid_bn = valid_bn & (~is_sdc_bn)
	other_world_bnd = np.stack([x_bn, y_bn, vx_bn, vy_bn, yaw_bn, length_bn, width_bn], axis=-1)

	ego_traj_norm, ego_state_norm, other_norm, other_valid = world_to_ego_normalized(
		ego_world_btd=ego_resampled_world,
		other_world_bnd=other_world_bnd,
		other_valid_bn=other_valid_bn,
		cfg=cfg,
	)

	object_types_bn = np.asarray(state["object_metadata_object_types"]).astype(np.int32)
	type_oh = np.eye(int(cfg.num_object_types), dtype=np.float32)[np.clip(object_types_bn, 0, int(cfg.num_object_types) - 1)]
	other_norm = np.concatenate([other_norm, type_oh], axis=-1)

	origin_xy = ego_resampled_world[:, 0, :2]
	anchor_yaw = ego_resampled_world[:, 0, 4]
	goal_xy = _rotate_xy(np.asarray(goal_xy_world) - origin_xy, anchor_yaw) / float(cfg.ego_range)
	# subgoal_xy = np.where(goal_step_b > anchor_step_b + horizon_world_steps_b, ego_traj_norm[:, -1, :2], goal_xy)
	subgoal_xy = ego_traj_norm[:, 15, :2]
	subgoal_valid = np.ones_like(subgoal_xy[..., 0], dtype=bool)
	remaining_timesteps = (goal_step_b - anchor_step_b) / 100.0

	map_features, map_valid, map_ids = _build_map_features(state, origin_xy, anchor_yaw, cfg)
	lane_features, lane_valid, lane_ids = _build_map_features(
		state,
		origin_xy,
		anchor_yaw,
		cfg,
		allowed_types=LANE_MAP_ELEMENT_TYPES,
		max_segments=64
	)
	tl_features, tl_valid = _build_tl_features(state, anchor_step_b, origin_xy, anchor_yaw, cfg)

	max_world_steps = int(math.ceil((cfg.predict_horizon * cfg.model_dt) / cfg.world_dt_fallback)) + 2
	world_step_idx = np.arange(max_world_steps, dtype=np.float32)[None, :]
	world_steps_b = np.floor(total_horizon_s / np.maximum(world_dt_b, 1e-3)).astype(np.int32)
	world_t = world_step_idx * world_dt_b[:, None]
	world_valid = world_step_idx <= world_steps_b[:, None].astype(np.float32)
	world_t = np.where(world_valid, world_t, total_horizon_s)

	features = {
		"ego_state": ego_state_norm.astype(np.float32),
		"ego_trajectory": ego_traj_norm.astype(np.float32),
		"goal_xy": goal_xy.astype(np.float32),
		"subgoal_xy": subgoal_xy.astype(np.float32),
		"subgoal_valid": subgoal_valid,
		"remaining_timesteps": remaining_timesteps[:, None].astype(np.float32),
		"other_states": other_norm.astype(np.float32),
		"other_valid": other_valid.astype(bool),
		"map_features": map_features.astype(np.float32),
		"map_valid": map_valid.astype(bool),
		"map_ids": map_ids.astype(np.int32),
		"lane_features": lane_features.astype(np.float32),
		"lane_valid": lane_valid.astype(bool),
		"lane_ids": lane_ids.astype(np.int32),
		"traffic_light_features": tl_features.astype(np.float32),
		"traffic_light_valid": tl_valid.astype(bool),
	}

	anchor_world_state = ego_resampled_world[:, 0, :]
	aux = {
		"origin_xy": origin_xy.astype(np.float32),
		"anchor_yaw": anchor_yaw.astype(np.float32),
		"anchor_timestamp_micros": (anchor_ts_b * 1e6).astype(np.int32),
		"model_t_seconds": np.broadcast_to(model_t[None, :], (bsz, model_t.shape[0])).astype(np.float32),
		"world_t_seconds": world_t.astype(np.float32),
		"world_t_valid": world_valid.astype(bool),
		"world_dt_seconds": world_dt_b.astype(np.float32),
		"ego_index": ego_idx.astype(np.int32),
		"anchor_step": anchor_step_b.astype(np.int32),
		"anchor_world_state": anchor_world_state.astype(np.float32),
		"goal_step": goal_step_b.astype(np.int32),
	}

	return features, aux

def _preprocess_single_cached_scenario_with_history(
	state: Mapping[str, np.ndarray],
	rng: np.random.Generator,
	cfg: PreprocessConfig,
	anchor_step_override: Any = None,
	goal_step_override: Any = None,
	goal_xy_override: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
	bsz = int(np.asarray(state["log_trajectory_x"]).shape[0])
	ego_idx = extract_ego_index(state)
	world_dt_b = compute_world_dt_seconds(state, cfg.world_dt_fallback)

	total_horizon_s = float(cfg.predict_horizon * cfg.model_dt)
	horizon_world_steps_b = np.ceil(total_horizon_s / np.maximum(world_dt_b, 1e-3)).astype(np.int32)

	if anchor_step_override is not None:
		anchor_step_b = np.full((bsz,), int(np.asarray(anchor_step_override)), dtype=np.int32)
	else:
		anchor_step_b, _ = sample_anchor_step(state, rng, horizon_world_steps_b)

	x_bt = _gather_ego(np.asarray(state["log_trajectory_x"]), ego_idx)
	y_bt = _gather_ego(np.asarray(state["log_trajectory_y"]), ego_idx)
	vx_bt = _gather_ego(np.asarray(state["log_trajectory_vel_x"]), ego_idx)
	vy_bt = _gather_ego(np.asarray(state["log_trajectory_vel_y"]), ego_idx)
	yaw_bt = _gather_ego(np.asarray(state["log_trajectory_yaw"]), ego_idx)
	ego_valid_bt = _gather_ego(np.asarray(state["log_trajectory_valid"]), ego_idx).astype(bool)
	ts_bt = _gather_ego(np.asarray(state["log_trajectory_timestamp_micros"]), ego_idx).astype(np.float32) * 1e-6

	if goal_step_override is not None:
		goal_step_b = np.full((bsz,), int(np.asarray(goal_step_override)), dtype=np.int32)
	else:
		goal_step_b, _ = sample_future_goal_step(ego_valid_bt, anchor_step_b, rng)

	if goal_xy_override is not None:
		goal_xy_world = np.asarray(goal_xy_override)
		if goal_xy_world.ndim == 1:
			goal_xy_world = np.broadcast_to(goal_xy_world[None, :], (bsz, goal_xy_world.shape[0]))
	else:
		goal_x_b = np.take_along_axis(x_bt, goal_step_b[:, None], axis=1)[:, 0]
		goal_y_b = np.take_along_axis(y_bt, goal_step_b[:, None], axis=1)[:, 0]
		goal_xy_world = np.stack([goal_x_b, goal_y_b], axis=-1)

	ego_world_btd_full = np.stack([x_bt, y_bt, vx_bt, vy_bt, yaw_bt], axis=-1)

	anchor_ts_b = np.take_along_axis(ts_bt, anchor_step_b[:, None], axis=1)[:, 0]
	src_t_b = ts_bt - anchor_ts_b[:, None]
	model_t = np.arange(cfg.predict_horizon + 1, dtype=np.float32) * float(cfg.model_dt)
	ego_resampled_world = resample_trajectory_timebase_numpy(ego_world_btd_full, src_t_b, model_t, yaw_index=4)

	x_bn = _gather_time(np.asarray(state["log_trajectory_x"]), anchor_step_b)
	y_bn = _gather_time(np.asarray(state["log_trajectory_y"]), anchor_step_b)
	vx_bn = _gather_time(np.asarray(state["log_trajectory_vel_x"]), anchor_step_b)
	vy_bn = _gather_time(np.asarray(state["log_trajectory_vel_y"]), anchor_step_b)
	yaw_bn = _gather_time(np.asarray(state["log_trajectory_yaw"]), anchor_step_b)
	length_bn = _gather_time(np.asarray(state["log_trajectory_length"]), anchor_step_b)
	width_bn = _gather_time(np.asarray(state["log_trajectory_width"]), anchor_step_b)
	valid_bn = _gather_time(np.asarray(state["log_trajectory_valid"]), anchor_step_b).astype(bool)
	is_sdc_bn = np.asarray(state["object_metadata_is_sdc"]).astype(bool)
	other_valid_bn = valid_bn & (~is_sdc_bn)
	other_world_bnd = np.stack([x_bn, y_bn, vx_bn, vy_bn, yaw_bn, length_bn, width_bn], axis=-1)

	ego_traj_norm, ego_state_norm, other_norm, other_valid = world_to_ego_normalized(
		ego_world_btd=ego_resampled_world,
		other_world_bnd=other_world_bnd,
		other_valid_bn=other_valid_bn,
		cfg=cfg,
	)

	object_types_bn = np.asarray(state["object_metadata_object_types"]).astype(np.int32)
	type_oh = np.eye(int(cfg.num_object_types), dtype=np.float32)[np.clip(object_types_bn, 0, int(cfg.num_object_types) - 1)]
	other_norm = np.concatenate([other_norm, type_oh], axis=-1)

	origin_xy = ego_resampled_world[:, 0, :2]
	anchor_yaw = ego_resampled_world[:, 0, 4]
	goal_xy = _rotate_xy(np.asarray(goal_xy_world) - origin_xy, anchor_yaw) / float(cfg.ego_range)
	# subgoal_xy = np.where(goal_step_b > anchor_step_b + horizon_world_steps_b, ego_traj_norm[:, -1, :2], goal_xy)
	subgoal_xy = ego_traj_norm[:, 15, :2]
	subgoal_valid = np.ones_like(subgoal_xy[..., 0], dtype=bool)
	remaining_timesteps = (goal_step_b - anchor_step_b) / 100.0

	map_features, map_valid, map_ids = _build_map_features(state, origin_xy, anchor_yaw, cfg)
	lane_features, lane_valid, lane_ids = _build_map_features(
		state,
		origin_xy,
		anchor_yaw,
		cfg,
		allowed_types=LANE_MAP_ELEMENT_TYPES,
		max_segments=64
	)
	tl_features, tl_valid = _build_tl_features(state, anchor_step_b, origin_xy, anchor_yaw, cfg)

	max_world_steps = int(math.ceil((cfg.predict_horizon * cfg.model_dt) / cfg.world_dt_fallback)) + 2
	world_step_idx = np.arange(max_world_steps, dtype=np.float32)[None, :]
	world_steps_b = np.floor(total_horizon_s / np.maximum(world_dt_b, 1e-3)).astype(np.int32)
	world_t = world_step_idx * world_dt_b[:, None]
	world_valid = world_step_idx <= world_steps_b[:, None].astype(np.float32)
	world_t = np.where(world_valid, world_t, total_horizon_s)

	features = {
		"ego_state": ego_state_norm.astype(np.float32),
		"ego_trajectory": ego_traj_norm.astype(np.float32),
		"goal_xy": goal_xy.astype(np.float32),
		"subgoal_xy": subgoal_xy.astype(np.float32),
		"subgoal_valid": subgoal_valid,
		"remaining_timesteps": remaining_timesteps[:, None].astype(np.float32),
		"other_states": other_norm.astype(np.float32),
		"other_valid": other_valid.astype(bool),
		"map_features": map_features.astype(np.float32),
		"map_valid": map_valid.astype(bool),
		"map_ids": map_ids.astype(np.int32),
		"lane_features": lane_features.astype(np.float32),
		"lane_valid": lane_valid.astype(bool),
		"lane_ids": lane_ids.astype(np.int32),
		"traffic_light_features": tl_features.astype(np.float32),
		"traffic_light_valid": tl_valid.astype(bool),
	}

	anchor_world_state = ego_resampled_world[:, 0, :]
	aux = {
		"origin_xy": origin_xy.astype(np.float32),
		"anchor_yaw": anchor_yaw.astype(np.float32),
		"anchor_timestamp_micros": (anchor_ts_b * 1e6).astype(np.int32),
		"model_t_seconds": np.broadcast_to(model_t[None, :], (bsz, model_t.shape[0])).astype(np.float32),
		"world_t_seconds": world_t.astype(np.float32),
		"world_t_valid": world_valid.astype(bool),
		"world_dt_seconds": world_dt_b.astype(np.float32),
		"ego_index": ego_idx.astype(np.int32),
		"anchor_step": anchor_step_b.astype(np.int32),
		"anchor_world_state": anchor_world_state.astype(np.float32),
	}

	return features, aux


def preprocess_cached_npz(
	cache_path: str | Path,
	cfg: PreprocessConfig,
	*,
	scenario_indices: Sequence[int] | int | None = None,
	seed: int | np.random.Generator | None = None,
	anchor_step_override: int = None,
	goal_step_override: int = None,
	goal_xy_override: Any = None,
	include_inst_features: bool = False
) -> dict[str, dict[str, np.ndarray]]:
	"""Preprocess a cached TFRecord shard into NumPy feature dictionaries.

	The returned structure mirrors :class:`data.types.PreprocessBatch` but uses
	plain NumPy arrays:

	``{"features": {...}, "aux": {...}, "metadata": {...}}``
	"""

	payload = _load_npz(cache_path)
	selected_payload, selected_indices = _subset_payload(payload, scenario_indices)
	rng = _ensure_rng(seed)

	feature_rows: list[dict[str, np.ndarray]] = []
	aux_rows: list[dict[str, np.ndarray]] = []
	for row_idx in range(len(selected_indices)):
		state = _prepare_batch_state(selected_payload, row_idx)
		features, aux = _preprocess_single_cached_scenario(
			state,
			rng,
			cfg,
			anchor_step_override=_select_override_value(anchor_step_override, len(selected_indices), row_idx),
			goal_step_override=_select_override_value(goal_step_override, len(selected_indices), row_idx),
			goal_xy_override=_select_override_value(goal_xy_override, len(selected_indices), row_idx),
		)
		if include_inst_features:
			assert anchor_step_override is not None, "anchor_step_override must be specified when include_inst_features is True"
			features["inst_features"] = selected_payload["inst_features"][row_idx, anchor_step_override // 10][None, ...]
			features["inst_valid"] = selected_payload["inst_valid"][row_idx, anchor_step_override // 10][None, ...]
		feature_rows.append(features)
		aux_rows.append(aux)

	def _stack_tree(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
		if not rows:
			return {}
		keys = rows[0].keys()
		return {
			key: np.concatenate([row[key] for row in rows], axis=0) for key in keys
		}

	metadata: dict[str, np.ndarray] = {
		"cache_path": np.asarray(str(Path(cache_path)), dtype=np.str_),
		"scenario_indices": np.asarray(selected_indices, dtype=np.int32),
		"num_scenarios": np.asarray(len(selected_indices), dtype=np.int32),
	}
	if "tfrecord_path" in payload:
		metadata["tfrecord_path"] = np.asarray(payload["tfrecord_path"], dtype=np.str_)
	if "tfrecord_name" in payload:
		metadata["tfrecord_name"] = np.asarray(payload["tfrecord_name"], dtype=np.str_)
	if "world_idx" in payload:
		metadata["world_idx"] = np.asarray(payload["world_idx"], dtype=np.int32)

	return {
		"features": _stack_tree(feature_rows),
		"aux": _stack_tree(aux_rows),
		"metadata": metadata,
	}


def preprocess_cached_flat(
	cache_path: str | Path,
	cfg: PreprocessConfig,
	*,
	scenario_indices: Sequence[int] | int | None = None,
	seed: int | np.random.Generator | None = None,
	anchor_step_override: Any = None,
	goal_step_override: Any = None,
	goal_xy_override: Any = None,
) -> dict[str, np.ndarray]:
	"""Return a flat NumPy dictionary with ``features/`` and ``aux/`` prefixes."""
	output = preprocess_cached_npz(
		cache_path,
		cfg,
		scenario_indices=scenario_indices,
		seed=seed,
		anchor_step_override=anchor_step_override,
		goal_step_override=goal_step_override,
		goal_xy_override=goal_xy_override,
	)
	flat: dict[str, np.ndarray] = {}
	flat.update({f"features/{key}": value for key, value in output["features"].items()})
	flat.update({f"aux/{key}": value for key, value in output["aux"].items()})
	flat.update({f"metadata/{key}": value for key, value in output["metadata"].items()})
	return flat
