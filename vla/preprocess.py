from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import torch
import torch.nn.functional as F

from data.types import PreprocessConfig

if TYPE_CHECKING:
	from waymax import datatypes


@dataclass
class TorchPreprocessBatch:
	features: dict[str, torch.Tensor]
	aux: dict[str, torch.Tensor]
	qa: Any | None = None


def _to_tensor(value: Any) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return value
	return torch.as_tensor(np.array(value))


def _slice_pytree(state: Any, index: int) -> Any:
	return jax.tree_util.tree_map(lambda x: x[index], state)


def _stack_batches(batches: list[TorchPreprocessBatch]) -> TorchPreprocessBatch:
	if len(batches) == 1:
		return batches[0]

	feature_keys = batches[0].features.keys()
	aux_keys = batches[0].aux.keys()
	features = {key: torch.stack([batch.features[key] for batch in batches], dim=0) for key in feature_keys}
	aux = {key: torch.stack([batch.aux[key] for batch in batches], dim=0) for key in aux_keys}
	return TorchPreprocessBatch(features=features, aux=aux)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
	return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _rotate_xy(xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
	c = torch.cos(-heading)[..., None]
	s = torch.sin(-heading)[..., None]
	x = xy[..., 0]
	y = xy[..., 1]
	xr = x * c[..., 0] - y * s[..., 0]
	yr = x * s[..., 0] + y * c[..., 0]
	return torch.stack([xr, yr], dim=-1)


def _gather_time(arr_bnt: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
	idx = t_b[:, None, None].expand(arr_bnt.shape[0], arr_bnt.shape[1], 1)
	return torch.gather(arr_bnt, dim=2, index=idx)[:, :, 0]


def _gather_ego(arr_bnt: torch.Tensor, ego_idx_b: torch.Tensor) -> torch.Tensor:
	idx = ego_idx_b[:, None, None].expand(arr_bnt.shape[0], 1, arr_bnt.shape[2])
	return torch.gather(arr_bnt, dim=1, index=idx)[:, 0, :]


def _first_true_index(mask_bk: torch.Tensor) -> torch.Tensor:
	return torch.argmax(mask_bk.to(torch.int64), dim=-1).to(torch.int32)


def _sample_masked_index(mask_bn: torch.Tensor, rng: torch.Generator, fallback_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	weights = mask_bn.to(torch.float32)
	has_any = mask_bn.any(dim=-1)
	if fallback_index.ndim == 0:
		fallback_index = fallback_index.expand(mask_bn.shape[0])
	fallback_oh = F.one_hot(fallback_index.to(torch.int64), num_classes=mask_bn.shape[1]).to(weights.dtype)
	weights = torch.where(has_any[:, None], weights, fallback_oh)
	chosen = torch.multinomial(weights, num_samples=1, replacement=True, generator=rng).squeeze(-1).to(torch.int32)
	return chosen, has_any


def _coerce_generator(rng: torch.Generator | int | None) -> torch.Generator:
	if isinstance(rng, torch.Generator):
		return rng
	generator = torch.Generator(device="cpu")
	if rng is None:
		generator.manual_seed(int(torch.seed()))
	else:
		generator.manual_seed(int(rng))
	return generator


def extract_ego_index(state: datatypes.SimulatorState) -> torch.Tensor:
	is_sdc = _to_tensor(state.object_metadata.is_sdc)
	if is_sdc.ndim != 2:
		raise ValueError("Expected batched SimulatorState with shape [B, N]")
	return torch.argmax(is_sdc.to(torch.int32), dim=-1).to(torch.int32)


def compute_world_dt_seconds(state: datatypes.SimulatorState, fallback_dt: float = 0.1) -> torch.Tensor:
	ego_idx = extract_ego_index(state)
	ego_ts_us = _gather_ego(_to_tensor(state.log_trajectory.timestamp_micros), ego_idx).to(torch.float32)
	ego_valid = _gather_ego(_to_tensor(state.log_trajectory.valid), ego_idx).to(torch.bool)

	dt = (ego_ts_us[:, 1:] - ego_ts_us[:, :-1]) * 1e-6
	valid_pair = ego_valid[:, 1:] & ego_valid[:, :-1] & (dt > 0.0)
	any_valid = valid_pair.any(dim=-1)

	first_idx = _first_true_index(valid_pair)
	first_dt = dt.gather(1, first_idx[:, None])[:, 0]
	fallback = torch.full_like(first_dt, float(fallback_dt))
	return torch.where(any_valid, first_dt, fallback)


def sample_anchor_step(
	state: datatypes.SimulatorState,
	rng: torch.Generator,
	horizon_world_steps_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	ego_idx = extract_ego_index(state)
	ego_valid = _gather_ego(_to_tensor(state.log_trajectory.valid), ego_idx).to(torch.bool)
	num_steps = int(_to_tensor(state.log_trajectory.x).shape[-1])
	arange_t = torch.arange(num_steps, dtype=torch.int32)[None, :]

	max_anchor = torch.clamp(num_steps - horizon_world_steps_b - 1, min=0)
	end_idx = torch.clamp(arange_t + horizon_world_steps_b[:, None], min=0, max=num_steps - 1)
	end_valid = torch.gather(ego_valid, dim=1, index=end_idx.to(torch.int64))

	valid_mask = (arange_t <= max_anchor[:, None]) & ego_valid & end_valid
	fallback = torch.zeros(valid_mask.shape[0], dtype=torch.int32)
	return _sample_masked_index(valid_mask, rng, fallback)


def sample_future_goal_step(
	ego_valid_bt: torch.Tensor,
	anchor_step_b: torch.Tensor,
	rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
	num_steps = int(ego_valid_bt.shape[1])
	arange_t = torch.arange(num_steps, dtype=torch.int32)[None, :]
	valid_mask = ego_valid_bt.to(torch.bool) & (arange_t > anchor_step_b[:, None])
	return _sample_masked_index(valid_mask, rng, anchor_step_b.to(torch.int32))


def resample_trajectory_timebase_torch(
	traj_btd: torch.Tensor,
	src_t_s: torch.Tensor,
	dst_t_s: torch.Tensor,
	yaw_index: int = 4,
) -> torch.Tensor:
	if traj_btd.ndim != 3:
		raise ValueError(f"traj_btd must be [B,T,D], got {tuple(traj_btd.shape)}")

	bsz, _, _ = traj_btd.shape
	src_t = src_t_s
	dst_t = dst_t_s
	if src_t.ndim == 1:
		src_t = src_t[None, :].expand(bsz, -1)
	elif src_t.ndim == 2 and src_t.shape[0] == 1 and bsz > 1:
		src_t = src_t.expand(bsz, -1)
	if dst_t.ndim == 1:
		dst_t = dst_t[None, :].expand(bsz, -1)
	elif dst_t.ndim == 2 and dst_t.shape[0] == 1 and bsz > 1:
		dst_t = dst_t.expand(bsz, -1)

	n = src_t.shape[1]
	hi = torch.searchsorted(src_t, dst_t, right=False)
	hi = torch.clamp(hi, 1, n - 1)
	lo = hi - 1

	t0 = src_t.gather(1, lo)
	t1 = src_t.gather(1, hi)
	denom = torch.clamp(t1 - t0, min=1e-6)
	w = (dst_t - t0) / denom

	y0 = traj_btd.gather(1, lo[..., None].expand(-1, -1, traj_btd.shape[-1]))
	y1 = traj_btd.gather(1, hi[..., None].expand(-1, -1, traj_btd.shape[-1]))
	out = y0 + w[..., None] * (y1 - y0)

	if 0 <= yaw_index < traj_btd.shape[-1]:
		yaw0 = y0[..., yaw_index]
		yaw1 = y1[..., yaw_index]
		sin_interp = torch.sin(yaw0) + w * (torch.sin(yaw1) - torch.sin(yaw0))
		cos_interp = torch.cos(yaw0) + w * (torch.cos(yaw1) - torch.cos(yaw0))
		out[..., yaw_index] = torch.atan2(sin_interp, cos_interp)

	return out.to(torch.float32)


def _build_map_features(
	state: datatypes.SimulatorState,
	origin_xy_b2: torch.Tensor,
	heading_b: torch.Tensor,
	cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
	x_bm = _to_tensor(state.roadgraph_points.x)
	y_bm = _to_tensor(state.roadgraph_points.y)
	dir_x_bm = _to_tensor(state.roadgraph_points.dir_x)
	dir_y_bm = _to_tensor(state.roadgraph_points.dir_y)
	types_bm = _to_tensor(state.roadgraph_points.types).to(torch.int32)
	valid_bm = _to_tensor(state.roadgraph_points.valid).to(torch.bool)
	ids_bm = _to_tensor(state.roadgraph_points.ids).to(torch.int32)

	max_segments = int(cfg.max_segments)
	max_points_per_segment = int(cfg.max_points_per_segment)
	feat_dim = 2 + 2 + cfg.num_map_type_classes

	feats = []
	valids = []
	batch_size = x_bm.shape[0]
	for batch_idx in range(batch_size):
		xy = torch.stack([x_bm[batch_idx], y_bm[batch_idx]], dim=-1)
		dirs = torch.stack([dir_x_bm[batch_idx], dir_y_bm[batch_idx]], dim=-1)
		types = types_bm[batch_idx]
		ids = ids_bm[batch_idx]
		valid = valid_bm[batch_idx]

		centered = xy - origin_xy_b2[batch_idx][None, :]
		dist = torch.linalg.norm(centered, dim=-1)
		range_keep = dist < cfg.max_range

		valid_and_range = valid & range_keep
		if not bool(valid_and_range.any()):
			batch_feat = torch.zeros((max_segments, max_points_per_segment, feat_dim), dtype=torch.float32)
			batch_valid = torch.zeros((max_segments, max_points_per_segment), dtype=torch.bool)
			feats.append(batch_feat)
			valids.append(batch_valid)
			continue

		valid_idx = torch.where(valid_and_range)[0]
		valid_ids = ids[valid_idx]
		valid_xy = xy[valid_idx]
		valid_dirs = dirs[valid_idx]
		valid_types = types[valid_idx]
		valid_dist = dist[valid_idx]

		unique_ids = torch.unique(valid_ids)
		segment_order = []
		for seg_id in unique_ids:
			seg_mask = valid_ids == seg_id
			segment_order.append((torch.min(valid_dist[seg_mask]), seg_id))
		segment_order.sort(key=lambda item: float(item[0].item()))
		n_segments = min(len(segment_order), max_segments)

		segment_feats = []
		segment_valids = []

		for seg_idx in range(n_segments):
			seg_id = segment_order[seg_idx][1]
			seg_mask = valid_ids == seg_id

			seg_xy = valid_xy[seg_mask]
			seg_dirs = valid_dirs[seg_mask]
			seg_types = valid_types[seg_mask]
			seg_dist = valid_dist[seg_mask]

			sorted_dist_idx = torch.argsort(seg_dist)
			seg_xy = seg_xy[sorted_dist_idx]
			seg_dirs = seg_dirs[sorted_dist_idx]
			seg_types = seg_types[sorted_dist_idx]

			n_points_in_seg = min(len(seg_xy), max_points_per_segment)
			seg_xy_selected = seg_xy[:n_points_in_seg]
			seg_dirs_selected = seg_dirs[:n_points_in_seg]
			seg_types_selected = seg_types[:n_points_in_seg]

			xy_rel = _rotate_xy(seg_xy_selected - origin_xy_b2[batch_idx][None, :], heading_b[batch_idx]) / cfg.max_range
			dir_rel = _rotate_xy(seg_dirs_selected, heading_b[batch_idx])
			dir_norm = torch.linalg.norm(dir_rel, dim=-1, keepdim=True)
			dir_rel = dir_rel / torch.clamp(dir_norm, min=1e-6)

			ty_safe = torch.where(
				(seg_types_selected >= 0) & (seg_types_selected < cfg.num_map_type_classes),
				seg_types_selected,
				torch.full_like(seg_types_selected, int(cfg.map_unknown_type_index)),
			)
			type_oh = F.one_hot(ty_safe.to(torch.int64), num_classes=cfg.num_map_type_classes).to(torch.float32)

			seg_feat = torch.cat([xy_rel.to(torch.float32), dir_rel.to(torch.float32), type_oh], dim=-1)
			pad_size = max_points_per_segment - n_points_in_seg
			if pad_size > 0:
				seg_feat = F.pad(seg_feat, (0, 0, 0, pad_size))

			seg_valid_mask = torch.cat(
				[
					torch.ones(n_points_in_seg, dtype=torch.bool),
					torch.zeros(pad_size, dtype=torch.bool),
				],
				dim=0,
			)

			segment_feats.append(seg_feat)
			segment_valids.append(seg_valid_mask)

		if len(segment_feats) > 0:
			stacked_feats = torch.stack(segment_feats, dim=0)
			stacked_valids = torch.stack(segment_valids, dim=0)
		else:
			stacked_feats = torch.zeros((0, max_points_per_segment, feat_dim), dtype=torch.float32)
			stacked_valids = torch.zeros((0, max_points_per_segment), dtype=torch.bool)

		seg_pad = max_segments - stacked_feats.shape[0]
		if seg_pad > 0:
			batch_feat = torch.zeros((max_segments, max_points_per_segment, feat_dim), dtype=torch.float32)
			batch_valid = torch.zeros((max_segments, max_points_per_segment), dtype=torch.bool)
			batch_feat[:stacked_feats.shape[0]] = stacked_feats
			batch_valid[:stacked_valids.shape[0]] = stacked_valids
		else:
			batch_feat = stacked_feats[:max_segments]
			batch_valid = stacked_valids[:max_segments]

		feats.append(batch_feat)
		valids.append(batch_valid)

	return torch.stack(feats, dim=0), torch.stack(valids, dim=0)


def _build_tl_features(
	state: datatypes.SimulatorState,
	anchor_step_b: torch.Tensor,
	origin_xy_b2: torch.Tensor,
	heading_b: torch.Tensor,
	cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
	x_bl = _gather_time(_to_tensor(state.log_traffic_light.x), anchor_step_b)
	y_bl = _gather_time(_to_tensor(state.log_traffic_light.y), anchor_step_b)
	st_bl = _gather_time(_to_tensor(state.log_traffic_light.state), anchor_step_b).to(torch.int32)
	va_bl = _gather_time(_to_tensor(state.log_traffic_light.valid), anchor_step_b).to(torch.bool)

	l_total = x_bl.shape[1]
	l = min(int(cfg.max_tl_points), l_total)

	x_bl = x_bl[:, :l]
	y_bl = y_bl[:, :l]
	st_bl = st_bl[:, :l]
	va_bl = va_bl[:, :l]

	xy = torch.stack([x_bl, y_bl], dim=-1)
	xy_rel = _rotate_xy(xy - origin_xy_b2[:, None, :], heading_b[:, None]) / cfg.max_range
	st_safe = torch.clamp(st_bl, 0, 6)
	st_oh = F.one_hot(st_safe.to(torch.int64), num_classes=7).to(torch.float32)
	feat = torch.cat([xy_rel.to(torch.float32), st_oh], dim=-1)
	feat = torch.where(va_bl[:, :, None], feat, torch.zeros_like(feat))

	pad = int(cfg.max_tl_points) - l
	feat = F.pad(feat, (0, 0, 0, pad))
	valid = F.pad(va_bl, (0, pad), value=False)
	return feat, valid


def world_to_ego_normalized(
	ego_world_btd: torch.Tensor,
	other_world_bnd: torch.Tensor,
	other_valid_bn: torch.Tensor,
	cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	origin = ego_world_btd[:, 0, :2]
	heading = ego_world_btd[:, 0, 4]

	pos = _rotate_xy(ego_world_btd[:, :, :2] - origin[:, None, :], heading[:, None]) / cfg.ego_range
	vel = _rotate_xy(ego_world_btd[:, :, 2:4], heading[:, None]) / cfg.max_velocity
	yaw = _wrap_to_pi(ego_world_btd[:, :, 4] - heading[:, None]) / math.pi
	ego_norm = torch.cat([pos, vel, yaw[:, :, None]], dim=-1)

	other_pos = _rotate_xy(other_world_bnd[:, :, :2] - origin[:, None, :], heading[:, None]) / cfg.max_range
	other_vel = _rotate_xy(other_world_bnd[:, :, 2:4], heading[:, None]) / cfg.max_velocity
	other_yaw = _wrap_to_pi(other_world_bnd[:, :, 4] - heading[:, None]) / math.pi
	other_size = other_world_bnd[:, :, 5:7] / cfg.max_width
	other_norm = torch.cat([other_pos, other_vel, other_yaw[:, :, None], other_size], dim=-1)
	other_norm = torch.where(other_valid_bn[:, :, None], other_norm, torch.zeros_like(other_norm))

	return ego_norm[:, 1:, :], ego_norm[:, 0, :], other_norm, other_valid_bn


def _preprocess_single_batch(
	state: datatypes.SimulatorState,
	rng: torch.Generator,
	cfg: PreprocessConfig,
	anchor_step_override: int | None = None,
	goal_step_override: int | None = None,
) -> tuple[TorchPreprocessBatch, torch.Generator]:
	bsz = int(_to_tensor(state.log_trajectory.x).shape[0])
	ego_idx = extract_ego_index(state)
	world_dt_b = compute_world_dt_seconds(state, cfg.world_dt_fallback)

	total_horizon_s = cfg.predict_horizon * cfg.model_dt
	horizon_world_steps_b = torch.ceil(total_horizon_s / torch.clamp(world_dt_b, min=1e-3)).to(torch.int32)

	if anchor_step_override is not None:
		anchor_step_b = torch.full((bsz,), int(anchor_step_override), dtype=torch.int32)
	else:
		anchor_step_b, _ = sample_anchor_step(state, rng, horizon_world_steps_b)

	x_bt = _gather_ego(_to_tensor(state.log_trajectory.x), ego_idx)
	y_bt = _gather_ego(_to_tensor(state.log_trajectory.y), ego_idx)
	vx_bt = _gather_ego(_to_tensor(state.log_trajectory.vel_x), ego_idx)
	vy_bt = _gather_ego(_to_tensor(state.log_trajectory.vel_y), ego_idx)
	yaw_bt = _gather_ego(_to_tensor(state.log_trajectory.yaw), ego_idx)
	ego_valid_bt = _gather_ego(_to_tensor(state.log_trajectory.valid), ego_idx).to(torch.bool)
	ts_bt = _gather_ego(_to_tensor(state.log_trajectory.timestamp_micros), ego_idx).to(torch.float32) * 1e-6

	rng_goal = torch.Generator(device="cpu")
	rng_goal.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item()))
	if goal_step_override is not None:
		goal_step_b = torch.full((bsz,), int(goal_step_override), dtype=torch.int32)
	else:
		goal_step_b, _ = sample_future_goal_step(ego_valid_bt, anchor_step_b, rng_goal)
	goal_x_b = x_bt.gather(1, goal_step_b[:, None])[:, 0]
	goal_y_b = y_bt.gather(1, goal_step_b[:, None])[:, 0]
	goal_xy_world = torch.stack([goal_x_b, goal_y_b], dim=-1)

	ego_world_btd_full = torch.stack([x_bt, y_bt, vx_bt, vy_bt, yaw_bt], dim=-1)

	anchor_ts_b = ts_bt.gather(1, anchor_step_b[:, None])[:, 0]
	src_t_b = ts_bt - anchor_ts_b[:, None]
	model_t = torch.arange(cfg.predict_horizon + 1, dtype=torch.float32) * cfg.model_dt
	ego_resampled_world = resample_trajectory_timebase_torch(ego_world_btd_full, src_t_b, model_t, yaw_index=4)

	x_bn = _gather_time(_to_tensor(state.log_trajectory.x), anchor_step_b)
	y_bn = _gather_time(_to_tensor(state.log_trajectory.y), anchor_step_b)
	vx_bn = _gather_time(_to_tensor(state.log_trajectory.vel_x), anchor_step_b)
	vy_bn = _gather_time(_to_tensor(state.log_trajectory.vel_y), anchor_step_b)
	yaw_bn = _gather_time(_to_tensor(state.log_trajectory.yaw), anchor_step_b)
	length_bn = _gather_time(_to_tensor(state.log_trajectory.length), anchor_step_b)
	width_bn = _gather_time(_to_tensor(state.log_trajectory.width), anchor_step_b)
	valid_bn = _gather_time(_to_tensor(state.log_trajectory.valid), anchor_step_b).to(torch.bool)
	is_sdc_bn = _to_tensor(state.object_metadata.is_sdc).to(torch.bool)
	other_valid_bn = valid_bn & (~is_sdc_bn)
	other_world_bnd = torch.stack([x_bn, y_bn, vx_bn, vy_bn, yaw_bn, length_bn, width_bn], dim=-1)

	ego_traj_norm, ego_state_norm, other_norm, other_valid = world_to_ego_normalized(
		ego_world_btd=ego_resampled_world,
		other_world_bnd=other_world_bnd,
		other_valid_bn=other_valid_bn,
		cfg=cfg,
	)

	# Add object type one-hot encoding to other_norm
	object_types_bn = _to_tensor(state.object_metadata.object_types).to(torch.int32)
	type_oh = F.one_hot(
		torch.clamp(object_types_bn, 0, cfg.num_object_types - 1).to(torch.int64),
		num_classes=cfg.num_object_types,
	).to(torch.float32)
	other_norm = torch.cat([other_norm, type_oh], dim=-1)

	origin_xy = ego_resampled_world[:, 0, :2]
	anchor_yaw = ego_resampled_world[:, 0, 4]
	goal_xy = _rotate_xy(goal_xy_world - origin_xy, anchor_yaw) / cfg.ego_range
	remaining_timesteps = (goal_step_b.to(torch.float32) - anchor_step_b.to(torch.float32)) / 100.0

	map_features, map_valid = _build_map_features(state, origin_xy, anchor_yaw, cfg)
	tl_features, tl_valid = _build_tl_features(state, anchor_step_b, origin_xy, anchor_yaw, cfg)

	max_world_steps = int(math.ceil((cfg.predict_horizon * cfg.model_dt) / cfg.world_dt_fallback)) + 2
	world_step_idx = torch.arange(max_world_steps, dtype=torch.float32)[None, :]
	world_steps_b = torch.floor(total_horizon_s / torch.clamp(world_dt_b, min=1e-3)).to(torch.int32)
	world_t = world_step_idx * world_dt_b[:, None]
	world_valid = world_step_idx <= world_steps_b[:, None].to(torch.float32)
	world_t = torch.where(world_valid, world_t, torch.full_like(world_t, total_horizon_s))

	features = {
		"ego_state": ego_state_norm.to(torch.float32),
		"ego_trajectory": ego_traj_norm.to(torch.float32),
		"goal_xy": goal_xy.to(torch.float32),
		"remaining_timesteps": remaining_timesteps[:, None].to(torch.float32),
		"other_states": other_norm.to(torch.float32),
		"other_valid": other_valid.to(torch.bool),
		"map_features": map_features.to(torch.float32),
		"map_valid": map_valid.to(torch.bool),
		"traffic_light_features": tl_features.to(torch.float32),
		"traffic_light_valid": tl_valid.to(torch.bool),
	}

	anchor_world_state = ego_resampled_world[:, 0, :]
	aux = {
		"origin_xy": origin_xy.to(torch.float32),
		"anchor_yaw": anchor_yaw.to(torch.float32),
		"anchor_timestamp_micros": (anchor_ts_b * 1e6).to(torch.int32),
		"model_t_seconds": model_t[None, :].expand(bsz, -1).to(torch.float32),
		"world_t_seconds": world_t.to(torch.float32),
		"world_t_valid": world_valid.to(torch.bool),
		"world_dt_seconds": world_dt_b.to(torch.float32),
		"ego_index": ego_idx.to(torch.int32),
		"anchor_step": anchor_step_b.to(torch.int32),
		"anchor_world_state": anchor_world_state.to(torch.float32),
	}

	return TorchPreprocessBatch(features=features, aux=aux), rng


def preprocess_simulator_state(
	state: datatypes.SimulatorState,
	rng: torch.Generator | int | None,
	cfg: PreprocessConfig,
	anchor_step_override: int | None = None,
	goal_step_override: int | None = None,
) -> tuple[TorchPreprocessBatch, torch.Generator]:
	if _to_tensor(state.log_trajectory.x).ndim < 3:
		raise ValueError("Expected batched SimulatorState with shape [..., N, T].")

	generator = _coerce_generator(rng)

	if _to_tensor(state.log_trajectory.x).ndim > 3:
		n_devices = int(_to_tensor(state.log_trajectory.x).shape[0])
		seeds = torch.randint(0, 2**31 - 1, (n_devices,), generator=generator, dtype=torch.int64)
		batches: list[TorchPreprocessBatch] = []
		for device_idx in range(n_devices):
			child_rng = torch.Generator(device="cpu")
			child_rng.manual_seed(int(seeds[device_idx].item()))
			batch, _ = _preprocess_single_batch(
				_slice_pytree(state, device_idx),
				child_rng,
				cfg,
				anchor_step_override,
				goal_step_override,
			)
			batches.append(batch)
		return _stack_batches(batches), generator

	return _preprocess_single_batch(state, generator, cfg, anchor_step_override, goal_step_override)


__all__ = [
	"TorchPreprocessBatch",
	"PreprocessConfig",
	"compute_world_dt_seconds",
	"extract_ego_index",
	"preprocess_simulator_state",
	"resample_trajectory_timebase_torch",
	"sample_anchor_step",
	"sample_future_goal_step",
	"world_to_ego_normalized",
]
