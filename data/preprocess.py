from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from data.types import PreprocessBatch, PreprocessConfig

if TYPE_CHECKING:
    from waymax import datatypes


def _wrap_to_pi(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _rotate_xy(xy: jax.Array, heading: jax.Array) -> jax.Array:
    c = jnp.cos(-heading)[..., None]
    s = jnp.sin(-heading)[..., None]
    x = xy[..., 0]
    y = xy[..., 1]
    xr = x * c[..., 0] - y * s[..., 0]
    yr = x * s[..., 0] + y * c[..., 0]
    return jnp.stack([xr, yr], axis=-1)


def _gather_time(arr_bnt: jax.Array, t_b: jax.Array) -> jax.Array:
    idx = jnp.broadcast_to(t_b[:, None, None], (arr_bnt.shape[0], arr_bnt.shape[1], 1))
    return jnp.take_along_axis(arr_bnt, idx, axis=2)[:, :, 0]


def _gather_ego(arr_bnt: jax.Array, ego_idx_b: jax.Array) -> jax.Array:
    idx = jnp.broadcast_to(ego_idx_b[:, None, None], (arr_bnt.shape[0], 1, arr_bnt.shape[2]))
    return jnp.take_along_axis(arr_bnt, idx, axis=1)[:, 0, :]


def _first_true_index(mask_bk: jax.Array) -> jax.Array:
    idx = jnp.argmax(mask_bk.astype(jnp.int32), axis=-1)
    return idx.astype(jnp.int32)


def extract_ego_index(state: datatypes.SimulatorState) -> jax.Array:
    is_sdc = state.object_metadata.is_sdc
    if is_sdc.ndim != 2:
        raise ValueError("Expected batched SimulatorState with shape [B, N]")
    return jnp.argmax(is_sdc.astype(jnp.int32), axis=-1)


def compute_world_dt_seconds(state: datatypes.SimulatorState, fallback_dt: float = 0.1) -> jax.Array:
    ego_idx = extract_ego_index(state)
    ego_ts_us = _gather_ego(state.log_trajectory.timestamp_micros, ego_idx).astype(jnp.float32)
    ego_valid = _gather_ego(state.log_trajectory.valid, ego_idx)

    dt = (ego_ts_us[:, 1:] - ego_ts_us[:, :-1]) * 1e-6
    valid_pair = ego_valid[:, 1:] & ego_valid[:, :-1] & (dt > 0.0)
    any_valid = jnp.any(valid_pair, axis=-1)

    first_idx = _first_true_index(valid_pair)
    first_dt = jnp.take_along_axis(dt, first_idx[:, None], axis=1)[:, 0]
    return jnp.where(any_valid, first_dt, jnp.full_like(first_dt, fallback_dt))


def sample_anchor_step(
    state: datatypes.SimulatorState,
    rng: jax.Array,
    horizon_world_steps_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    ego_idx = extract_ego_index(state)
    ego_valid = _gather_ego(state.log_trajectory.valid, ego_idx)
    num_steps = state.log_trajectory.shape[-1]
    arange_t = jnp.arange(num_steps, dtype=jnp.int32)[None, :]

    max_anchor = jnp.clip(num_steps - horizon_world_steps_b - 1, min=0)
    end_idx = jnp.clip(arange_t + horizon_world_steps_b[:, None], min=0, max=num_steps - 1)
    end_valid = jnp.take_along_axis(ego_valid, end_idx, axis=1)

    valid_mask = (arange_t <= max_anchor[:, None]) & ego_valid & end_valid

    keys = jax.random.split(rng, valid_mask.shape[0])

    def _sample_one(mask_t: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        count = jnp.sum(mask_t.astype(jnp.int32))
        has_any = count > 0
        target_rank = jax.random.randint(key, (), 0, jnp.maximum(count, 1), dtype=jnp.int32)
        rank = jnp.cumsum(mask_t.astype(jnp.int32)) - 1
        chosen = jnp.argmax(((rank == target_rank) & mask_t).astype(jnp.int32))
        chosen = jnp.where(has_any, chosen, jnp.int32(0))
        return chosen.astype(jnp.int32), has_any

    return jax.vmap(_sample_one)(valid_mask, keys)


def sample_future_goal_step(
    ego_valid_bt: jax.Array,
    anchor_step_b: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    num_steps = ego_valid_bt.shape[1]
    arange_t = jnp.arange(num_steps, dtype=jnp.int32)[None, :]
    valid_mask = ego_valid_bt & (arange_t > anchor_step_b[:, None])
    keys = jax.random.split(rng, valid_mask.shape[0])

    def _sample_one(mask_t: jax.Array, anchor_t: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        count = jnp.sum(mask_t.astype(jnp.int32))
        has_any = count > 0
        target_rank = jax.random.randint(key, (), 0, jnp.maximum(count, 1), dtype=jnp.int32)
        rank = jnp.cumsum(mask_t.astype(jnp.int32)) - 1
        chosen = jnp.argmax(((rank == target_rank) & mask_t).astype(jnp.int32))
        fallback = jnp.clip(anchor_t, 0, num_steps - 1)
        chosen = jnp.where(has_any, chosen, fallback)
        return chosen.astype(jnp.int32), has_any

    return jax.vmap(_sample_one)(valid_mask, anchor_step_b, keys)


def resample_trajectory_timebase_jax(
    traj_btd: jax.Array,
    src_t_s: jax.Array,
    dst_t_s: jax.Array,
    yaw_index: int = 4,
) -> jax.Array:
    if traj_btd.ndim != 3:
        raise ValueError(f"traj_btd must be [B,T,D], got {traj_btd.shape}")

    bsz, _, _ = traj_btd.shape
    src_t = src_t_s
    dst_t = dst_t_s
    if src_t.ndim == 1:
        src_t = jnp.broadcast_to(src_t[None, :], (bsz, src_t.shape[0]))
    elif src_t.ndim == 2 and src_t.shape[0] == 1 and bsz > 1:
        src_t = jnp.broadcast_to(src_t, (bsz, src_t.shape[1]))
    if dst_t.ndim == 1:
        dst_t = jnp.broadcast_to(dst_t[None, :], (bsz, dst_t.shape[0]))
    elif dst_t.ndim == 2 and dst_t.shape[0] == 1 and bsz > 1:
        dst_t = jnp.broadcast_to(dst_t, (bsz, dst_t.shape[1]))

    def _interp_one(traj_td: jax.Array, src_t_t: jax.Array, dst_t_k: jax.Array) -> jax.Array:
        n = src_t_t.shape[0]
        hi = jnp.searchsorted(src_t_t, dst_t_k, side="left")
        hi = jnp.clip(hi, 1, n - 1)
        lo = hi - 1

        t0 = src_t_t[lo]
        t1 = src_t_t[hi]
        denom = jnp.maximum(t1 - t0, 1e-6)
        w = (dst_t_k - t0) / denom

        y0 = traj_td[lo, :]
        y1 = traj_td[hi, :]
        out = y0 + w[:, None] * (y1 - y0)

        if 0 <= yaw_index < traj_td.shape[-1]:
            yaw0 = traj_td[lo, yaw_index]
            yaw1 = traj_td[hi, yaw_index]
            sin_interp = jnp.sin(yaw0) + w * (jnp.sin(yaw1) - jnp.sin(yaw0))
            cos_interp = jnp.cos(yaw0) + w * (jnp.cos(yaw1) - jnp.cos(yaw0))
            yaw = jnp.arctan2(sin_interp, cos_interp)
            out = out.at[:, yaw_index].set(yaw)

        return out.astype(jnp.float32)

    return jax.vmap(_interp_one)(traj_btd, src_t, dst_t)


def _build_map_features(
    state: datatypes.SimulatorState,
    origin_xy_b2: jax.Array,
    heading_b: jax.Array,
    cfg: PreprocessConfig,
) -> tuple[jax.Array, jax.Array]:
    x_bm = state.roadgraph_points.x
    y_bm = state.roadgraph_points.y
    dir_x_bm = state.roadgraph_points.dir_x
    dir_y_bm = state.roadgraph_points.dir_y
    types_bm = state.roadgraph_points.types.astype(jnp.int32)
    valid_bm = state.roadgraph_points.valid
    ids_bm = state.roadgraph_points.ids.astype(jnp.int32)

    max_segments = int(cfg.max_segments)
    max_points_per_segment = int(cfg.max_points_per_segment)
    feat_dim = 2 + 2 + cfg.num_map_type_classes
    sentinel_id = jnp.iinfo(jnp.int32).max

    def _build_for_batch(batch_idx: int) -> tuple[jax.Array, jax.Array]:
        xy = jnp.stack([x_bm[batch_idx], y_bm[batch_idx]], axis=-1)
        dirs = jnp.stack([dir_x_bm[batch_idx], dir_y_bm[batch_idx]], axis=-1)
        types = types_bm[batch_idx]
        ids = ids_bm[batch_idx]
        valid = valid_bm[batch_idx]

        centered = xy - origin_xy_b2[batch_idx][None, :]
        dist = jnp.linalg.norm(centered, axis=-1)
        valid_and_range = valid & (dist < cfg.max_range)

        empty_feat = jnp.zeros((max_segments, max_points_per_segment, feat_dim), dtype=jnp.float32)
        empty_valid = jnp.zeros((max_segments, max_points_per_segment), dtype=jnp.bool_)

        def _build(_: None) -> tuple[jax.Array, jax.Array]:
            sort_ids = jnp.where(valid_and_range, ids, sentinel_id)
            sort_dist = jnp.where(valid_and_range, dist, jnp.inf)
            order = jnp.lexsort((sort_dist, sort_ids))

            sort_ids = sort_ids[order]
            sort_xy = xy[order]
            sort_dirs = dirs[order]
            sort_types = types[order]

            unique_ids, counts = jnp.unique(
                sort_ids,
                size=max_segments + 1,
                fill_value=sentinel_id,
                return_counts=True,
            )
            unique_ids = unique_ids[:max_segments]
            counts = counts[:max_segments]
            starts = jnp.concatenate(
                [jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts[:-1], dtype=jnp.int32)],
                axis=0,
            )

            pad = max_points_per_segment
            sort_xy = jnp.pad(sort_xy, ((0, pad), (0, 0)))
            sort_dirs = jnp.pad(sort_dirs, ((0, pad), (0, 0)))
            sort_types = jnp.pad(sort_types, ((0, pad),))

            point_rank = jnp.arange(max_points_per_segment, dtype=jnp.int32)

            def _build_segment(seg_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
                seg_id = unique_ids[seg_idx]
                count = counts[seg_idx]
                start = starts[seg_idx]

                def _compute(_: None) -> tuple[jax.Array, jax.Array]:
                    seg_xy = jax.lax.dynamic_slice(sort_xy, (start, 0), (max_points_per_segment, 2))
                    seg_dirs = jax.lax.dynamic_slice(sort_dirs, (start, 0), (max_points_per_segment, 2))
                    seg_types = jax.lax.dynamic_slice(sort_types, (start,), (max_points_per_segment,))

                    xy_rel = _rotate_xy(seg_xy - origin_xy_b2[batch_idx][None, :], heading_b[batch_idx]) / cfg.max_range
                    dir_rel = _rotate_xy(seg_dirs, heading_b[batch_idx])
                    dir_norm = jnp.linalg.norm(dir_rel, axis=-1, keepdims=True)
                    dir_rel = dir_rel / jnp.maximum(dir_norm, 1e-6)

                    ty_safe = jnp.where(
                        (seg_types >= 0) & (seg_types < cfg.num_map_type_classes),
                        seg_types,
                        jnp.full_like(seg_types, cfg.map_unknown_type_index),
                    )
                    type_oh = jax.nn.one_hot(ty_safe, cfg.num_map_type_classes, dtype=jnp.float32)
                    seg_feat = jnp.concatenate([xy_rel.astype(jnp.float32), dir_rel.astype(jnp.float32), type_oh], axis=-1)

                    seg_valid = point_rank < count
                    seg_feat = jnp.where(seg_valid[:, None], seg_feat, 0.0)
                    return seg_feat, seg_valid

                return jax.lax.cond(
                    (seg_id != sentinel_id) & (count > 0),
                    _compute,
                    lambda _: (jnp.zeros((max_points_per_segment, feat_dim), dtype=jnp.float32), jnp.zeros((max_points_per_segment,), dtype=jnp.bool_)),
                    operand=None,
                )

            segment_feats, segment_valids = jax.vmap(_build_segment)(jnp.arange(max_segments, dtype=jnp.int32))
            return segment_feats, segment_valids

        return jax.lax.cond(jnp.any(valid_and_range), _build, lambda _: (empty_feat, empty_valid), operand=None)

    feats, valids = jax.vmap(_build_for_batch)(jnp.arange(x_bm.shape[0], dtype=jnp.int32))
    return feats, valids


def _build_tl_features(
    state: datatypes.SimulatorState,
    anchor_step_b: jax.Array,
    origin_xy_b2: jax.Array,
    heading_b: jax.Array,
    cfg: PreprocessConfig,
) -> tuple[jax.Array, jax.Array]:
    x_bl = _gather_time(state.log_traffic_light.x, anchor_step_b)
    y_bl = _gather_time(state.log_traffic_light.y, anchor_step_b)
    st_bl = _gather_time(state.log_traffic_light.state, anchor_step_b).astype(jnp.int32)
    va_bl = _gather_time(state.log_traffic_light.valid, anchor_step_b)

    l_total = x_bl.shape[1]
    l = min(cfg.max_tl_points, l_total)

    x_bl = x_bl[:, :l]
    y_bl = y_bl[:, :l]
    st_bl = st_bl[:, :l]
    va_bl = va_bl[:, :l]

    xy = jnp.stack([x_bl, y_bl], axis=-1)
    xy_rel = _rotate_xy(xy - origin_xy_b2[:, None, :], heading_b[:, None]) / cfg.max_range
    st_safe = jnp.clip(st_bl, 0, 8)
    st_oh = jax.nn.one_hot(st_safe, 9, dtype=jnp.float32)
    feat = jnp.concatenate([xy_rel.astype(jnp.float32), st_oh], axis=-1)
    feat = jnp.where(va_bl[:, :, None], feat, 0.0)

    pad = cfg.max_tl_points - l
    feat = jnp.pad(feat, ((0, 0), (0, pad), (0, 0)))
    valid = jnp.pad(va_bl, ((0, 0), (0, pad)), constant_values=False)
    return feat, valid


def world_to_ego_normalized(
    ego_world_btd: jax.Array,
    other_world_bnd: jax.Array,
    other_valid_bn: jax.Array,
    cfg: PreprocessConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    origin = ego_world_btd[:, 0, :2]
    heading = ego_world_btd[:, 0, 4]

    pos = _rotate_xy(ego_world_btd[:, :, :2] - origin[:, None, :], heading[:, None]) / cfg.ego_range
    vel = _rotate_xy(ego_world_btd[:, :, 2:4], heading[:, None]) / cfg.max_velocity
    yaw = _wrap_to_pi(ego_world_btd[:, :, 4] - heading[:, None]) / jnp.pi
    ego_norm = jnp.concatenate([pos, vel, yaw[:, :, None]], axis=-1)

    other_pos = _rotate_xy(other_world_bnd[:, :, :2] - origin[:, None, :], heading[:, None]) / cfg.max_range
    other_vel = _rotate_xy(other_world_bnd[:, :, 2:4], heading[:, None]) / cfg.max_velocity
    other_yaw = _wrap_to_pi(other_world_bnd[:, :, 4] - heading[:, None]) / jnp.pi
    other_size = other_world_bnd[:, :, 5:7] / cfg.max_width
    other_norm = jnp.concatenate([other_pos, other_vel, other_yaw[:, :, None], other_size], axis=-1)
    other_norm = jnp.where(other_valid_bn[:, :, None], other_norm, 0.0)

    return ego_norm[:, 1:, :], ego_norm[:, 0, :], other_norm, other_valid_bn


def _preprocess_single_batch(
    state: datatypes.SimulatorState,
    rng: jax.Array,
    cfg: PreprocessConfig,
    anchor_step_override: int | None = None,
    goal_step_override: int | None = None,
    goal_xy_override: jax.Array | None = None,
) -> tuple[PreprocessBatch, jax.Array]:
    bsz = state.log_trajectory.x.shape[0]
    ego_idx = extract_ego_index(state)
    world_dt_b = compute_world_dt_seconds(state, cfg.world_dt_fallback)

    total_horizon_s = cfg.predict_horizon * cfg.model_dt
    horizon_world_steps_b = jnp.ceil(total_horizon_s / jnp.maximum(world_dt_b, 1e-3)).astype(jnp.int32)

    if anchor_step_override is not None:
        anchor_step_b = jnp.full((bsz,), anchor_step_override, dtype=jnp.int32)
    else:
        rng, rng_anchor = jax.random.split(rng)
        anchor_step_b, _ = sample_anchor_step(state, rng_anchor, horizon_world_steps_b)

    x_bt = _gather_ego(state.log_trajectory.x, ego_idx)
    y_bt = _gather_ego(state.log_trajectory.y, ego_idx)
    vx_bt = _gather_ego(state.log_trajectory.vel_x, ego_idx)
    vy_bt = _gather_ego(state.log_trajectory.vel_y, ego_idx)
    yaw_bt = _gather_ego(state.log_trajectory.yaw, ego_idx)
    ego_valid_bt = _gather_ego(state.log_trajectory.valid, ego_idx)
    ts_bt = _gather_ego(state.log_trajectory.timestamp_micros, ego_idx).astype(jnp.float32) * 1e-6


    if goal_step_override is not None:
        goal_step_b = jnp.full((bsz,), goal_step_override, dtype=jnp.int32)
    else:
        rng, rng_goal = jax.random.split(rng)
        goal_step_b, _ = sample_future_goal_step(ego_valid_bt, anchor_step_b, rng_goal)
    if goal_xy_override is not None:
        goal_xy_world = goal_xy_override
    else:
        goal_x_b = jnp.take_along_axis(x_bt, goal_step_b[:, None], axis=1)[:, 0]
        goal_y_b = jnp.take_along_axis(y_bt, goal_step_b[:, None], axis=1)[:, 0]
        goal_xy_world = jnp.stack([goal_x_b, goal_y_b], axis=-1)

    ego_world_btd_full = jnp.stack([x_bt, y_bt, vx_bt, vy_bt, yaw_bt], axis=-1)

    anchor_ts_b = jnp.take_along_axis(ts_bt, anchor_step_b[:, None], axis=1)[:, 0]
    src_t_b = ts_bt - anchor_ts_b[:, None]
    model_t = jnp.arange(cfg.predict_horizon + 1, dtype=jnp.float32) * cfg.model_dt
    ego_resampled_world = resample_trajectory_timebase_jax(ego_world_btd_full, src_t_b, model_t, yaw_index=4)

    x_bn = _gather_time(state.log_trajectory.x, anchor_step_b)
    y_bn = _gather_time(state.log_trajectory.y, anchor_step_b)
    vx_bn = _gather_time(state.log_trajectory.vel_x, anchor_step_b)
    vy_bn = _gather_time(state.log_trajectory.vel_y, anchor_step_b)
    yaw_bn = _gather_time(state.log_trajectory.yaw, anchor_step_b)
    length_bn = _gather_time(state.log_trajectory.length, anchor_step_b)
    width_bn = _gather_time(state.log_trajectory.width, anchor_step_b)
    valid_bn = _gather_time(state.log_trajectory.valid, anchor_step_b)
    is_sdc_bn = state.object_metadata.is_sdc
    other_valid_bn = valid_bn & (~is_sdc_bn)
    other_world_bnd = jnp.stack([x_bn, y_bn, vx_bn, vy_bn, yaw_bn, length_bn, width_bn], axis=-1)

    ego_traj_norm, ego_state_norm, other_norm, other_valid = world_to_ego_normalized(
        ego_world_btd=ego_resampled_world,
        other_world_bnd=other_world_bnd,
        other_valid_bn=other_valid_bn,
        cfg=cfg,
    )

    object_types_bn = state.object_metadata.object_types.astype(jnp.int32)
    type_oh = jax.nn.one_hot(
        jnp.clip(object_types_bn, 0, cfg.num_object_types - 1),
        cfg.num_object_types,
        dtype=jnp.float32,
    )
    other_norm = jnp.concatenate([other_norm, type_oh], axis=-1)

    origin_xy = ego_resampled_world[:, 0, :2]
    anchor_yaw = ego_resampled_world[:, 0, 4]
    goal_xy = _rotate_xy(goal_xy_world - origin_xy, anchor_yaw) / cfg.ego_range
    remaining_timesteps = (goal_step_b - anchor_step_b) / 100.0

    map_features, map_valid = _build_map_features(state, origin_xy, anchor_yaw, cfg)
    tl_features, tl_valid = _build_tl_features(state, anchor_step_b, origin_xy, anchor_yaw, cfg)

    max_world_steps = int(math.ceil((cfg.predict_horizon * cfg.model_dt) / cfg.world_dt_fallback)) + 2
    world_step_idx = jnp.arange(max_world_steps, dtype=jnp.float32)[None, :]
    world_steps_b = jnp.floor(total_horizon_s / jnp.maximum(world_dt_b, 1e-3)).astype(jnp.int32)
    world_t = world_step_idx * world_dt_b[:, None]
    world_valid = world_step_idx <= world_steps_b[:, None].astype(jnp.float32)
    world_t = jnp.where(world_valid, world_t, total_horizon_s)

    features = {
        "ego_state": ego_state_norm.astype(jnp.float32),
        "ego_trajectory": ego_traj_norm.astype(jnp.float32),
        "goal_xy": goal_xy.astype(jnp.float32),
        "remaining_timesteps": remaining_timesteps[:, None].astype(jnp.float32),
        "other_states": other_norm.astype(jnp.float32),
        "other_valid": other_valid,
        "map_features": map_features.astype(jnp.float32),
        "map_valid": map_valid,
        "traffic_light_features": tl_features.astype(jnp.float32),
        "traffic_light_valid": tl_valid,
    }

    anchor_world_state = ego_resampled_world[:, 0, :]
    aux = {
        "origin_xy": origin_xy.astype(jnp.float32),
        "anchor_yaw": anchor_yaw.astype(jnp.float32),
        "anchor_timestamp_micros": (anchor_ts_b * 1e6).astype(jnp.int32),
        "model_t_seconds": jnp.broadcast_to(model_t[None, :], (bsz, model_t.shape[0])),
        "world_t_seconds": world_t.astype(jnp.float32),
        "world_t_valid": world_valid,
        "world_dt_seconds": world_dt_b.astype(jnp.float32),
        "ego_index": ego_idx.astype(jnp.int32),
        "anchor_step": anchor_step_b.astype(jnp.int32),
        "anchor_world_state": anchor_world_state.astype(jnp.float32),
    }

    return PreprocessBatch(features=features, aux=aux), rng


def preprocess_simulator_state(
    state: datatypes.SimulatorState,
    rng: jax.Array,
    cfg: PreprocessConfig,
    anchor_step_override: int | None = None,
    goal_step_override: int | None = None,
    goal_xy_override: jax.Array | None = None,
) -> tuple[PreprocessBatch, jax.Array]:
    if state.log_trajectory.x.ndim < 3:
        raise ValueError("Expected batched SimulatorState with shape [..., N, T].")

    if state.log_trajectory.x.ndim > 3:
        n_devices = state.log_trajectory.x.shape[0]
        keys = jax.random.split(rng, n_devices)
        return jax.vmap(
            lambda s, k, t, t_g, g: preprocess_simulator_state(s, k, cfg, t, t_g, g)
        )(state, keys, anchor_step_override, goal_step_override, goal_xy_override)

    return _preprocess_single_batch(state, rng, cfg, anchor_step_override, goal_step_override, goal_xy_override)