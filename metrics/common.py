from __future__ import annotations

from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from metrics.helpers import World


def as_state_array(trajectories: jax.Array) -> jax.Array:
	arr = jnp.asarray(trajectories, dtype=jnp.float32)
	if arr.ndim != 3:
		raise ValueError(f"trajectories must have shape (num_proposals, horizon, dim), got {arr.shape}")
	if arr.shape[-1] < 2:
		raise ValueError(f"trajectories last dimension must be at least 2, got {arr.shape[-1]}")
	return arr


def positions_xy(trajectories: jax.Array) -> jax.Array:
	return as_state_array(trajectories)[..., :2]


def yaw_from_states(trajectories: jax.Array, world: Optional[World] = None) -> jax.Array:
	states = as_state_array(trajectories)
	if states.shape[-1] >= 3:
		return states[..., 2].astype(jnp.float32)
	if world is None:
		raise ValueError("world is required when trajectories do not include yaw")
	return jnp.full(states.shape[:2], world.vehiclegraph.yaw[0], dtype=jnp.float32)


def velocity_xy(
		trajectories: jax.Array,
		world: Optional[World] = None,
		dt_s: float = 0.1,
) -> jax.Array:
	states = as_state_array(trajectories)
	if states.shape[-1] >= 5:
		return states[..., 3:5].astype(jnp.float32)

	xy = states[..., :2]
	diff = (xy[:, 1:, :] - xy[:, :-1, :]) / jnp.asarray(dt_s, dtype=jnp.float32)
	if world is not None and int(world.ego_velocity_xy.shape[0]) == 2:
		first = jnp.broadcast_to(world.ego_velocity_xy[None, None, :], (xy.shape[0], 1, 2))
	else:
		first = diff[:, :1, :]
	return jnp.concatenate([first, diff], axis=1).astype(jnp.float32)


def speed(
		trajectories: jax.Array,
		world: Optional[World] = None,
		dt_s: float = 0.1,
) -> jax.Array:
	return jnp.linalg.norm(velocity_xy(trajectories, world=world, dt_s=dt_s), axis=-1).astype(jnp.float32)


def headings_from_xy(trajectories_xy: jax.Array) -> jax.Array:
	xy = jnp.asarray(trajectories_xy, dtype=jnp.float32)
	diff = xy[:, 1:, :] - xy[:, :-1, :]
	head = jnp.arctan2(diff[..., 1], diff[..., 0])
	return jnp.concatenate([head[:, :1], head], axis=1).astype(jnp.float32)


def proposal_polygons(
		world: World,
		trajectories: jax.Array,
		scale: float = 1.0,
		length_width_reduction_m: float = 0.0,
) -> jax.Array:
	xy = positions_xy(trajectories)
	yaw = yaw_from_states(trajectories, world=world)
	length = jnp.asarray(world.vehiclegraph.length[0], dtype=jnp.float32)
	width = jnp.asarray(world.vehiclegraph.width[0], dtype=jnp.float32)
	length = jnp.maximum(length * scale - 2.0 * length_width_reduction_m, 0.0)
	width = jnp.maximum(width * scale - 2.0 * length_width_reduction_m, 0.0)
	half_l = 0.5 * length
	half_w = 0.5 * width
	local_corners = jnp.asarray(
			[
				[half_l, half_w],
				[half_l, -half_w],
				[-half_l, -half_w],
				[-half_l, half_w],
			],
			dtype=jnp.float32,
	)
	local_corners = jnp.broadcast_to(local_corners, xy.shape[:-1] + (4, 2))
	cos_yaw = jnp.cos(yaw)[..., None]
	sin_yaw = jnp.sin(yaw)[..., None]
	rot_x = local_corners[..., 0] * cos_yaw - local_corners[..., 1] * sin_yaw
	rot_y = local_corners[..., 0] * sin_yaw + local_corners[..., 1] * cos_yaw
	corners = jnp.stack([rot_x, rot_y], axis=-1) + xy[..., None, :]
	return jnp.concatenate([corners, corners[..., :1, :]], axis=-2).astype(jnp.float32)


def proposal_ego_area_masks(
		world: World,
		trajectories: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
	if not world.lane_nodes:
		num_proposals = as_state_array(trajectories).shape[0]
		horizon = as_state_array(trajectories).shape[1]
		return (
				jnp.zeros((num_proposals, horizon), dtype=jnp.bool_),
				jnp.ones((num_proposals, horizon), dtype=jnp.bool_),
		)

	polygons = proposal_polygons(
			world,
			trajectories,
			scale=1.0,
			length_width_reduction_m=world.drivable_area_reduction_m,
	)[..., :4, :]
	node_positions = jnp.stack([node.position for node in world.lane_nodes], axis=0).astype(jnp.float32)
	node_lane_ids = jnp.asarray([node.lane_id for node in world.lane_nodes], dtype=jnp.int32)

	center_xy = jnp.mean(polygons, axis=-2)
	center_diffs = center_xy[..., None, :] - node_positions[None, None, :, :]
	center_dists = jnp.linalg.norm(center_diffs, axis=-1)
	center_near_mask = center_dists <= world.drivable_area_tolerance_m
	lane_diff = node_lane_ids[:, None] != node_lane_ids[None, :]
	pairwise_near = center_near_mask[..., :, None] & center_near_mask[..., None, :]
	multiple_lanes = jnp.any(pairwise_near & lane_diff[None, None, :, :], axis=(-2, -1))

	corner_diffs = polygons[..., None, :] - node_positions[None, None, None, :, :]
	corner_dists = jnp.linalg.norm(corner_diffs, axis=-1)
	corner_min_dists = jnp.min(corner_dists, axis=-1)
	non_drivable_area = jnp.any(corner_min_dists > world.drivable_area_tolerance_m, axis=-1)
	return multiple_lanes.astype(jnp.bool_), non_drivable_area.astype(jnp.bool_)
