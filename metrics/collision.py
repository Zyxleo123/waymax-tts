from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from metrics.helpers import World
from metrics.common import ProposalContext, non_drivable_area_from_open_polygons, open_polygons


NO_COLLISION = 0
ACTIVE_FRONT_COLLISION = 1
STOPPED_TRACK_COLLISION = 2
ACTIVE_LATERAL_COLLISION = 3
ACTIVE_REAR_COLLISION = 4


@dataclass
class CollisionMetricResult:
	no_at_fault_collision_score: jax.Array
	collision_time_index: jax.Array
	collision_type_code: jax.Array
	all_collisions: jax.Array
	collided_target_mask: jax.Array
	ego_polygons_xy: jax.Array
	ego_in_non_drivable_area: jax.Array

def _polygon_center(polygons_xy: jax.Array) -> jax.Array:
	return jnp.mean(polygons_xy, axis=-2)


def _polygon_heading_xy(polygons_xy: jax.Array) -> jax.Array:
	front_mid = 0.5 * (polygons_xy[..., 0, :] + polygons_xy[..., 1, :])
	rear_mid = 0.5 * (polygons_xy[..., 2, :] + polygons_xy[..., 3, :])
	heading = front_mid - rear_mid
	norm = jnp.maximum(jnp.linalg.norm(heading, axis=-1, keepdims=True), 1e-6)
	return heading / norm


def _polygon_axes(polygons_xy: jax.Array) -> jax.Array:
	closed = jnp.concatenate([polygons_xy, polygons_xy[..., :1, :]], axis=-2)
	edges = closed[..., 1:, :] - closed[..., :-1, :]
	axes = jnp.stack([-edges[..., 1], edges[..., 0]], axis=-1)
	norm = jnp.maximum(jnp.linalg.norm(axes, axis=-1, keepdims=True), 1e-6)
	return axes / norm


def _project_polygon(polygons_xy: jax.Array, axis_xy: jax.Array) -> tuple[jax.Array, jax.Array]:
	proj = jnp.sum(polygons_xy * axis_xy[..., None, :], axis=-1)
	return jnp.min(proj, axis=-1), jnp.max(proj, axis=-1)


def _intersects_convex_polygons(poly_a_xy: jax.Array, poly_b_xy: jax.Array) -> jax.Array:
	axes_a = _polygon_axes(poly_a_xy)
	axes_b = _polygon_axes(poly_b_xy)
	axes = jnp.concatenate([axes_a, axes_b], axis=0)

	def _axis_overlap(axis_xy: jax.Array) -> jax.Array:
		a_min, a_max = _project_polygon(poly_a_xy, axis_xy)
		b_min, b_max = _project_polygon(poly_b_xy, axis_xy)
		return (a_max >= b_min) & (b_max >= a_min)

	return jnp.all(jax.vmap(_axis_overlap)(axes))


def _pairwise_intersections(ego_polygons_xy: jax.Array, target_polygons_xy: jax.Array) -> jax.Array:
	def _for_ego(ego_poly_xy: jax.Array) -> jax.Array:
		return jax.vmap(lambda target_poly_xy: _intersects_convex_polygons(ego_poly_xy, target_poly_xy))(target_polygons_xy)

	return jax.vmap(_for_ego)(ego_polygons_xy)


def _as_state_array(trajectories: jax.Array) -> jax.Array:
	arr = jnp.asarray(trajectories, dtype=jnp.float32)
	if arr.ndim != 3:
		raise ValueError(f"trajectories must have shape (num_proposals, horizon, dim), got {arr.shape}")
	if arr.shape[-1] < 2:
		raise ValueError(f"trajectories last dimension must be at least 2, got {arr.shape[-1]}")
	return arr


def _proposal_pose_from_world(
		world: World,
		trajectories: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
	states = _as_state_array(trajectories)
	centers_xy = states[..., :2]
	if states.shape[-1] >= 3:
		yaw = states[..., 2]
	else:
		yaw = jnp.full(centers_xy.shape[:2], world.vehiclegraph.yaw[0], dtype=jnp.float32)
	return centers_xy.astype(jnp.float32), yaw.astype(jnp.float32)


def _proposal_polygons(
		world: World,
		trajectories: jax.Array,
		scale: float = 1.0,
		length_width_reduction_m: float = 0.0,
) -> jax.Array:
	centers_xy, yaw = _proposal_pose_from_world(world, trajectories)
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
	local_corners = jnp.broadcast_to(local_corners, centers_xy.shape[:-1] + (4, 2))

	cos_yaw = jnp.cos(yaw)[..., None]
	sin_yaw = jnp.sin(yaw)[..., None]
	rot_x = local_corners[..., 0] * cos_yaw - local_corners[..., 1] * sin_yaw
	rot_y = local_corners[..., 0] * sin_yaw + local_corners[..., 1] * cos_yaw
	corners = jnp.stack([rot_x, rot_y], axis=-1) + centers_xy[..., None, :]
	return jnp.concatenate([corners, corners[..., :1, :]], axis=-2).astype(jnp.float32)


def _ego_area_masks(
		world: World,
		ego_map_polygons_xy: jax.Array,
) -> jax.Array:
	return non_drivable_area_from_open_polygons(world, open_polygons(ego_map_polygons_xy))


def _bounding_radius(polygons_xy: jax.Array, centers_xy: jax.Array) -> jax.Array:
	return jnp.max(
		jnp.linalg.norm(polygons_xy - centers_xy[..., None, :], axis=-1),
		axis=-1,
	).astype(jnp.float32)


def _pairwise_intersections_with_mask(
		ego_polygons_xy: jax.Array,
		target_polygons_xy: jax.Array,
		active_mask: jax.Array,
) -> jax.Array:
	def _for_ego(ego_poly_xy: jax.Array, active_row: jax.Array) -> jax.Array:
		def _for_target(target_poly_xy: jax.Array, active: jax.Array) -> jax.Array:
			return jax.lax.cond(
				active,
				lambda _: _intersects_convex_polygons(ego_poly_xy, target_poly_xy),
				lambda _: jnp.asarray(False, dtype=jnp.bool_),
				operand=None,
			)

		return jax.vmap(_for_target)(target_polygons_xy, active_row)

	return jax.vmap(_for_ego)(ego_polygons_xy, active_mask)


def compute_no_at_fault_collision(
		world: World,
		trajectories: jax.Array,
		proposal_ctx: ProposalContext | None = None,
		target_is_agent: Optional[jax.Array] = None,
		stopped_speed_threshold: float = 5e-3,
) -> CollisionMetricResult:
	"""Computes a JAX version of the PDM no-at-fault collision metric.

	Args:
		world: Current world helper with predicted other-vehicle futures cached.
		trajectories: Ego proposal states of shape `(num_proposals, horizon, dim)`.
			The first two channels are `x, y`; the third channel is interpreted
			as `yaw` if present, otherwise the current ego yaw is reused.
		target_is_agent: Optional `(num_targets,)` boolean mask.
		stopped_speed_threshold: Threshold below which a target is treated as stopped.
	"""
	if proposal_ctx is None:
		states = _as_state_array(trajectories)
		ego_polygons_xy = _proposal_polygons(
				world,
				states,
				scale=world.collision_scale,
				length_width_reduction_m=0.0,
		)
		ego_map_polygons_xy = _proposal_polygons(
				world,
				states,
				scale=1.0,
				length_width_reduction_m=world.drivable_area_reduction_m,
		)
		ego_in_non_drivable_area = _ego_area_masks(world, ego_map_polygons_xy)
	else:
		states = proposal_ctx.states
		ego_polygons_xy = proposal_ctx.ego_polygons_xy
		ego_in_non_drivable_area = proposal_ctx.ego_non_drivable_area

	del target_is_agent, stopped_speed_threshold
	target_polygons_xy = world.other_vehicle_future_polygons_xy
	target_valid = world.other_vehicle_future_valid

	if states.shape[1] != target_polygons_xy.shape[1]:
		raise ValueError(
				"trajectories horizon must match world prediction horizon + 1, "
				f"got {states.shape[1]} and {target_polygons_xy.shape[1]}",
		)

	ego_polygons = proposal_ctx.ego_open_polygons_xy if proposal_ctx is not None else open_polygons(ego_polygons_xy)
	target_polygons = open_polygons(target_polygons_xy)

	if ego_polygons.shape[1] != target_polygons.shape[1]:
		raise ValueError(
				"ego_polygons_xy and target_polygons_xy must share the same horizon, "
				f"got {ego_polygons.shape[1]} and {target_polygons.shape[1]}",
		)

	num_proposals = ego_polygons.shape[0]
	horizon = ego_polygons.shape[1]
	num_targets = target_polygons.shape[0]

	init_score = jnp.ones((num_proposals,), dtype=jnp.float32)
	init_time = jnp.full((num_proposals,), jnp.inf, dtype=jnp.float32)
	init_type = jnp.full((num_proposals,), NO_COLLISION, dtype=jnp.int32)
	init_all_collisions = jnp.ones((1, num_proposals), dtype=jnp.float32)
	init_seen = jnp.zeros((num_proposals, num_targets), dtype=jnp.bool_)

	ego_centers = proposal_ctx.xy if proposal_ctx is not None else states[..., :2]
	ego_radii = _bounding_radius(ego_polygons, ego_centers)
	target_centers = world.other_vehicle_future_centers_xy.astype(jnp.float32)
	target_radii = _bounding_radius(target_polygons, target_centers)

	def _scan_step(carry, time_idx):
		score, first_time, first_type, all_collisions, seen_targets = carry

		ego_t = ego_polygons[:, time_idx, :, :]
		target_t = target_polygons[:, time_idx, :, :]
		valid_hits = target_valid[:, time_idx]
		ego_center_t = ego_centers[:, time_idx, :]
		target_center_t = target_centers[:, time_idx, :]
		center_dists = jnp.linalg.norm(
			ego_center_t[:, None, :] - target_center_t[None, :, :],
			axis=-1,
		)
		broad_phase_hits = center_dists <= (ego_radii[:, time_idx, None] + target_radii[None, :, time_idx])
		active_pairs = broad_phase_hits & valid_hits[None, :] & (~seen_targets)
		pair_hits = _pairwise_intersections_with_mask(ego_t, target_t, active_pairs)

		any_collision = jnp.any(pair_hits, axis=1)
		score = jnp.where(any_collision, 0.0, score)
		first_time = jnp.where(any_collision, jnp.minimum(first_time, time_idx), first_time)
		first_type = jnp.where(
			(first_type == NO_COLLISION) & any_collision,
			ACTIVE_FRONT_COLLISION,
			first_type,
		)
		all_collisions = all_collisions.at[0].set(
			jnp.where(any_collision, 0.0, all_collisions[0]),
		)

		seen_targets = seen_targets | pair_hits
		return (score, first_time, first_type, all_collisions, seen_targets), None

	(score, first_time, first_type, all_collisions, seen_targets), _ = jax.lax.scan(
			_scan_step,
			(init_score, init_time, init_type, init_all_collisions, init_seen),
			jnp.arange(horizon),
	)

	return CollisionMetricResult(
			no_at_fault_collision_score=score,
			collision_time_index=first_time,
			collision_type_code=first_type,
			all_collisions=all_collisions,
			collided_target_mask=seen_targets,
			ego_polygons_xy=ego_polygons_xy,
			ego_in_non_drivable_area=ego_in_non_drivable_area,
	)
