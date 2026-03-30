from __future__ import annotations

from typing import Optional

import jax
from jax import numpy as jnp

from metrics.collision import _intersects_convex_polygons
from metrics.common import ProposalContext, positions_xy, speed
from metrics.helpers import World


def compute_proximity_metric(
		trajectories: jax.Array,
		world: World,
		proposal_ctx: ProposalContext | None = None,
		min_distance_to_lead: float = 5.0,
		lead_object_id: Optional[int] = None,
) -> jax.Array:
	if lead_object_id is None:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)

	matches = jnp.where(world.other_vehicle_object_ids == int(lead_object_id))[0]
	if int(matches.shape[0]) == 0:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)

	target_idx = int(matches[0])
	lead_centers = world.other_vehicle_future_centers_xy[target_idx]
	ego_centers = positions_xy(trajectories) if proposal_ctx is None else proposal_ctx.xy
	dists = jnp.linalg.norm(ego_centers - lead_centers[None, :, :], axis=-1)
	min_dists = jnp.min(dists, axis=1)
	return jnp.clip(min_dists / min_distance_to_lead, 0.0, 1.0).astype(jnp.float32)


def compute_yield_metric(
		trajectories: jax.Array,
		world: World,
		target_object_id: int,
		proposal_ctx: ProposalContext | None = None,
) -> jax.Array:
	matches = jnp.where(world.other_vehicle_object_ids == int(target_object_id))[0]
	if int(matches.shape[0]) == 0:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)

	target_idx = int(matches[0])
	if proposal_ctx is None:
		from metrics.common import open_polygons, proposal_polygons
		ego_polygons = open_polygons(
			proposal_polygons(world, trajectories, scale=world.collision_scale)
		)
	else:
		ego_polygons = proposal_ctx.ego_open_polygons_xy
	target_polygons = world.other_vehicle_future_polygons_xy[target_idx, :, :4, :]

	def _per_proposal(ego_proposal_xy: jax.Array) -> jax.Array:
		hits = jax.vmap(_intersects_convex_polygons)(ego_proposal_xy, target_polygons)
		return jnp.where(jnp.any(hits), 0.0, 1.0)

	return jax.vmap(_per_proposal)(ego_polygons).astype(jnp.float32)


def compute_overtake_metric(
		trajectories: jax.Array,
		world: World,
		target_object_id: int,
		proposal_ctx: ProposalContext | None = None,
) -> jax.Array:
	matches = jnp.where(world.other_vehicle_object_ids == int(target_object_id))[0]
	if int(matches.shape[0]) == 0:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)

	target_idx = int(matches[0])
	target_speed = world.other_vehicle_future_speed[target_idx, 0]
	ego_speed = speed(trajectories, world=world, dt_s=world.prediction_dt_s) if proposal_ctx is None else proposal_ctx.speed
	speeds_under_target = jnp.clip(target_speed - ego_speed, 0.0, None)
	scenario_duration_s = max(world.prediction_dt_s * max(trajectories.shape[1] - 1, 1), world.prediction_dt_s)
	speed_loss = jnp.sum(speeds_under_target, axis=1) * world.prediction_dt_s / (10.0 * scenario_duration_s)
	speed_score = jnp.clip(1.0 - speed_loss, 1e-3, None)

	target_pos = world.other_vehicle_future_centers_xy[target_idx, 0]
	target_heading = world.other_vehicle_future_yaw[target_idx, 0]
	target_heading_vector = jnp.asarray([jnp.cos(target_heading), jnp.sin(target_heading)], dtype=jnp.float32)
	trajectory_xy = positions_xy(trajectories) if proposal_ctx is None else proposal_ctx.xy
	diff_vector = trajectory_xy - target_pos[None, None, :]
	distance_behind = jnp.sum(diff_vector * target_heading_vector[None, None, :], axis=-1)
	distance_behind = jnp.clip(distance_behind, a_min=None, a_max=0.0)
	distance_loss = -0.001 * jnp.sum(distance_behind, axis=1)
	distance_score = jnp.clip(1.0 - distance_loss, 1e-3, None)
	return (speed_score * distance_score).astype(jnp.float32)


def compute_give_way_metric(
		trajectories: jax.Array,
		world: World,
		target_object_id: int,
		proposal_ctx: ProposalContext | None = None,
) -> jax.Array:
	matches = jnp.where(world.other_vehicle_object_ids == int(target_object_id))[0]
	if int(matches.shape[0]) == 0:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)

	target_idx = int(matches[0])
	target_speed = world.other_vehicle_future_speed[target_idx, 0]
	ego_speed = speed(trajectories, world=world, dt_s=world.prediction_dt_s) if proposal_ctx is None else proposal_ctx.speed
	speeds_over_target = jnp.clip(ego_speed - target_speed, 0.0, None)
	scenario_duration_s = max(world.prediction_dt_s * max(trajectories.shape[1] - 1, 1), world.prediction_dt_s)
	speed_loss = jnp.sum(speeds_over_target, axis=1) * world.prediction_dt_s / (10.0 * scenario_duration_s)
	speed_score = jnp.clip(1.0 - speed_loss, 1e-3, None)

	target_pos = world.other_vehicle_future_centers_xy[target_idx, 0]
	target_heading = world.other_vehicle_future_yaw[target_idx, 0]
	target_heading_vector = jnp.asarray([jnp.cos(target_heading), jnp.sin(target_heading)], dtype=jnp.float32)
	trajectory_xy = positions_xy(trajectories) if proposal_ctx is None else proposal_ctx.xy
	diff_vector = trajectory_xy - target_pos[None, None, :]
	distance_front = jnp.sum(diff_vector * target_heading_vector[None, None, :], axis=-1)
	distance_front = jnp.clip(distance_front, 0.0, None)
	distance_loss = 0.001 * jnp.sum(distance_front, axis=1)
	distance_score = jnp.clip(1.0 - distance_loss, 1e-3, None)
	return (speed_score * distance_score).astype(jnp.float32)
