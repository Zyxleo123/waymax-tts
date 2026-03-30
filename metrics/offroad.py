from __future__ import annotations

import jax
from jax import numpy as jnp

from metrics.common import ProposalContext, positions_xy, proposal_ego_area_masks, yaw_from_states
from metrics.helpers import World


def compute_drivable_area_metric(
		trajectories: jax.Array,
		world: World,
		proposal_ctx: ProposalContext | None = None,
) -> jax.Array:
	"""Returns a drivable-area compliance score per proposal."""
	non_drivable_area = proposal_ego_area_masks(world, trajectories, proposal_ctx=proposal_ctx)
	off_road_mask = jnp.any(non_drivable_area, axis=-1)
	all_offroad = jnp.all(off_road_mask)
	return jnp.where(all_offroad, 1.0, jnp.where(off_road_mask, 0.0, 1.0)).astype(jnp.float32)


def compute_driving_direction_metric(
		trajectories: jax.Array,
		world: World,
		proposal_ctx: ProposalContext | None = None,
		driving_direction_compliance_threshold: float = 2.0,
		driving_direction_violation_threshold: float = 6.0,
) -> jax.Array:
	"""Approximates driving-direction compliance from local lane heading.

	This is a route-agnostic approximation. It treats timesteps where the planned
	heading differs from the nearest local lane heading by more than 90 degrees as
	oncoming-motion segments, then scores each proposal by the maximum contiguous
	distance traveled in such a segment.
	"""
	if not world.lane_nodes:
		return jnp.ones((positions_xy(trajectories).shape[0],), dtype=jnp.float32)

	if proposal_ctx is None:
		xy = positions_xy(trajectories)
		yaw = yaw_from_states(trajectories, world=world)
		heading_xy = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1).astype(jnp.float32)
	else:
		xy = proposal_ctx.xy
		heading_xy = proposal_ctx.heading_xy

	node_positions = jnp.stack([node.position for node in world.lane_nodes], axis=0).astype(jnp.float32)
	node_dirs = jnp.stack([node.direction for node in world.lane_nodes], axis=0).astype(jnp.float32)
	node_dirs = node_dirs / jnp.maximum(jnp.linalg.norm(node_dirs, axis=-1, keepdims=True), 1e-6)

	diffs = xy[..., None, :] - node_positions[None, None, :, :]
	dists = jnp.linalg.norm(diffs, axis=-1)
	nearest_idx = jnp.argmin(dists, axis=-1)
	lane_heading_xy = node_dirs[nearest_idx]
	heading_cos = jnp.sum(heading_xy * lane_heading_xy, axis=-1)
	oncoming_mask = heading_cos < 0.0

	step_dist = jnp.linalg.norm(xy[:, 1:, :] - xy[:, :-1, :], axis=-1)
	step_dist = jnp.concatenate([jnp.zeros((xy.shape[0], 1), dtype=jnp.float32), step_dist], axis=1)
	oncoming_progress = jnp.where(oncoming_mask, step_dist, 0.0)

	def _max_segment_sum(progress_row: jax.Array, mask_row: jax.Array) -> jax.Array:
		def _scan(carry, x):
			running, best = carry
			progress, mask = x
			running = jnp.where(mask, running + progress, 0.0)
			best = jnp.maximum(best, running)
			return (running, best), None

		(_, best), _ = jax.lax.scan(_scan, (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)), (progress_row, mask_row))
		return best

	segment_progress = jax.vmap(_max_segment_sum)(oncoming_progress, oncoming_mask)
	return jnp.where(
			segment_progress < driving_direction_compliance_threshold,
			1.0,
			jnp.where(
					segment_progress < driving_direction_violation_threshold,
					0.5,
					0.0,
			),
	).astype(jnp.float32)
