from __future__ import annotations

from typing import Optional

import jax
from jax import numpy as jnp

from metrics.common import headings_from_xy, positions_xy
from metrics.helpers import World
from metrics.progress import get_reference_centerline


def _estimate_heading(polyline_xy: jax.Array) -> jax.Array:
	diff = polyline_xy[1:] - polyline_xy[:-1]
	head = jnp.arctan2(diff[:, 1], diff[:, 0])
	return jnp.concatenate([head[:1], head], axis=0).astype(jnp.float32)


def compute_lane_following_metric(
		trajectories: jax.Array,
		world: World,
		max_lane_deviation: float = 10.0,
		reference_centerline_xy: Optional[jax.Array] = None,
		pull_over: bool = False,
) -> jax.Array:
	centerline_xy = (
			get_reference_centerline(world)
			if reference_centerline_xy is None
			else jnp.asarray(reference_centerline_xy, dtype=jnp.float32)
	)
	xy = positions_xy(trajectories)
	if int(centerline_xy.shape[0]) < 2:
		return jnp.ones((xy.shape[0],), dtype=jnp.float32)

	trajectory_heading = headings_from_xy(xy)
	centerline_heading = _estimate_heading(centerline_xy)
	xy_loss = jnp.linalg.norm(xy[:, None, :, :] - centerline_xy[None, :, None, :], axis=-1)
	closest_idx = jnp.argmin(xy_loss, axis=1)
	min_xy_loss = jnp.min(xy_loss, axis=1)
	heading_loss_all = -jnp.cos(trajectory_heading[:, None, :] - centerline_heading[None, :, None])
	heading_loss = jnp.take_along_axis(heading_loss_all, closest_idx[:, None, :], axis=1).squeeze(1)

	position_score = jnp.clip(max_lane_deviation - min_xy_loss, 0.0, max_lane_deviation) / max_lane_deviation
	if pull_over:
		position_score = jnp.ones_like(position_score)
	heading_score = 1.0 - ((heading_loss + 1.0) / 2.0)
	score = 0.5 * (position_score + heading_score)
	return jnp.mean(score, axis=1).astype(jnp.float32)


def compute_lane_goal_metric(
		trajectories: jax.Array,
		world: World,
		lane_goal_xy: jax.Array,
		use_dense: bool = False,
) -> jax.Array:
	trajectory_xy = positions_xy(trajectories)
	lane_goal_xy = jnp.asarray(lane_goal_xy, dtype=jnp.float32)[..., :2]
	dists = jnp.linalg.norm(trajectory_xy[:, :, None, :] - lane_goal_xy[None, None, :, :], axis=-1)
	dists = jnp.min(dists, axis=2)
	dists = jnp.mean(dists, axis=1)
	max_lane_dist = 5.0 if use_dense else 10.0
	return ((max_lane_dist - jnp.clip(dists, 0.0, max_lane_dist)) / max_lane_dist).astype(jnp.float32)
