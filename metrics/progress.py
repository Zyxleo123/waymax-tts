from __future__ import annotations

from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from metrics.helpers import World


def _as_xy(trajectories: jax.Array) -> jax.Array:
	arr = jnp.asarray(trajectories, dtype=jnp.float32)
	if arr.ndim != 3:
		raise ValueError(f"trajectories must have shape (num_proposals, horizon, dim), got {arr.shape}")
	if arr.shape[-1] < 2:
		raise ValueError(f"trajectories last dimension must be at least 2, got {arr.shape[-1]}")
	return arr[..., :2]


def _polyline_progress(polyline_xy: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
	segments = polyline_xy[1:] - polyline_xy[:-1]
	segment_lengths = jnp.linalg.norm(segments, axis=-1)
	cumulative = jnp.concatenate(
			[jnp.zeros((1,), dtype=jnp.float32), jnp.cumsum(segment_lengths, axis=0)],
			axis=0,
	)
	return segments, segment_lengths, cumulative


def _project_point_to_polyline(point_xy: jax.Array, polyline_xy: jax.Array) -> jax.Array:
	segments, segment_lengths, cumulative = _polyline_progress(polyline_xy)
	segment_starts = polyline_xy[:-1]
	to_point = point_xy[None, :] - segment_starts
	segment_len_sq = jnp.maximum(segment_lengths * segment_lengths, 1e-6)
	proj_t = jnp.sum(to_point * segments, axis=-1) / segment_len_sq
	proj_t = jnp.clip(proj_t, 0.0, 1.0)
	projections = segment_starts + proj_t[:, None] * segments
	distances = jnp.linalg.norm(point_xy[None, :] - projections, axis=-1)
	best_idx = jnp.argmin(distances)
	return cumulative[best_idx] + proj_t[best_idx] * segment_lengths[best_idx]


def get_reference_centerline(world: World, lane_idx: Optional[int] = None) -> jax.Array:
	"""Returns the current ego lane centerline used for progress scoring."""
	if lane_idx is None:
		lane_idx = world.ego_lane_node_index
	if lane_idx is None:
		lane_idx = world.get_current_lane(world.ego_absolute_index)
	if lane_idx is None:
		return jnp.empty((0, 2), dtype=jnp.float32)

	_, positions, _ = world.extend_centerline(int(lane_idx))
	return positions.astype(jnp.float32)


def compute_progress_raw(
		world: World,
		trajectories_xy: jax.Array,
		reference_centerline_xy: Optional[jax.Array] = None,
) -> jax.Array:
	"""Computes non-normalized progress in meters along the local centerline.

	Args:
		world: Current world helper with ego/lane caches already updated.
		trajectories_xy: Proposed trajectories of shape `(num_proposals, horizon, >=2)`.
		reference_centerline_xy: Optional polyline override of shape `(num_points, 2)`.

	Returns:
		Array of shape `(num_proposals,)` containing raw progress in meters.
	"""
	xy = _as_xy(trajectories_xy)
	centerline_xy = (
			get_reference_centerline(world)
			if reference_centerline_xy is None
			else jnp.asarray(reference_centerline_xy, dtype=jnp.float32)
	)

	if centerline_xy.ndim != 2 or centerline_xy.shape[-1] != 2:
		raise ValueError(
				f"reference_centerline_xy must have shape (num_points, 2), got {centerline_xy.shape}",
		)

	if int(centerline_xy.shape[0]) < 2:
		return jnp.zeros((xy.shape[0],), dtype=jnp.float32)

	start_points = xy[:, 0, :]
	end_points = xy[:, -1, :]
	start_progress = jax.vmap(lambda p: _project_point_to_polyline(p, centerline_xy))(start_points)
	end_progress = jax.vmap(lambda p: _project_point_to_polyline(p, centerline_xy))(end_points)
	return (end_progress - start_progress).astype(jnp.float32)


def normalize_progress(
		progress_raw: jax.Array,
		multiplicative_mask: Optional[jax.Array] = None,
		progress_distance_threshold: float = 0.1,
) -> jax.Array:
	"""Normalizes raw progress using the same policy as `pdm_scorer.py`."""
	progress_raw = jnp.asarray(progress_raw, dtype=jnp.float32)
	if multiplicative_mask is None:
		mask = jnp.ones_like(progress_raw, dtype=jnp.float32)
	else:
		mask = jnp.asarray(multiplicative_mask, dtype=jnp.float32)
		if mask.shape != progress_raw.shape:
			raise ValueError(
					f"multiplicative_mask must have shape {progress_raw.shape}, got {mask.shape}",
			)

	masked_progress = progress_raw * mask
	max_raw_progress = jnp.max(masked_progress, initial=jnp.asarray(0.0, dtype=jnp.float32))

	def _normalize(_: None) -> jax.Array:
		return masked_progress / max_raw_progress

	def _fallback(_: None) -> jax.Array:
		return jnp.where(mask > 0.0, 1.0, 0.0).astype(jnp.float32)

	return jax.lax.cond(
			max_raw_progress > jnp.asarray(progress_distance_threshold, dtype=jnp.float32),
			_normalize,
			_fallback,
			operand=None,
	)


def compute_progress_metric(
		world: World,
		trajectories_xy: jax.Array,
		multiplicative_mask: Optional[jax.Array] = None,
		progress_distance_threshold: float = 0.1,
		reference_centerline_xy: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
	"""Returns `(raw_progress_m, normalized_progress)` for proposal scoring."""
	raw_progress = compute_progress_raw(
			world=world,
			trajectories_xy=trajectories_xy,
			reference_centerline_xy=reference_centerline_xy,
	)
	normalized_progress = normalize_progress(
			progress_raw=raw_progress,
			multiplicative_mask=multiplicative_mask,
			progress_distance_threshold=progress_distance_threshold,
	)
	return raw_progress, normalized_progress
