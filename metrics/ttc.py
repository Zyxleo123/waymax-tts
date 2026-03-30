from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import numpy as jnp

from metrics.collision import _pairwise_intersections
from metrics.common import positions_xy, proposal_polygons, speed, yaw_from_states
from metrics.helpers import World


@dataclass
class TTCMetricResult:
	score: jax.Array
	ttc_time_index: jax.Array
	projected_ego_polygons_xy: jax.Array


def compute_ttc_metric(
		trajectories: jax.Array,
		world: World,
		ttc_fixed_speed: bool = False,
		stopped_speed_threshold: float = 5e-3,
) -> TTCMetricResult:
	del stopped_speed_threshold
	states = jnp.asarray(trajectories, dtype=jnp.float32)
	ego_polygons_xy = proposal_polygons(world, states, scale=world.collision_scale)
	ego_xy = positions_xy(states)
	yaw = yaw_from_states(states, world=world)
	ego_speed = speed(states, world=world, dt_s=world.prediction_dt_s)
	if ttc_fixed_speed:
		project_speed = jnp.full_like(ego_speed, 1.5)
	else:
		project_speed = jnp.clip(ego_speed, 1.5, None)

	dxy_per_s = jnp.stack([jnp.cos(yaw) * project_speed, jnp.sin(yaw) * project_speed], axis=-1)
	future_time_idcs = jnp.asarray([0, 3, 6, 9], dtype=jnp.int32)
	deltas = future_time_idcs.astype(jnp.float32) * world.prediction_dt_s
	projected_centers = ego_xy[:, :, None, :] + dxy_per_s[:, :, None, :] * deltas[None, None, :, None]

	base_polygons = ego_polygons_xy[..., :4, :]
	base_centers = jnp.mean(base_polygons, axis=-2)
	local_corners = base_polygons - base_centers[..., None, :]
	projected_corners = local_corners[:, :, None, :, :] + projected_centers[:, :, :, None, :]
	projected_polygons = jnp.concatenate([projected_corners, projected_corners[..., :1, :]], axis=-2)

	target_polygons = world.other_vehicle_future_polygons_xy[..., :4, :]
	target_valid = world.other_vehicle_future_valid

	num_proposals, horizon = states.shape[:2]
	num_targets = target_polygons.shape[0]
	init_score = jnp.ones((num_proposals,), dtype=jnp.float32)
	init_time = jnp.full((num_proposals,), jnp.inf, dtype=jnp.float32)
	init_seen = jnp.zeros((num_proposals, num_targets), dtype=jnp.bool_)

	def _scan_step(carry, time_idx):
		score, first_time, seen = carry

		def _for_offset(offset_idx, inner):
			score_inner, first_time_inner, seen_inner = inner
			offset = future_time_idcs[offset_idx]
			current_idx = time_idx + offset
			valid_step = current_idx < horizon

			def _valid_case(_: None):
				ego_t = projected_corners[:, time_idx, offset_idx, :, :]
				target_t = target_polygons[:, current_idx, :, :]
				hits = _pairwise_intersections(ego_t, target_t)
				hits = hits & target_valid[:, current_idx][None, :] & (~seen_inner)
				any_hits = jnp.any(hits, axis=1)
				score_next = jnp.where(any_hits, 0.0, score_inner)
				time_next = jnp.where(any_hits, jnp.minimum(first_time_inner, time_idx), first_time_inner)
				seen_next = seen_inner | hits
				return score_next, time_next, seen_next

			return jax.lax.cond(valid_step, _valid_case, lambda _: (score_inner, first_time_inner, seen_inner), operand=None)

		(score, first_time, seen) = jax.lax.fori_loop(
				0,
				future_time_idcs.shape[0],
				_for_offset,
				(score, first_time, seen),
		)
		return (score, first_time, seen), None

	(score, first_time, _), _ = jax.lax.scan(
			_scan_step,
			(init_score, init_time, init_seen),
			jnp.arange(horizon),
	)
	return TTCMetricResult(score=score, ttc_time_index=first_time, projected_ego_polygons_xy=projected_polygons)
