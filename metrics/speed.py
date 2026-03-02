from __future__ import annotations

import jax
from jax import numpy as jnp

from metrics.common import speed
from metrics.helpers import World


def _speed_penalty_score(
		speeds: jax.Array,
		speed_limit_mps: float,
		dt_s: float,
		scenario_duration_s: float,
		max_overspeed_value_threshold: float,
) -> jax.Array:
	speeds_over_limit = jnp.clip(speeds - jnp.asarray(speed_limit_mps, dtype=jnp.float32), 0.0, None)
	violation_loss = (
			jnp.sum(speeds_over_limit, axis=1) * dt_s
			/ (jnp.asarray(max_overspeed_value_threshold, dtype=jnp.float32) * scenario_duration_s)
	)
	return jnp.clip(1.0 - violation_loss, 0.0, None).astype(jnp.float32)


def compute_speed_limit_metric(
		trajectories: jax.Array,
		world: World,
		speed_limit_mps: float | None,
		max_overspeed_value_threshold: float = 2.23,
) -> jax.Array:
	if speed_limit_mps is None:
		return jnp.ones((trajectories.shape[0],), dtype=jnp.float32)
	speeds = speed(trajectories, world=world, dt_s=world.prediction_dt_s)
	scenario_duration_s = max(world.prediction_dt_s * max(trajectories.shape[1] - 1, 1), world.prediction_dt_s)
	return _speed_penalty_score(
			speeds=speeds,
			speed_limit_mps=float(speed_limit_mps),
			dt_s=world.prediction_dt_s,
			scenario_duration_s=scenario_duration_s,
			max_overspeed_value_threshold=max_overspeed_value_threshold,
	)


def compute_soft_speed_metric(
		trajectories: jax.Array,
		world: World,
		speed_limit_mps: float,
) -> jax.Array:
	speeds = speed(trajectories, world=world, dt_s=world.prediction_dt_s)
	scenario_duration_s = max(world.prediction_dt_s * max(trajectories.shape[1] - 1, 1), world.prediction_dt_s)
	return _speed_penalty_score(
			speeds=speeds,
			speed_limit_mps=float(speed_limit_mps),
			dt_s=world.prediction_dt_s,
			scenario_duration_s=scenario_duration_s,
			max_overspeed_value_threshold=10.0,
	)


def compute_stopping_metric(trajectories: jax.Array, world: World) -> jax.Array:
	speeds = speed(trajectories, world=world, dt_s=world.prediction_dt_s)
	scenario_duration_s = max(world.prediction_dt_s * max(trajectories.shape[1] - 1, 1), world.prediction_dt_s)
	violation_loss = jnp.sum(speeds, axis=1) * world.prediction_dt_s / (10.0 * scenario_duration_s)
	return jnp.clip(1.0 - violation_loss, 0.0, None).astype(jnp.float32)
