from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
from jax import numpy as jnp

from metrics.common import speed, yaw_from_states
from metrics.helpers import World


@dataclass
class ComfortMetricResult:
	score: jax.Array
	is_comfortable: jax.Array
	longitudinal_accel: jax.Array
	lateral_accel: jax.Array
	jerk: jax.Array
	yaw_rate: jax.Array


def _threshold(config: Mapping[str, float], key: str, default: float) -> float:
	return float(config.get(key, default))


def compute_comfort_metric(
		trajectories: jax.Array,
		world: World,
		comfort_config: Mapping[str, float] | None = None,
) -> ComfortMetricResult:
	config = dict(comfort_config or {})
	max_longitudinal_accel = _threshold(config, "max_longitudinal_accel", 4.0)
	max_lateral_accel = _threshold(config, "max_lateral_accel", 5.0)
	max_jerk = _threshold(config, "max_jerk", 8.0)
	max_yaw_rate = _threshold(config, "max_yaw_rate", 1.0)

	dt = jnp.asarray(world.prediction_dt_s, dtype=jnp.float32)
	speeds = speed(trajectories, world=world, dt_s=world.prediction_dt_s)
	yaw = yaw_from_states(trajectories, world=world)

	longitudinal_accel = jnp.concatenate(
			[(speeds[:, 1:2] - speeds[:, :1]) / dt, (speeds[:, 1:] - speeds[:, :-1]) / dt],
			axis=1,
	)
	yaw_unwrapped = jnp.unwrap(yaw, axis=1)
	yaw_rate = jnp.concatenate(
			[(yaw_unwrapped[:, 1:2] - yaw_unwrapped[:, :1]) / dt, (yaw_unwrapped[:, 1:] - yaw_unwrapped[:, :-1]) / dt],
			axis=1,
	)
	lateral_accel = speeds * jnp.abs(yaw_rate)
	jerk = jnp.concatenate(
			[(longitudinal_accel[:, 1:2] - longitudinal_accel[:, :1]) / dt, (longitudinal_accel[:, 1:] - longitudinal_accel[:, :-1]) / dt],
			axis=1,
	)

	is_comfortable = (
			(jnp.abs(longitudinal_accel) <= max_longitudinal_accel)
			& (jnp.abs(lateral_accel) <= max_lateral_accel)
			& (jnp.abs(jerk) <= max_jerk)
			& (jnp.abs(yaw_rate) <= max_yaw_rate)
	)
	score = jnp.all(is_comfortable, axis=1).astype(jnp.float32)
	return ComfortMetricResult(
			score=score,
			is_comfortable=is_comfortable,
			longitudinal_accel=longitudinal_accel,
			lateral_accel=lateral_accel,
			jerk=jerk,
			yaw_rate=yaw_rate,
	)
