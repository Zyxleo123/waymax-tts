from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import jax
from jax import numpy as jnp

from metrics.collision import compute_no_at_fault_collision
from metrics.comfort import compute_comfort_metric
from metrics.lane_following import compute_lane_following_metric, compute_lane_goal_metric
from metrics.offroad import compute_drivable_area_metric, compute_driving_direction_metric
from metrics.progress import compute_progress_metric
from metrics.proximity import (
	compute_give_way_metric,
	compute_overtake_metric,
	compute_proximity_metric,
	compute_yield_metric,
)
from metrics.speed import (
	compute_soft_speed_metric,
	compute_speed_limit_metric,
	compute_stopping_metric,
)
from metrics.ttc import compute_ttc_metric


MULTIPLICATIVE_METRICS = (
	"no_collision",
	"drivable_area",
	"driving_direction",
	"speed_limit",
)

WEIGHTED_METRICS = (
	"progress",
	"ttc",
	"comfortable",
	"lane_following",
	"proximity",
)

OPTIONAL_METRICS = (
	"yield",
	"lane",
	"soft_speed",
	"overtake",
	"stop",
	"give_way",
)


DEFAULT_WEIGHTED_WEIGHTS = {
	"progress": 2.0,
	"ttc": 5.0,
	"comfortable": 5.0,
	"lane_following": 1.0,
	"proximity": 1.0,
}


@dataclass
class MetricRecord:
	score: jax.Array
	details: Any = None


@dataclass
class ScoreResult:
	final_scores: jax.Array
	multiplicative_scores: jax.Array
	weighted_scores: jax.Array
	raw_progress: jax.Array
	normalized_progress: jax.Array
	multiplicative_metrics: Dict[str, jax.Array] = field(default_factory=dict)
	weighted_metrics: Dict[str, jax.Array] = field(default_factory=dict)
	optional_metrics: Dict[str, jax.Array] = field(default_factory=dict)
	details: Dict[str, Any] = field(default_factory=dict)


class ProposalScorer:
	"""Aggregates per-metric proposal scores in a PDMScorer-like structure."""

	def __init__(
			self,
			weighted_metric_weights: Mapping[str, float] | None = None,
	):
		self._weighted_metric_weights = dict(DEFAULT_WEIGHTED_WEIGHTS)
		if weighted_metric_weights is not None:
			self._weighted_metric_weights.update(dict(weighted_metric_weights))

		self._enabled_multiplicative = set(MULTIPLICATIVE_METRICS)
		self._enabled_weighted = set(WEIGHTED_METRICS)
		self._enabled_optional = set()
		self._metric_params: Dict[str, Dict[str, Any]] = {
				name: {} for name in MULTIPLICATIVE_METRICS + WEIGHTED_METRICS + OPTIONAL_METRICS
		}

	def add_metric(self, name: str, **params: Any) -> None:
		metric_name = self._normalize_metric_name(name)
		if metric_name in MULTIPLICATIVE_METRICS:
			self._enabled_multiplicative.add(metric_name)
		elif metric_name in WEIGHTED_METRICS:
			self._enabled_weighted.add(metric_name)
		elif metric_name in OPTIONAL_METRICS:
			self._enabled_optional.add(metric_name)
		else:
			raise ValueError(f"Unknown metric: {name}")
		self._metric_params[metric_name].update(params)

	def remove_metric(self, name: str) -> None:
		metric_name = self._normalize_metric_name(name)
		self._enabled_multiplicative.discard(metric_name)
		self._enabled_weighted.discard(metric_name)
		self._enabled_optional.discard(metric_name)

	def set_metric_params(self, name: str, **params: Any) -> None:
		metric_name = self._normalize_metric_name(name)
		if metric_name not in self._metric_params:
			raise ValueError(f"Unknown metric: {name}")
		self._metric_params[metric_name].update(params)

	def set_weight(self, name: str, weight: float) -> None:
		metric_name = self._normalize_metric_name(name)
		if metric_name not in WEIGHTED_METRICS:
			raise ValueError(f"Only weighted metrics have weights, got: {name}")
		self._weighted_metric_weights[metric_name] = float(weight)

	def score_proposals(self, trajectories: jax.Array, world: Any) -> ScoreResult:
		num_proposals = int(jnp.asarray(trajectories).shape[0])
		multi_records = self._compute_multiplicative_metrics(trajectories, world)
		optional_records = self._compute_optional_metrics(trajectories, world)

		multiplicative_metrics = {
				name: record.score for name, record in multi_records.items()
		}
		optional_metrics = {
				name: record.score for name, record in optional_records.items()
		}

		multiplicative_scores = jnp.ones((num_proposals,), dtype=jnp.float32)
		for name in MULTIPLICATIVE_METRICS:
			if name in multiplicative_metrics:
				multiplicative_scores = multiplicative_scores * multiplicative_metrics[name]
		for name in OPTIONAL_METRICS:
			if name in optional_metrics:
				multiplicative_scores = multiplicative_scores * optional_metrics[name]

		weighted_records, raw_progress, normalized_progress = self._compute_weighted_metrics(
				trajectories=trajectories,
				world=world,
				multiplicative_scores=multiplicative_scores,
		)
		weighted_metrics = {
				name: record.score for name, record in weighted_records.items()
		}

		if "stop" in self._enabled_optional and "progress" in weighted_metrics:
			weighted_metrics["progress"] = jnp.zeros_like(weighted_metrics["progress"])

		weight_sum = 0.0
		weighted_scores = jnp.zeros((num_proposals,), dtype=jnp.float32)
		for name in WEIGHTED_METRICS:
			if name not in weighted_metrics:
				continue
			weight = float(self._weighted_metric_weights.get(name, 0.0))
			if weight <= 0.0:
				continue
			weighted_scores = weighted_scores + weighted_metrics[name] * weight
			weight_sum += weight
		if weight_sum > 0.0:
			weighted_scores = weighted_scores / weight_sum
		else:
			weighted_scores = jnp.ones((num_proposals,), dtype=jnp.float32)

		final_scores = multiplicative_scores * weighted_scores
		details: Dict[str, Any] = {}
		for name, record in {**multi_records, **weighted_records, **optional_records}.items():
			if record.details is not None:
				details[name] = record.details

		return ScoreResult(
				final_scores=final_scores.astype(jnp.float32),
				multiplicative_scores=multiplicative_scores.astype(jnp.float32),
				weighted_scores=weighted_scores.astype(jnp.float32),
				raw_progress=raw_progress.astype(jnp.float32),
				normalized_progress=normalized_progress.astype(jnp.float32),
				multiplicative_metrics=multiplicative_metrics,
				weighted_metrics=weighted_metrics,
				optional_metrics=optional_metrics,
				details=details,
		)

	def _compute_multiplicative_metrics(self, trajectories: jax.Array, world: Any) -> Dict[str, MetricRecord]:
		records: Dict[str, MetricRecord] = {}

		if "no_collision" in self._enabled_multiplicative:
			params = self._metric_params["no_collision"]
			result = compute_no_at_fault_collision(trajectories=trajectories, world=world, **params)
			records["no_collision"] = MetricRecord(score=result.no_at_fault_collision_score, details=result)

		if "drivable_area" in self._enabled_multiplicative:
			params = self._metric_params["drivable_area"]
			score = compute_drivable_area_metric(trajectories=trajectories, world=world, **params)
			records["drivable_area"] = MetricRecord(score=score)

		if "driving_direction" in self._enabled_multiplicative:
			params = self._metric_params["driving_direction"]
			score = compute_driving_direction_metric(trajectories=trajectories, world=world, **params)
			records["driving_direction"] = MetricRecord(score=score)

		if "speed_limit" in self._enabled_multiplicative:
			params = self._require_metric_params("speed_limit", required=("speed_limit_mps",))
			score = compute_speed_limit_metric(trajectories=trajectories, world=world, **params)
			records["speed_limit"] = MetricRecord(score=score)

		return records

	def _compute_weighted_metrics(
			self,
			trajectories: jax.Array,
			world: Any,
			multiplicative_scores: jax.Array,
	) -> tuple[Dict[str, MetricRecord], jax.Array, jax.Array]:
		records: Dict[str, MetricRecord] = {}
		raw_progress = jnp.zeros((jnp.asarray(trajectories).shape[0],), dtype=jnp.float32)
		normalized_progress = jnp.ones_like(raw_progress)

		if "progress" in self._enabled_weighted:
			params = self._metric_params["progress"]
			raw_progress, normalized_progress = compute_progress_metric(
					trajectories_xy=trajectories,
					world=world,
					multiplicative_mask=multiplicative_scores,
					**params,
			)
			records["progress"] = MetricRecord(score=normalized_progress, details={"raw_progress": raw_progress})

		if "ttc" in self._enabled_weighted:
			params = self._metric_params["ttc"]
			result = compute_ttc_metric(trajectories=trajectories, world=world, **params)
			records["ttc"] = MetricRecord(score=result.score, details=result)

		if "comfortable" in self._enabled_weighted:
			params = self._metric_params["comfortable"]
			result = compute_comfort_metric(trajectories=trajectories, world=world, **params)
			records["comfortable"] = MetricRecord(score=result.score, details=result)

		if "lane_following" in self._enabled_weighted:
			params = self._metric_params["lane_following"]
			score = compute_lane_following_metric(trajectories=trajectories, world=world, **params)
			records["lane_following"] = MetricRecord(score=score)

		if "proximity" in self._enabled_weighted:
			params = self._require_metric_params("proximity", required=("lead_object_id",), allow_missing=True)
			score = compute_proximity_metric(trajectories=trajectories, world=world, **params)
			records["proximity"] = MetricRecord(score=score)

		return records, raw_progress, normalized_progress

	def _compute_optional_metrics(self, trajectories: jax.Array, world: Any) -> Dict[str, MetricRecord]:
		records: Dict[str, MetricRecord] = {}

		if "yield" in self._enabled_optional:
			params = self._require_metric_params("yield", required=("target_object_id",))
			score = compute_yield_metric(trajectories=trajectories, world=world, **params)
			records["yield"] = MetricRecord(score=score)

		if "lane" in self._enabled_optional:
			params = self._require_metric_params("lane", required=("lane_goal_xy",))
			score = compute_lane_goal_metric(trajectories=trajectories, world=world, **params)
			records["lane"] = MetricRecord(score=score)

		if "soft_speed" in self._enabled_optional:
			params = self._require_metric_params("soft_speed", required=("speed_limit_mps",))
			score = compute_soft_speed_metric(trajectories=trajectories, world=world, **params)
			records["soft_speed"] = MetricRecord(score=score)

		if "overtake" in self._enabled_optional:
			params = self._require_metric_params("overtake", required=("target_object_id",))
			score = compute_overtake_metric(trajectories=trajectories, world=world, **params)
			records["overtake"] = MetricRecord(score=score)

		if "stop" in self._enabled_optional:
			score = compute_stopping_metric(trajectories=trajectories, world=world)
			records["stop"] = MetricRecord(score=score)

		if "give_way" in self._enabled_optional:
			params = self._require_metric_params("give_way", required=("target_object_id",))
			score = compute_give_way_metric(trajectories=trajectories, world=world, **params)
			records["give_way"] = MetricRecord(score=score)

		return records

	def _require_metric_params(
			self,
			name: str,
			required: tuple[str, ...],
			allow_missing: bool = False,
	) -> Dict[str, Any]:
		params = dict(self._metric_params.get(name, {}))
		missing = [key for key in required if key not in params]
		if missing and not allow_missing:
			raise ValueError(f"Metric '{name}' requires parameters: {', '.join(missing)}")
		return params

	def _normalize_metric_name(self, name: str) -> str:
		metric_name = name.strip().lower()
		aliases = {
				"collision": "no_collision",
				"offroad": "drivable_area",
				"drivable_area_compliance": "drivable_area",
				"driving_direction_compliance": "driving_direction",
				"speed": "speed_limit",
				"comfort": "comfortable",
				"lane": "lane",
		}
		return aliases.get(metric_name, metric_name)
