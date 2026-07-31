from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig
from data.utils import VLA_FUNCTION_PROMPT
from model.diffusion.diffusion_policy import DiffusionPolicy
from model.vla.temporal_vla import TemporalGemmaVLA
from planner.abstract_planner import AbstractPlanner, PlannerResult
from scores.scorer_lane_graph import Scorer
from scores.helpers import get_current_speed

@dataclass
class TemporalVLAPlannerResult(PlannerResult):
	reward_texts: Sequence[str]
	target_lane_ids: Sequence[Any] | None = None
	subgoal_preds: Sequence[Any] | None = None


class TemporalVLAPlanner(AbstractPlanner):
	def __init__(
		self,
		policy: DiffusionPolicy,
		temporal_vla: TemporalGemmaVLA | str | Path,
		preprocess_cfg: PreprocessConfig,
		population_size: int = 64,
		num_worlds: int = 1,
		*,
		torch_device: str | torch.device | None = None,
		jax_device: Any | None = None,
		reward_prompt_text: str = VLA_FUNCTION_PROMPT,
		history_steps: int = 5,
		history_stride: int = 10,
		resample_timesteps: int = 3,
		elite_size: int = 4,
		num_iterations: int = 3,
		metrics: list[str] | None = None,
		weights: dict[str, float] | None = None,
		**kwargs: Any,
	) -> None:
		del kwargs
		self.policy = policy
		self.preprocess_cfg = preprocess_cfg
		self.population_size = int(population_size)
		self.num_worlds = int(num_worlds)
		self.reward_prompt_text = reward_prompt_text
		self.history_steps = int(history_steps)
		self.history_stride = int(history_stride)
		self.resample_timesteps = int(resample_timesteps)
		self.elite_size = int(elite_size)
		self.num_iterations = int(num_iterations)
		self.default_metrics = list(metrics) if metrics is not None else ["collision", "offroad", "direction", "tl_violation"]
		self.default_weights = dict(weights) if weights is not None else None

		self.scorers = [Scorer(metrics=[]) for _ in range(num_worlds)]

		if torch_device is None:
			if torch.cuda.is_available():
				torch_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
			else:
				torch_device = torch.device("cpu")
		self.torch_device = torch.device(torch_device)

		if isinstance(temporal_vla, (str, Path)):
			self.temporal_vla = TemporalGemmaVLA.from_pretrained(temporal_vla, device=self.torch_device, dtype=torch.bfloat16)
		else:
			self.temporal_vla = temporal_vla.to(self.torch_device)
		self.temporal_vla.eval()

		if jax_device is None:
			gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
			jax_device = gpu_devices[0] if gpu_devices else jax.devices()[0]
		self.jax_device = jax_device

		self._compute_condition_jit = jax.jit(self.policy.compute_condition)
		self._sample_population_jit = jax.jit(
			lambda *, cond_bf, inst_cond_bf, inst_cond_mask_bf, rng: self._sample_population(
				cond_bf=cond_bf,
				inst_cond_bf=inst_cond_bf,
				inst_cond_mask_bf=inst_cond_mask_bf,
				rng=rng,
			),
		)
		self._resample_population_jit = jax.jit(
			lambda *, cond_bf, inst_cond_bf, inst_cond_mask_bf, proposals_bktd, rng: self._resample_population(
				cond_bf=cond_bf,
				inst_cond_bf=inst_cond_bf,
				inst_cond_mask_bf=inst_cond_mask_bf,
				proposals_bktd=proposals_bktd,
				rng=rng,
			),
		)

	def plan_trajectory(
		self,
		sim_state,
		goal,
		*,
		rng: jax.Array,
		timestep: int = 0,
		mask_goal: bool = False,
		instruction_texts: list[str] | None = None,
		use_subgoal: bool = False,
		target_lane_ids: Any | None = None,
		**kwargs: Any,
	) -> TemporalVLAPlannerResult:
		del instruction_texts
		if use_subgoal:
			del use_subgoal

		rng, key_history, key_current, key_sample = jax.random.split(rng, 4)
		history_steps = self._build_history_steps(int(timestep))
		history_batch, _ = self._build_temporal_history_batch(
			sim_state,
			goal,
			history_steps=history_steps,
			key=key_history,
		)
		current_pre_batch, _ = preprocess_simulator_state(
			sim_state,
			key_current,
			self.preprocess_cfg,
			anchor_step_override=int(timestep),
			goal_step_override=90,
			goal_xy_override=goal,
		)
		current_features = self._prepare_current_features(current_pre_batch.features, mask_goal=mask_goal)

		prompt_ids, prompt_mask = self._build_reward_prompt_batch(
			batch_size=int(history_batch.features["ego_state"].shape[0]),
			num_history_steps=len(history_steps),
		)
		with torch.inference_mode():
			amp_enabled = self.torch_device.type == "cuda"
			with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16, enabled=amp_enabled):
				predictions = self.temporal_vla.generate_predictions(
					input_features=history_batch.features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					max_new_tokens=16,
				)

		reward_texts_b = self._extract_latest_reward_texts(predictions["answer"])
		if target_lane_ids is None:
			target_lane_ids = predictions.get("target_lane_ids", None)
			target_lane_ids_b = [lane_ids[-1] for lane_ids in target_lane_ids] if target_lane_ids is not None else [None] * len(reward_texts_b)
		else:
			target_lane_ids_b = target_lane_ids

		for world_idx, reward_text in enumerate(reward_texts_b):
			current_speed = get_current_speed(self.scorers[world_idx], sim_state, timestep, world_idx)
			for metric_name in self._parse_reward_text(reward_text):
				self._configure_scorer_metric(
					self.scorers[world_idx],
					metric_name,
					goal=goal,
					target_lane_id=target_lane_ids_b[world_idx],
					current_speed=current_speed,
				)

		with jax.default_device(self.jax_device):
			cond_bf, inst_cond_bf = self._compute_condition_jit(current_features)
			current_norm_bktd = self._sample_population_jit(
				cond_bf=cond_bf,
				inst_cond_bf=inst_cond_bf,
				inst_cond_mask_bf=current_features["inst_valid"],
				rng=key_sample,
			)

		current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
			current_norm_bktd,
			current_pre_batch,
		)
		current_scores_bk = self._score_population(current_world_bkt5, self.scorers, sim_state, int(timestep))

		for iteration_index in range(self.num_iterations + 1):
			elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(current_scores_bk, top_k=self.elite_size)
			if iteration_index == self.num_iterations:
				break

			elite_norm_betd = self._gather_population(current_norm_bktd, elite_indices_bk)
			elite_world_bet5, _, _ = self._postprocess_population(elite_norm_betd, current_pre_batch)
			replicated_norm_bktd = self._replicate_elites(elite_norm_betd, population_size=self.population_size)

			rng, key_resample = jax.random.split(rng)
			current_norm_bktd = self._resample_population_jit(
				cond_bf=cond_bf,
				inst_cond_bf=inst_cond_bf,
				inst_cond_mask_bf=current_features["inst_valid"],
				proposals_bktd=replicated_norm_bktd,
				rng=key_resample,
			)
			current_world_bkt5, _, _ = self._postprocess_population(current_norm_bktd, current_pre_batch)
			current_scores_bk = self._score_population(current_world_bkt5, self.scorers, sim_state, int(timestep))
			current_norm_bktd = jnp.concatenate([elite_norm_betd, current_norm_bktd], axis=1)
			current_scores_bk = jnp.concatenate([elite_scores_bk, current_scores_bk], axis=1)
			current_world_bkt5 = jnp.concatenate([elite_world_bet5, current_world_bkt5], axis=1)

		best_indices_b1, _ = self._select_topk_per_scenario(current_scores_bk, top_k=1)
		best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices_b1)
		best_world_bt15 = self._gather_population(current_world_bkt5, best_indices_b1)

		return TemporalVLAPlannerResult(
			start_t_b=current_pre_batch.aux["anchor_step"].astype(jnp.int32),
			trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
			trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
			world_t_seconds_bt=world_t_seconds_bt,
			world_t_valid_bt=world_t_valid_bt,
			aux=current_pre_batch.aux,
			reward_texts=reward_texts_b,
			target_lane_ids=target_lane_ids_b,
		)

	def _build_history_steps(self, timestep: int) -> list[int]:
		steps: list[int] = []
		for offset in range(self.history_steps - 1, -1, -1):
			steps.append(max(0, timestep - offset * self.history_stride))
		return steps

	def _build_temporal_history_batch(
		self,
		sim_state,
		goal,
		*,
		history_steps: Sequence[int],
		key: jax.Array,
	) -> tuple[PreprocessBatch, jax.Array]:
		step_batches: list[PreprocessBatch] = []
		current_key = key
		for step in history_steps:
			current_key, step_key = jax.random.split(current_key)
			pre_batch, _ = preprocess_simulator_state(
				sim_state,
				step_key,
				self.preprocess_cfg,
				anchor_step_override=int(step),
				goal_step_override=90,
				goal_xy_override=goal,
			)
			step_batches.append(pre_batch)

		stacked_features = self._stack_temporal_features([batch.features for batch in step_batches])
		stacked_features["history_valid"] = jnp.ones(
			(stacked_features["ego_state"].shape[0], len(history_steps)),
			dtype=jnp.bool_,
		)
		if "history_timesteps" not in stacked_features:
			stacked_features["history_timesteps"] = jnp.asarray(history_steps, dtype=jnp.int32)[None, :]
		current_pre_batch = step_batches[-1]
		return PreprocessBatch(features=stacked_features, aux=current_pre_batch.aux), current_key

	def _prepare_current_features(self, features: Mapping[str, Any], *, mask_goal: bool = False) -> dict[str, jax.Array]:
		current_features = {key: jnp.asarray(value) for key, value in features.items()}
		batch_size = int(current_features["ego_state"].shape[0])
		current_features["inst_features"] = jnp.zeros((batch_size, self.preprocess_cfg.inst_dim), dtype=jnp.float32)
		current_features["inst_valid"] = jnp.zeros((batch_size,), dtype=jnp.bool_)
		current_features["subgoal_xy"] = jnp.zeros((batch_size, 2), dtype=jnp.float32)
		current_features["subgoal_valid"] = jnp.zeros((batch_size,), dtype=jnp.bool_)
		if mask_goal:
			current_features["goal_xy"] = jnp.zeros_like(current_features["goal_xy"])
			current_features["remaining_timesteps"] = jnp.zeros_like(current_features["remaining_timesteps"])
		return current_features

	def _stack_temporal_features(self, step_features: Sequence[Mapping[str, Any]]) -> dict[str, jax.Array]:
		if not step_features:
			raise ValueError("step_features cannot be empty")
		keys = list(step_features[0].keys())
		stacked: dict[str, jax.Array] = {}
		for key in keys:
			stacked[key] = jnp.stack([jnp.asarray(step_features[step][key]) for step in range(len(step_features))], axis=1)
		return stacked

	def _build_reward_prompt_batch(self, *, batch_size: int, num_history_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
		texts = [[self.reward_prompt_text] * num_history_steps for _ in range(batch_size)]
		flat_texts = [text for scenario_texts in texts for text in scenario_texts]
		tokenized = self.temporal_vla.tokenizer(
			flat_texts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			add_special_tokens=False,
		)
		prompt_ids = tokenized["input_ids"].to(self.torch_device).reshape(batch_size, num_history_steps, -1)
		prompt_mask = tokenized["attention_mask"].to(self.torch_device).reshape(batch_size, num_history_steps, -1)
		return prompt_ids, prompt_mask

	def _extract_latest_reward_texts(self, reward_texts: Sequence[str] | Sequence[Sequence[str]]) -> list[str]:
		if not reward_texts:
			return []
		if isinstance(reward_texts[0], str):
			return [str(text).strip() for text in reward_texts]  # type: ignore[index]
		latest_texts: list[str] = []
		for scenario_texts in reward_texts:
			if not scenario_texts:
				latest_texts.append("")
			else:
				latest_texts.append(str(scenario_texts[-1]).strip())
		return latest_texts

	def _normalize_reward_name(self, name: str) -> tuple[str, dict[str, Any] | None]:
		token = name.strip().lower().replace(" ", "_")
		aliases = {
			"off_road": "offroad",
			"lane_following": "follow_lane",
			"follow_lane": "follow_lane",
			"collision": "collision",
			"offroad": "offroad",
			"direction": "direction",
			"traffic_light": "tl_violation",
			"tl_violation": "tl_violation",
			"progress": "progress",
			"goal": "goal",
			"overtake": "overtake",
			"speed": "speed",
			"set_speed": "speed",
		}
		return aliases.get(token, token), None

	def _build_world_scorers(
		self,
		reward_texts_b: Sequence[str],
		target_lane_ids_b: Sequence[Any],
		sim_state,
		timestep: int,
		*,
		goal,
	) -> list[Scorer]:
		scorers: list[Scorer] = []
		for world_idx, reward_text in enumerate(reward_texts_b):
			scorer = Scorer(metrics=list(self.default_metrics), weights=self.default_weights)
			scorer.reset()
			current_speed = get_current_speed(scorer, sim_state, timestep, world_idx)
			for metric_name in self._parse_reward_text(reward_text):
				self._configure_scorer_metric(
					scorer,
					metric_name,
					goal=goal,
					target_lane_id=target_lane_ids_b[world_idx],
					current_speed=current_speed,
				)
			scorers.append(scorer)
		return scorers

	def _parse_reward_text(self, reward_text: str) -> list[str]:
		parts = [part.strip() for part in reward_text.split(",")]
		return [part for part in parts if part]

	def _resolve_target_lane_id(self, target_lane_ids: Any | None, world_idx: int) -> Any | None:
		if target_lane_ids is None:
			return None
		if isinstance(target_lane_ids, (list, tuple)):
			if not target_lane_ids:
				return None
			item = target_lane_ids[min(world_idx, len(target_lane_ids) - 1)]
			if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
				return item[-1]
			return item
		if isinstance(target_lane_ids, np.ndarray):
			if target_lane_ids.ndim == 2:
				return target_lane_ids[min(world_idx, target_lane_ids.shape[0] - 1), -1]
			if target_lane_ids.ndim == 1:
				return target_lane_ids[min(world_idx, target_lane_ids.shape[0] - 1)]
		return target_lane_ids

	def _configure_scorer_metric(
		self,
		scorer: Scorer,
		metric_name: str,
		*,
		goal,
		target_lane_id: Any | None,
		current_speed: float,
	) -> None:
		normalized, _ = self._normalize_reward_name(metric_name)
		if normalized == "stop":
			scorer.add_metric("set_speed", 0.0)
		if normalized == "accelerate":
			scorer.add_metric("set_speed", current_speed * 1.2)
		if normalized == "decelerate":
			scorer.add_metric("set_speed", current_speed * 0.8)

		if normalized in {"follow_current_lane", "change_lane_left", "change_lane_right"}:
			scorer.add_metric("follow_lane_by_id", target_lane_id)
		return


	def _score_population(
		self,
		trajectories_world_bkt5: jax.Array,
		scorers: Sequence[Scorer],
		sim_state,
		timestep: int,
	) -> jax.Array:
		scores_bk: list[jax.Array] = []
		for world_idx in range(trajectories_world_bkt5.shape[0]):
			world_scores = scorers[world_idx].compute_score(
				trajectories=trajectories_world_bkt5[world_idx, :, :],
				sim_state=sim_state,
				timestep=timestep,
				world_idx=world_idx,
			)
			scores_bk.append(world_scores)
		return jnp.stack(scores_bk)

	def _resample_population(
		self,
		*,
		cond_bf: jax.Array,
		inst_cond_bf: jax.Array,
		inst_cond_mask_bf: jax.Array,
		proposals_bktd: jax.Array,
		rng: jax.Array,
	) -> jax.Array:
		batch_size, population_size, horizon, target_dim = proposals_bktd.shape
		cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
		inst_cond_bkf = jnp.repeat(inst_cond_bf, repeats=population_size, axis=0)
		inst_cond_mask_bkf = jnp.repeat(inst_cond_mask_bf, repeats=population_size, axis=0)
		flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
		samples_btd = self.policy.resample_from_condition(
			cond_bkf,
			inst_cond_bkf,
			inst_cond_mask_bkf,
			flat_proposals,
			self.resample_timesteps,
			rng=rng,
		)
		return samples_btd.reshape(batch_size, population_size, horizon, target_dim)

	def _sample_population(
		self,
		*,
		cond_bf: jax.Array,
		inst_cond_bf: jax.Array,
		inst_cond_mask_bf: jax.Array,
		rng: jax.Array,
	) -> jax.Array:
		batch_size = int(cond_bf.shape[0])
		cond_bkf = jnp.repeat(cond_bf, repeats=self.population_size, axis=0)
		inst_cond_bkf = jnp.repeat(inst_cond_bf, repeats=self.population_size, axis=0)
		inst_cond_mask_bkf = jnp.repeat(inst_cond_mask_bf, repeats=self.population_size, axis=0)
		samples_btd = self.policy.sample_from_condition(
			cond_bkf,
			inst_cond_bkf,
			inst_cond_mask_bkf,
			rng=rng,
		)
		horizon = int(samples_btd.shape[1])
		target_dim = int(samples_btd.shape[2])
		return samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)

	@staticmethod
	def _select_topk_per_scenario(scores_bk: jax.Array, *, top_k: int) -> tuple[jax.Array, jax.Array]:
		top_scores_bk, indices_bk = jax.lax.top_k(scores_bk, top_k)
		return jnp.asarray(indices_bk, dtype=jnp.int32), jnp.asarray(top_scores_bk, dtype=jnp.float32)

	@staticmethod
	def _replicate_elites(elite_betd: jax.Array, *, population_size: int) -> jax.Array:
		batch_size, elite_size, horizon, target_dim = elite_betd.shape
		del batch_size, horizon, target_dim

		base = population_size // elite_size
		remainder = population_size % elite_size
		selector = np.concatenate(
			[
				np.full((base + (1 if elite_idx < remainder else 0),), elite_idx, dtype=np.int32)
				for elite_idx in range(elite_size)
			],
			axis=0,
		)
		return jnp.take(elite_betd, jnp.asarray(selector, dtype=jnp.int32), axis=1)

	def _postprocess_population(
		self,
		proposals_bktd: jax.Array,
		pre_batch: PreprocessBatch,
	) -> tuple[jax.Array, jax.Array, jax.Array]:
		batch_size, population_size, horizon, target_dim = proposals_bktd.shape
		flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
		repeated_aux = self._repeat_dict_leading_axis(pre_batch.aux, repeats=population_size)
		post = postprocess_predictions(flat_proposals, repeated_aux, self.preprocess_cfg)

		world_flat = post["trajectory_world_world_dt"]
		world_steps = int(world_flat.shape[1])
		world_dim = int(world_flat.shape[2])
		trajectories_world_bkt5 = world_flat.reshape(batch_size, population_size, world_steps, world_dim)
		world_t_seconds_bt = jnp.asarray(pre_batch.aux["world_t_seconds"], dtype=jnp.float32)
		world_t_valid_bt = jnp.asarray(pre_batch.aux["world_t_valid"])
		return trajectories_world_bkt5, world_t_seconds_bt, world_t_valid_bt

	@staticmethod
	def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
		gather_indices = jnp.asarray(indices_bm, dtype=jnp.int32)
		trailing_shape = population_bk.shape[2:]
		reshape_shape = gather_indices.shape + (1,) * len(trailing_shape)
		broadcast_shape = gather_indices.shape + trailing_shape
		gather_indices = jnp.broadcast_to(gather_indices.reshape(reshape_shape), broadcast_shape)
		return jnp.take_along_axis(population_bk, gather_indices, axis=1)

	@staticmethod
	def _repeat_dict_leading_axis(values: dict[str, jax.Array], *, repeats: int) -> dict[str, jax.Array]:
		return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.asarray(x), repeats=repeats, axis=0), values)


__all__ = ["TemporalVLAPlanner", "TemporalVLAPlannerResult"]
