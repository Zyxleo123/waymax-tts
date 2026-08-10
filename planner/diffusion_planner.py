from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scores.scorer import Scorer
from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig

from model.diffusion.diffusion_policy import DiffusionPolicy
from planner.abstract_planner import AbstractPlanner, PlannerResult


class DiffusionPlanner(AbstractPlanner):
    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        population_size: int = 1,
        num_worlds: int = 1,
        **kwargs: Any,
    ) -> None:
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.num_worlds = int(num_worlds)
        self.population_size = int(population_size)
        self._compute_condition_jit = jax.jit(self.policy.compute_condition)
        self._sample_population_jit = jax.jit(
            lambda *, cond_bf, inst_cond_bf, inst_cond_mask_bf, rng: self._sample_population(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=inst_cond_mask_bf,
                rng=rng,
            ),
        )

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng: jax.Array,
        instruction: jnp.ndarray = None,
        instruction_mask: jnp.ndarray = None,
        timestep: int = 0,
        mask_goal: bool = False,
        goal_timestep: jnp.ndarray | int | None = None,
        **kwargs: Any,
    ) -> PlannerResult:
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        # Use each scene's real goal timestep (its last valid ego log step),
        # threaded in as `goal_timestep`, instead of a hard-coded constant. The
        # goal step only sets `remaining_timesteps=(goal_step-anchor)/100` in the
        # conditioning; a fixed 90 mis-told the policy how much time remained for
        # every scene whose horizon differed.
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=goal_timestep,
            goal_xy_override=goal,
        )
        features = pre_batch.features
        if instruction is None:
            instruction = jnp.zeros((self.num_worlds, self.preprocess_cfg.inst_dim), dtype=jnp.float32)
            instruction_mask = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        else:
            raise NotImplementedError("Conditioning on instruction is not yet implemented.")
        features["inst_features"] = instruction
        features["inst_valid"] = instruction_mask
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        features["subgoal_xy"] = jnp.zeros((self.num_worlds, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((self.num_worlds,), dtype=jnp.float32)
        cond_bf, inst_cond_bf = self._compute_condition_jit(features)

        current_norm_bktd = self._sample_population_jit(
            cond_bf=cond_bf,
            inst_cond_bf=inst_cond_bf,
            inst_cond_mask_bf=instruction_mask,
            rng=key_sample,
        )
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        best_indices = jnp.zeros((cond_bf.shape[0], 1), dtype=jnp.int32)
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=pre_batch.aux,
        )

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
        inst_cond_bf = jnp.repeat(inst_cond_bf, repeats=self.population_size, axis=0)
        inst_cond_mask_bf = jnp.repeat(inst_cond_mask_bf, repeats=self.population_size, axis=0)
        samples_btd = self.policy.sample_from_condition(cond_bkf, inst_cond_bf, inst_cond_mask_bf, rng=rng)
        horizon = int(samples_btd.shape[1])
        target_dim = int(samples_btd.shape[2])
        return samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)

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
        trajectories_world_bkt5 = world_flat.reshape(
            batch_size,
            population_size,
            world_steps,
            world_dim,
        )
        world_t_seconds_bt = jnp.asarray(pre_batch.aux["world_t_seconds"], dtype=jnp.float32)
        world_t_valid_bt = jnp.asarray(pre_batch.aux["world_t_valid"])
        return trajectories_world_bkt5, world_t_seconds_bt, world_t_valid_bt

    @staticmethod
    def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
        gather_indices = jnp.asarray(indices_bm, dtype=jnp.int32)
        trailing_shape = population_bk.shape[2:]
        reshape_shape = gather_indices.shape + (1,) * len(trailing_shape)
        broadcast_shape = gather_indices.shape + trailing_shape
        gather_indices = jnp.broadcast_to(
            gather_indices.reshape(reshape_shape),
            broadcast_shape,
        )
        return jnp.take_along_axis(population_bk, gather_indices, axis=1)

    @staticmethod
    def _repeat_dict_leading_axis(
        values: dict[str, jax.Array],
        *,
        repeats: int,
    ) -> dict[str, jax.Array]:
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x), repeats=repeats, axis=0),
            values,
        )