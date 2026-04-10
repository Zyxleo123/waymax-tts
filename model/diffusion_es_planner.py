from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from scores.scorer import Scorer
from train.postprocess import postprocess_predictions
from train.preprocess import preprocess_simulator_state
from train.types import PreprocessBatch, PreprocessConfig

from .diffusion_policy import DiffusionPolicy


@dataclass
class PlannerResult:
    start_t_b: jax.Array
    trajectory_norm_btd: jax.Array
    trajectory_world_bt5: jax.Array
    world_t_seconds_bt: jax.Array
    world_t_valid_bt: jax.Array
    best_score_b: jax.Array
    aux: dict[str, jax.Array]
    current_scores_bk: jax.Array = None  # For debugging, not populated in final result
    current_world_bkt5: jax.Array = None  # For debugging, not populated in final result


class DiffusionPlanner:
    """Diffusion evolutionary search planner over batched Waymax scenarios."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        *,
        population_size: int = 64,
        resample_timesteps: int = 5,
        num_worlds: int = 1,
    ) -> None:
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.population_size = int(population_size)
        self.resample_timesteps = int(resample_timesteps)
        self.num_worlds = int(num_worlds)
        self.scorers = [Scorer() for _ in range(num_worlds)]
        
        self._compute_condition_jit = jax.jit(self.policy.compute_condition)
        self._sample_population_jit = jax.jit(
            lambda *, cond_bf, rng: self._sample_population(
                cond_bf=cond_bf,
                rng=rng,
            ),
        )
        self._resample_population_jit = jax.jit(
            lambda *, cond_bf, proposals_bktd, rng: self._resample_population(
                cond_bf=cond_bf,
                proposals_bktd=proposals_bktd,
                rng=rng,
            ),
        )

    def plan_trajectory(
        self,
        sim_state,
        *,
        rng: jax.Array,
        elite_size: int = 4,
        num_iterations: int = 5,
        timestep: int = 0,
    ) -> PlannerResult:
        elite_size = int(elite_size)
        num_iterations = int(num_iterations)

        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            timestep=timestep,
        )
        cond_bf = self._compute_condition_jit(pre_batch.features)

        current_norm_bktd = self._sample_population_jit(
            cond_bf=cond_bf,
            rng=key_sample,
        )
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        current_scores_bk = []
        for world_idx in range(current_world_bkt5.shape[0]):
            current_scores_bk.append(self.scorers[world_idx].compute_score(
                trajectories=current_world_bkt5[world_idx, :, :],
                sim_state=sim_state,
                timestep=timestep,
                world_idx=world_idx
            ))
        current_scores_bk = jnp.stack(current_scores_bk)
        # print(f"Initial scores: {current_scores_bk} \n Shape: {current_scores_bk.shape} \n Mean: {jnp.mean(current_scores_bk)}, Std: {jnp.std(current_scores_bk)}")
        for iteration_index in range(num_iterations + 1):
            elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk,
                top_k=elite_size,
            )
            if iteration_index == num_iterations:
                break

            elite_norm_betd = self._gather_population(
                current_norm_bktd,
                elite_indices_bk,
            )
            elite_world_bet5, _, _ = self._postprocess_population(
                elite_norm_betd,
                pre_batch,
            )
            replicated_norm_bktd = self._replicate_elites(
                elite_norm_betd,
                population_size=self.population_size,
            )

            rng, key_resample = jax.random.split(rng)
            current_norm_bktd = self._resample_population_jit(
                cond_bf=cond_bf,
                proposals_bktd=replicated_norm_bktd,
                rng=key_resample,
            )
            current_world_bkt5, _, _ = self._postprocess_population(
                current_norm_bktd,
                pre_batch,
            )
            current_scores_bk = []
            for world_idx in range(current_world_bkt5.shape[0]):
                current_scores_k = self.scorers[world_idx].compute_score(
                    trajectories=current_world_bkt5[world_idx, :, :],
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx
                )
                current_scores_bk.append(current_scores_k)
            current_scores_bk = jnp.stack(current_scores_bk)
            current_norm_bktd = jnp.concat([elite_norm_betd, current_norm_bktd], axis=1)
            current_scores_bk = jnp.concat([elite_scores_bk, current_scores_bk], axis=1)
            current_world_bkt5 = jnp.concat([elite_world_bet5, current_world_bkt5], axis=1)

        # print(f"Final scores: {current_scores_bk} \n Shape: {current_scores_bk.shape} \n Mean: {jnp.mean(current_scores_bk)}, Std: {jnp.std(current_scores_bk)}")
        best_indices_b1, best_scores_b1 = self._select_topk_per_scenario(
            current_scores_bk,
            top_k=1,
        )
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices_b1)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices_b1)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            best_score_b=jnp.squeeze(best_scores_b1, axis=1),
            aux=pre_batch.aux,
            current_scores_bk=current_scores_bk,
            current_world_bkt5=current_world_bkt5,
        )

    def _sample_population(
        self,
        *,
        cond_bf: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size = int(cond_bf.shape[0])
        cond_bkf = jnp.repeat(cond_bf, repeats=self.population_size, axis=0)
        samples_btd = self.policy.sample_from_condition(cond_bkf, rng=rng)
        horizon = int(samples_btd.shape[1])
        target_dim = int(samples_btd.shape[2])
        return samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)

    def _resample_population(
        self,
        *,
        cond_bf: jax.Array,
        proposals_bktd: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
        flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        samples_btd = self.policy.resample_from_condition(
            cond_bkf,
            flat_proposals,
            self.resample_timesteps,
            rng=rng,
        )
        return samples_btd.reshape(batch_size, population_size, horizon, target_dim)

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
    def _select_topk_per_scenario(
        scores_bk: jax.Array,
        *,
        top_k: int,
    ) -> tuple[jax.Array, jax.Array]:
        top_scores_bk, indices_bk = jax.lax.top_k(scores_bk, top_k)
        return (
            jnp.asarray(indices_bk, dtype=jnp.int32),
            jnp.asarray(top_scores_bk, dtype=jnp.float32),
        )

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

    def _update_archive(
        self,
        *,
        archive_norm_bktd: jax.Array,
        archive_world_bkt5: jax.Array,
        archive_scores_bk: jax.Array,
        candidate_norm_bktd: jax.Array,
        candidate_world_bkt5: jax.Array,
        candidate_scores_bk: jax.Array,
        keep_top_k: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        combined_scores_bk = jnp.concatenate([archive_scores_bk, candidate_scores_bk], axis=1)
        combined_norm_bktd = jnp.concatenate([archive_norm_bktd, candidate_norm_bktd], axis=1)
        combined_world_bkt5 = jnp.concatenate([archive_world_bkt5, candidate_world_bkt5], axis=1)

        keep_indices_bk, keep_scores_bk = self._select_topk_per_scenario(
            combined_scores_bk,
            top_k=keep_top_k,
        )
        kept_norm_bktd = self._gather_population(combined_norm_bktd, keep_indices_bk)
        kept_world_bkt5 = self._gather_population(combined_world_bkt5, keep_indices_bk)
        return kept_norm_bktd, kept_world_bkt5, keep_scores_bk

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
