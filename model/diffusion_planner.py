from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from metrics.helpers import World
from metrics.scorer import ProposalScorer
from train.postprocess import postprocess_predictions
from train.preprocess import preprocess_simulator_state
from train.types import PreprocessBatch, PreprocessConfig

from .diffusion_policy import DiffusionPolicy


@dataclass
class PlannerIterationStats:
    iteration_index: int
    population_scores_bk: jax.Array
    elite_indices_bk: jax.Array
    elite_scores_bk: jax.Array
    archive_scores_bk: jax.Array


@dataclass
class PlannerResult:
    start_t_b: jax.Array
    trajectories_norm_bktd: jax.Array
    trajectories_world_bkt5: jax.Array
    world_t_seconds_bt: jax.Array
    world_t_valid_bt: jax.Array
    final_scores_bk: jax.Array
    history: tuple[PlannerIterationStats, ...]
    aux: dict[str, jax.Array]


class DiffusionPlanner:
    """Diffusion evolutionary search planner over batched Waymax scenarios."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        *,
        scorer: ProposalScorer | None = None,
        speed_limit_mps: float | None = None,
        world_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.scorer = scorer if scorer is not None else ProposalScorer()
        # ProposalScorer enables speed_limit by default, so provide a default value explicitly.
        self.scorer.set_metric_params("speed_limit", speed_limit_mps=speed_limit_mps)
        self.world_kwargs = dict(world_kwargs or {})

    def plan_trajectory(
        self,
        sim_state,
        *,
        rng: jax.Array,
        population_size: int = 64,
        elite_size: int = 4,
        num_iterations: int = 5,
        resample_timesteps: int = 20,
        anchor_step_override_b: jax.Array | np.ndarray | None = None,
    ) -> PlannerResult:
        population_size = int(population_size)
        elite_size = int(elite_size)
        num_iterations = int(num_iterations)
        resample_timesteps = int(resample_timesteps)

        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}.")
        if elite_size < 1 or elite_size > population_size:
            raise ValueError(
                f"elite_size must be in [1, {population_size}], got {elite_size}."
            )
        if num_iterations < 0:
            raise ValueError(f"num_iterations must be >= 0, got {num_iterations}.")
        if resample_timesteps < 1:
            raise ValueError(
                f"resample_timesteps must be >= 1, got {resample_timesteps}."
            )
        if resample_timesteps > int(self.policy.diffusion.timesteps):
            raise ValueError(
                "resample_timesteps exceeds diffusion horizon: "
                f"{resample_timesteps} > {int(self.policy.diffusion.timesteps)}."
            )

        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override_b=anchor_step_override_b,
        )
        cond_bf = self.policy.compute_condition(pre_batch.features)

        current_norm_bktd = self._sample_population(
            cond_bf=cond_bf,
            population_size=population_size,
            rng=key_sample,
        )
        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        current_scores_bk = self._score_population(
            trajectories_world_bkt5=current_world_bkt5,
            sim_state=sim_state,
            aux=pre_batch.aux,
        )

        archive_norm_bktd = current_norm_bktd
        archive_world_bkt5 = current_world_bkt5
        archive_scores_bk = current_scores_bk
        history: list[PlannerIterationStats] = []

        for iteration_index in range(num_iterations + 1):
            elite_indices_bk, elite_scores_bk = self._select_topk_per_scenario(
                current_scores_bk,
                top_k=elite_size,
            )
            history.append(
                PlannerIterationStats(
                    iteration_index=iteration_index,
                    population_scores_bk=current_scores_bk,
                    elite_indices_bk=elite_indices_bk,
                    elite_scores_bk=elite_scores_bk,
                    archive_scores_bk=archive_scores_bk,
                )
            )
            if iteration_index == num_iterations:
                break

            elite_norm_betd = self._gather_population(
                current_norm_bktd,
                elite_indices_bk,
            )
            replicated_norm_bktd = self._replicate_elites(
                elite_norm_betd,
                population_size=population_size,
            )

            rng, key_resample = jax.random.split(rng)
            current_norm_bktd = self._resample_population(
                cond_bf=cond_bf,
                proposals_bktd=replicated_norm_bktd,
                resample_timesteps=resample_timesteps,
                rng=key_resample,
            )
            current_world_bkt5, _, _ = self._postprocess_population(
                current_norm_bktd,
                pre_batch,
            )
            current_scores_bk = self._score_population(
                trajectories_world_bkt5=current_world_bkt5,
                sim_state=sim_state,
                aux=pre_batch.aux,
            )

            archive_norm_bktd, archive_world_bkt5, archive_scores_bk = self._update_archive(
                archive_norm_bktd=archive_norm_bktd,
                archive_world_bkt5=archive_world_bkt5,
                archive_scores_bk=archive_scores_bk,
                candidate_norm_bktd=current_norm_bktd,
                candidate_world_bkt5=current_world_bkt5,
                candidate_scores_bk=current_scores_bk,
                keep_top_k=population_size,
            )

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectories_norm_bktd=archive_norm_bktd,
            trajectories_world_bkt5=archive_world_bkt5,
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            final_scores_bk=archive_scores_bk,
            history=tuple(history),
            aux=pre_batch.aux,
        )

    def _sample_population(
        self,
        *,
        cond_bf: jax.Array,
        population_size: int,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size = int(cond_bf.shape[0])
        cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
        samples_btd = self.policy.sample_from_condition(cond_bkf, rng=rng)
        horizon = int(samples_btd.shape[1])
        target_dim = int(samples_btd.shape[2])
        return samples_btd.reshape(batch_size, population_size, horizon, target_dim)

    def _resample_population(
        self,
        *,
        cond_bf: jax.Array,
        proposals_bktd: jax.Array,
        resample_timesteps: int,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        cond_bkf = jnp.repeat(cond_bf, repeats=population_size, axis=0)
        flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        samples_btd = self.policy.resample_from_condition(
            cond_bkf,
            flat_proposals,
            resample_timesteps,
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

    def _score_population(
        self,
        *,
        trajectories_world_bkt5: jax.Array,
        sim_state,
        aux: dict[str, jax.Array],
    ) -> jax.Array:
        traj_np = np.asarray(trajectories_world_bkt5, dtype=np.float32)
        world_dt_b = np.asarray(aux["world_dt_seconds"], dtype=np.float32)
        batch_size = int(traj_np.shape[0])

        scores_b = []
        for batch_index in range(batch_size):
            world = World(
                sim_state,
                batch_index=batch_index,
                prediction_horizon=max(int(traj_np.shape[2]) - 1, 1),
                prediction_dt_s=float(world_dt_b[batch_index]),
                **self.world_kwargs,
            )
            result = self.scorer.score_proposals(
                trajectories=jnp.asarray(traj_np[batch_index]),
                world=world,
            )
            scores_b.append(np.asarray(result.final_scores, dtype=np.float32))
        return jnp.asarray(np.stack(scores_b, axis=0), dtype=jnp.float32)

    @staticmethod
    def _select_topk_per_scenario(
        scores_bk: jax.Array,
        *,
        top_k: int,
    ) -> tuple[jax.Array, jax.Array]:
        score_np = np.asarray(scores_bk, dtype=np.float32)
        indices_bk = np.argsort(-score_np, axis=1)[:, :top_k]
        top_scores_bk = np.take_along_axis(score_np, indices_bk, axis=1)
        return (
            jnp.asarray(indices_bk, dtype=jnp.int32),
            jnp.asarray(top_scores_bk, dtype=jnp.float32),
        )

    @staticmethod
    def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
        population_np = np.asarray(population_bk)
        indices_np = np.asarray(indices_bm, dtype=np.int32)
        gathered = np.take_along_axis(
            population_np,
            indices_np[..., None, None],
            axis=1,
        )
        return jnp.asarray(gathered)

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
        replicated = np.asarray(elite_betd)[:, selector, :, :]
        return jnp.asarray(replicated)

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
