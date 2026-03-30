from __future__ import annotations

import dataclasses
import time
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
class PlannerScoreScenarioTiming:
    batch_index: int
    score_proposals_s: float
    total_s: float


@dataclass
class PlannerScoreTiming:
    total_s: float
    scenarios: tuple[PlannerScoreScenarioTiming, ...]


@dataclass
class PlannerIterationTiming:
    iteration_index: int
    select_topk_s: float
    gather_elites_s: float
    replicate_elites_s: float
    resample_s: float
    postprocess_s: float
    score: PlannerScoreTiming
    archive_update_s: float
    total_s: float


@dataclass
class PlannerTiming:
    preprocess_s: float
    condition_s: float
    initial_sample_s: float
    initial_postprocess_s: float
    world_cache_build_s: float
    initial_score: PlannerScoreTiming
    iterations: tuple[PlannerIterationTiming, ...]
    total_s: float


@dataclass
class PlannerResult:
    start_t_b: jax.Array
    trajectory_norm_btd: jax.Array
    trajectory_world_bt5: jax.Array
    world_t_seconds_bt: jax.Array
    world_t_valid_bt: jax.Array
    best_score_b: jax.Array
    aux: dict[str, jax.Array]


class DiffusionPlanner:
    """Diffusion evolutionary search planner over batched Waymax scenarios."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        *,
        population_size: int = 64,
        resample_timesteps: int = 5,
        scorer: ProposalScorer | None = None,
        speed_limit_mps: float | None = None,
        world_kwargs: Mapping[str, Any] | None = None,
        synchronize_timings: bool = False,
    ) -> None:
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.population_size = int(population_size)
        if self.population_size < 1:
            raise ValueError(
                f"population_size must be >= 1, got {self.population_size}."
            )
        self.resample_timesteps = int(resample_timesteps)
        if self.resample_timesteps < 1:
            raise ValueError(
                f"resample_timesteps must be >= 1, got {self.resample_timesteps}."
            )
        if self.resample_timesteps > int(self.policy.diffusion.timesteps):
            raise ValueError(
                "resample_timesteps exceeds diffusion horizon: "
                f"{self.resample_timesteps} > {int(self.policy.diffusion.timesteps)}."
            )
        self.scorer = scorer if scorer is not None else ProposalScorer()
        # ProposalScorer enables speed_limit by default, so provide a default value explicitly.
        self.scorer.set_metric_params("speed_limit", speed_limit_mps=speed_limit_mps)
        self.world_kwargs = dict(world_kwargs or {})
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
        self._world_cache: list[World] | None = None
        self._world_cache_key: tuple[int, int, int] | None = None

    def change_lane_right(self, sim_state) -> bool:
        assert self._world_cache is not None, "World cache must be initialized before calling change_lane_right"
        for world in self._world_cache:
            current_lane_idx = world.ego_lane_node_index
            if current_lane_idx is None:
                current_lane_idx = world.get_current_lane(world.ego_absolute_index)
            if current_lane_idx is None:
                return False

            right_lane_idx = world.get_right_lane(int(current_lane_idx))
            if right_lane_idx is None:
                return False

            _, right_centerline_xy, _ = world.extend_centerline(int(right_lane_idx))
            if int(right_centerline_xy.shape[0]) < 2:
                return False

            self.scorer.set_metric_params(
                "lane_following",
                reference_centerline_xy=np.asarray(right_centerline_xy, dtype=np.float32),
            )
        return True

    def plan_trajectory(
        self,
        sim_state,
        *,
        rng: jax.Array,
        elite_size: int = 4,
        num_iterations: int = 5,
        anchor_step_override_b: jax.Array | np.ndarray | None = None,
    ) -> PlannerResult:
        elite_size = int(elite_size)
        num_iterations = int(num_iterations)

        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override_b=anchor_step_override_b,
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
        world_cache = self.build_world_cache(
            sim_state=sim_state,
            aux=pre_batch.aux,
            batch_size=int(current_world_bkt5.shape[0]),
            prediction_horizon=max(
                min(int(current_world_bkt5.shape[2]), int(self.scorer.max_score_steps)) - 1,
                1,
            ),
        )
        current_scores_bk = self._score_population(
            trajectories_world_bkt5=current_world_bkt5,
            sim_state=sim_state,
            aux=pre_batch.aux,
            world_cache=world_cache,
        )

        archive_norm_bktd = current_norm_bktd
        archive_world_bkt5 = current_world_bkt5
        archive_scores_bk = current_scores_bk

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
            current_scores_bk = self._score_population(
                trajectories_world_bkt5=current_world_bkt5,
                sim_state=sim_state,
                aux=pre_batch.aux,
                world_cache=world_cache,
            )

            archive_norm_bktd, archive_world_bkt5, archive_scores_bk = self._update_archive(
                archive_norm_bktd=archive_norm_bktd,
                archive_world_bkt5=archive_world_bkt5,
                archive_scores_bk=archive_scores_bk,
                candidate_norm_bktd=current_norm_bktd,
                candidate_world_bkt5=current_world_bkt5,
                candidate_scores_bk=current_scores_bk,
                keep_top_k=self.population_size,
            )

        best_indices_b1, best_scores_b1 = self._select_topk_per_scenario(
            archive_scores_bk,
            top_k=1,
        )
        best_norm_bt1d = self._gather_population(archive_norm_bktd, best_indices_b1)
        best_world_bt15 = self._gather_population(archive_world_bkt5, best_indices_b1)

        return PlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            best_score_b=jnp.squeeze(best_scores_b1, axis=1),
            aux=pre_batch.aux,
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

    def _score_population(
        self,
        *,
        trajectories_world_bkt5: jax.Array,
        sim_state,
        aux: dict[str, jax.Array],
        world_cache: list[World] | None = None,
    ) -> tuple[jax.Array, PlannerScoreTiming]:
        del sim_state, aux
        batch_size = int(trajectories_world_bkt5.shape[0])

        scores_b: list[jax.Array] = []
        for batch_index in range(batch_size):
            world = world_cache[batch_index]
            result = self.scorer.score_proposals(
                trajectories=trajectories_world_bkt5[batch_index],
                world=world,
            )
            scores_b.append(jnp.asarray(result.final_scores, dtype=jnp.float32))
            
        return jnp.stack(scores_b, axis=0).astype(jnp.float32)

    def build_world_cache(
        self,
        *,
        sim_state,
        aux: dict[str, jax.Array],
        batch_size: int,
        prediction_horizon: int,
        world_cache: list[World] | None = None,
    ) -> list[World]:
        scenario_cache_key = self._scenario_cache_key(sim_state, batch_size)
        if self._world_cache_key != scenario_cache_key:
            self._world_cache = None
        if self._world_cache is None:
            world_cache = [
                World(
                    sim_state,
                    batch_index=batch_index,
                    prediction_horizon=prediction_horizon,
                    prediction_dt_s=float(aux["world_dt_seconds"][batch_index]),
                    **self.world_kwargs,
                )
                for batch_index in range(batch_size)
            ]
            self._world_cache = world_cache
            self._world_cache_key = scenario_cache_key
            return world_cache
        else:
            for batch_index in range(batch_size):
                self._world_cache[batch_index].sync(
                    sim_state,
                    prediction_horizon=prediction_horizon,
                    prediction_dt_s=float(aux["world_dt_seconds"][batch_index]),
                )
            world_cache = self._world_cache
            return world_cache

    def reset_world_cache(self) -> None:
        self._world_cache = None
        self._world_cache_key = None

    @staticmethod
    def _scenario_cache_key(sim_state, batch_size: int) -> tuple[int, int, int]:
        roadgraph_points = getattr(sim_state, "road_graph", None)
        if roadgraph_points is None:
            roadgraph_points = getattr(sim_state, "roadgraph_points", None)
        object_ids = getattr(sim_state.object_metadata, "ids", None)
        return (id(roadgraph_points), id(object_ids), int(batch_size))

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
