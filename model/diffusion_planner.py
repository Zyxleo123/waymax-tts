from __future__ import annotations

from dataclasses import dataclass

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
    aux: dict[str, jax.Array]
    current_world_bkt5: jax.Array = None  # For debugging, not populated in final result


class DiffusionPlanner:
    """Diffusion evolutionary search planner over batched Waymax scenarios."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        preprocess_cfg: PreprocessConfig,
        population_size: int = 1,
        num_worlds: int = 1,
    ) -> None:
        self.policy = policy
        self.preprocess_cfg = preprocess_cfg
        self.num_worlds = int(num_worlds)
        self.population_size = int(population_size)
        self._compute_condition_jit = jax.jit(self.policy.compute_condition)
        self._sample_population_jit = jax.jit(
            lambda *, cond_bf, rng: self._sample_population(
                cond_bf=cond_bf,
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
    ) -> PlannerResult:
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            timestep=timestep,
            goal=goal,
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
