from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from flax import nnx
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diffusion_language_planner import (  # noqa: E402
    DiffusionLanguagePlanner,
    PlannerResult,
)
from train.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from simulation_utils.visualize import _save_population_xy_score_image  # noqa: E402

def _planner_result_to_prediction_batch(result: PlannerResult) -> PredictionBatch:
    return PredictionBatch(
        start_t_b=result.start_t_b,
        trajectories_world_bkt5=result.trajectory_world_bkt5,
        world_t_seconds_bk=result.world_t_seconds_bt,
        world_t_valid_bk=result.world_t_valid_bt,
        aux=result.aux,
    )

def _predict_planner_trajectories_with_periodic_replan(
    sim_state,
    planner: DiffusionLanguagePlanner,
    task: str,
    target_vehicle_indices: list[int],
    *,
    output_dir: Path,
    scenario_indices: list[int],
    rng_key: jax.Array,
    elite_size: int,
    num_iterations: int,
    replan_interval_steps: int,
    front_x: float,
    back_x: float,
    front_y: float,
    back_y: float,
) -> tuple[PredictionBatch, np.ndarray, list[dict[str, Any]]]:
    if int(replan_interval_steps) <= 0:
        raise ValueError(f"replan_interval_steps must be > 0, got {replan_interval_steps}.")

    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b1 = np.zeros((num_worlds,), dtype=np.int32)
    
    rng_key, key_initial = jax.random.split(rng_key)

    planner.reset(num_worlds=num_worlds)

    initial_result = planner.plan_trajectory(
        sim_state,
        rng=key_initial,
        elite_size=int(elite_size),
        num_iterations=int(num_iterations),
        timestep=0,
    )
    # _save_population_xy_score_image(
    #     sim_state=sim_state,
    #     timestep=0,
    #     current_world_bkt5=np.asarray(initial_result.current_world_bkt5),
    #     current_scores_bk=np.asarray(initial_result.current_scores_bk),
    #     output_path=output_dir / "population_xy_scores",
    #     scenario_indices=scenario_indices,
    #     front_x=float(front_x),
    #     back_x=float(back_x),
    #     front_y=float(front_y),
    #     back_y=float(back_y),
    #     target_lanes=[planner.scorers[i].target_lane for i in range(num_worlds)],
    # )

    initial_pred = _planner_result_to_prediction_batch(initial_result)
    initial_traj_bl5 = np.asarray(initial_result.trajectory_world_bkt5, dtype=np.float32)
    model_horizon_len = int(initial_traj_bl5.shape[1])

    traj_b1l5 = np.zeros((num_worlds, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_b1l5[:, :init_copy_len, :] = initial_traj_bl5[:, :init_copy_len, :]
    current_score_b = np.asarray(initial_result.best_score_b, dtype=np.float32)

    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in tqdm(range(interval, max_steps, interval)):
        replaced_state = _apply_ego_replacements_to_expanded_state(
            sim_state,
            start_t_bk=start_t_b1,
            traj_bkl5=traj_b1l5,
        )
        rng_key, key_replan = jax.random.split(rng_key)

        repl_result = planner.plan_trajectory(
            replaced_state,
            rng=key_replan,
            elite_size=int(elite_size),
            num_iterations=int(num_iterations),
            timestep=int(step_offset),
        )
        # _save_population_xy_score_image(
        #     sim_state=replaced_state,
        #     timestep=int(step_offset),
        #     current_world_bkt5=np.asarray(repl_result.current_world_bkt5),
        #     current_scores_bk=np.asarray(repl_result.current_scores_bk),
        #     output_path=output_dir / "population_xy_scores",
        #     scenario_indices=scenario_indices,
        #     front_x=float(front_x),
        #     back_x=float(back_x),
        #     front_y=float(front_y),
        #     back_y=float(back_y),
        #     target_lanes=[planner.scorers[i].target_lane for i in range(num_worlds)],
        # )

        repl_score_b = np.asarray(repl_result.best_score_b, dtype=np.float32)
        repl_traj_b1l5 = np.asarray(repl_result.trajectory_world_bkt5, dtype=np.float32)

        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            copy_len = min(remaining, int(repl_traj_b1l5.shape[1]))
            traj_b1l5[:, step_offset : step_offset + copy_len, :] = repl_traj_b1l5[:, :copy_len, :]

        current_score_b = repl_score_b
    world_dt = float(np.asarray(initial_pred.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)

    pred = PredictionBatch(
        start_t_b=jnp.zeros((num_worlds,), dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_b1l5[:, None, :, :], dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial_pred.aux,
    )

    planner.timestep = 90
    planner.sim_state = replaced_state
    task_results = []
    for world_idx in range(num_worlds):
        planner.world_pointer = world_idx
        if task == "overtake":
            task_result = planner.is_ahead_of(target_vehicle_indices[world_idx])
        elif task == "give_way":
            task_result = planner.is_behind_of(target_vehicle_indices[world_idx])
        elif task == "pull_over":
            on_roadside = planner.on_target_lane()
            stopped = planner.get_current_speed() < 0.1
            task_result = on_roadside & stopped
        else:
            raise ValueError(f"Unsupported task: {task}")
        task_results.append(task_result)
    task_results = np.asarray(task_results, dtype=bool)
    return pred, current_score_b[:, None], task_results
