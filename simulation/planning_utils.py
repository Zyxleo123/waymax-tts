from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from planner.abstract_planner import (
    AbstractPlanner,
    PlannerResult,
)


@dataclasses.dataclass
class PredictionBatch:
    start_t_b: jax.Array
    trajectory_world_bt5: jax.Array
    world_t_seconds_bt: jax.Array
    world_t_valid_bt: jax.Array
    aux: dict[str, jax.Array]


def get_sdc_indices_for_batched_state(state_batched) -> np.ndarray:
    """Returns one SDC index per batch item; errors if not exactly one."""
    is_sdc = np.asarray(state_batched.object_metadata.is_sdc).astype(bool)
    if is_sdc.ndim != 2:
        raise ValueError(f"Expected is_sdc shape [B,N], got {is_sdc.shape}.")
    sdc_count = np.sum(is_sdc, axis=1)
    if np.any(sdc_count != 1):
        bad = np.where(sdc_count != 1)[0].tolist()
        raise ValueError(f"Expected exactly one SDC per batch item; bad indices: {bad}.")
    return np.argmax(is_sdc.astype(np.int32), axis=1)


def apply_ego_replacements_to_expanded_state(
    expanded_state,
    *,
    start_t_b: np.ndarray,
    traj_bt5: np.ndarray,
):
    """Applies ego trajectory replacements to expanded batched state."""
    traj = expanded_state.log_trajectory
    x = np.array(traj.x, copy=True)
    y = np.array(traj.y, copy=True)
    yaw = np.array(traj.yaw, copy=True)
    vel_x = np.array(traj.vel_x, copy=True)
    vel_y = np.array(traj.vel_y, copy=True)
    valid = np.array(traj.valid, copy=True)
    num_steps = traj.x.shape[-1]

    sdc_idx = get_sdc_indices_for_batched_state(expanded_state)
    horizon = int(traj_bt5.shape[1])

    for b in range(traj_bt5.shape[0]):
        start_t = int(start_t_b[b])
        end_t = start_t + horizon
        if end_t > num_steps:
            end_t = num_steps
        ego = int(sdc_idx[b])
        sl = slice(start_t, end_t)
        repl = traj_bt5[b]
        x[b, ego, sl] = repl[:, 0]
        y[b, ego, sl] = repl[:, 1]
        yaw[b, ego, sl] = repl[:, 2]
        vel_x[b, ego, sl] = repl[:, 3]
        vel_y[b, ego, sl] = repl[:, 4]
        valid[b, ego, sl] = True

    new_log = traj.replace(
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=valid,
    )
    return dataclasses.replace(expanded_state, log_trajectory=new_log)


def predict_planner_trajectories_with_periodic_replan(
    cfg,
    sim_state,
    goal,
    planner: AbstractPlanner,
    replan_interval_steps: int,
    rng_key: jax.Array,
):
    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b1 = np.zeros((num_worlds,), dtype=np.int32)
    instructions = [dict() for _ in range(num_worlds)]   
    rng_key, key_initial = jax.random.split(rng_key)

    initial_result = planner.plan_trajectory(
        sim_state,
        goal,
        rng=key_initial,
        timestep=0,
        mask_goal=cfg.mask_goal
    )
    instructions_t = getattr(initial_result, "instruction_texts", None)  # touch to avoid unused import warning for VLAPlannerResult
    for b in range(num_worlds):
        instructions[b]["t0"] = instructions_t[b] if instructions_t is not None else ""

    initial_traj_bt5 = np.asarray(initial_result.trajectory_world_bt5, dtype=np.float32)
    model_horizon_len = int(initial_traj_bt5.shape[1])

    traj_bt5 = np.zeros((num_worlds, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_bt5[:, :init_copy_len, :] = initial_traj_bt5[:, :init_copy_len, :]
    
    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in tqdm(range(interval, max_steps, interval)):
        replaced_state = apply_ego_replacements_to_expanded_state(
            sim_state,
            start_t_b=start_t_b1,
            traj_bt5=traj_bt5,
        )
        rng_key, key_replan = jax.random.split(rng_key)

        repl_result = planner.plan_trajectory(
            replaced_state,
            goal,
            rng=key_replan,
            timestep=int(step_offset),
            mask_goal=cfg.mask_goal
        )
        instructions_t = getattr(repl_result, "instruction_texts", None)
        for b in range(num_worlds):
            instructions[b][f"t{step_offset}"] = instructions_t[b] if instructions_t is not None else ""

        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            repl_traj_bt5 = np.asarray(repl_result.trajectory_world_bt5, dtype=np.float32)
            copy_len = min(remaining, int(repl_traj_bt5.shape[1]))
            traj_bt5[:, step_offset : step_offset + copy_len, :] = repl_traj_bt5[:, :copy_len, :]

    world_dt = float(np.asarray(initial_result.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)

    pred = PredictionBatch(
        start_t_b=jnp.zeros((num_worlds,), dtype=jnp.int32),
        trajectory_world_bt5=jnp.asarray(traj_bt5, dtype=jnp.float32),
        world_t_seconds_bt=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bt=jnp.asarray(world_valid),
        aux=initial_result.aux,
    )

    return pred, replaced_state, instructions
