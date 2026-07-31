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
    start_t_b: np.ndarray | int,
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
        start_t = int(start_t_b[b]) if isinstance(start_t_b, np.ndarray) else int(start_t_b)
        end_t = start_t + horizon
        if end_t > num_steps:
            end_t = num_steps
            
        ego = int(sdc_idx[b])
        sl = slice(start_t, end_t)
        repl = traj_bt5[b]
        x[b, ego, sl] = repl[:end_t - start_t, 0]
        y[b, ego, sl] = repl[:end_t - start_t, 1]
        yaw[b, ego, sl] = repl[:end_t - start_t, 2]
        vel_x[b, ego, sl] = repl[:end_t - start_t, 3]
        vel_y[b, ego, sl] = repl[:end_t - start_t, 4]
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
    instruction_interval_steps: int,
    start_timestep: int,
    use_subgoal: bool,
    rng_key: jax.Array,
):
    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b1 = np.full([num_worlds], start_timestep, dtype=np.int32)
    instructions = [dict() for _ in range(num_worlds)]
    subgoals = [dict() for _ in range(num_worlds)]
    reward_functions = [dict() for _ in range(num_worlds)]
    target_lane_ids = [dict() for _ in range(num_worlds)]
    target_lane_xys = [dict() for _ in range(num_worlds)]

    replaced_state = sim_state  
    prev_instruction_texts = None
    for step_offset in tqdm(range(start_timestep, episode_num_steps - 1, replan_interval_steps)):
        rng_key, key_plan = jax.random.split(rng_key)
        instruction_texts = None if (step_offset - start_timestep) % instruction_interval_steps == 0 else prev_instruction_texts
        plan_result = planner.plan_trajectory(
            replaced_state,
            goal,
            rng=key_plan,
            timestep=step_offset,
            mask_goal=cfg.mask_goal,
            instruction_texts=instruction_texts,
            use_subgoal=cfg.use_subgoal,
        )
        instruction_t = getattr(plan_result, "instruction_texts", None)
        subgoal_t = getattr(plan_result, "subgoal_preds", None)
        reward_functions_t = getattr(plan_result, "reward_texts", None)
        target_lane_ids_t = getattr(plan_result, "target_lane_ids", None)
        target_lane_xys_t = [np.asarray(planner.scorers[i].target_lane) for i in range(num_worlds)]

        prev_instruction_texts = instruction_t
        for b in range(num_worlds):
            instructions[b][f"t{step_offset}"] = instruction_t[b] if instruction_t is not None else ""
            subgoals[b][f"t{step_offset}"] = subgoal_t[b] if subgoal_t is not None else None
            reward_functions[b][f"t{step_offset}"] = reward_functions_t[b] if reward_functions_t is not None else None
            target_lane_ids[b][f"t{step_offset}"] = target_lane_ids_t[b] if target_lane_ids_t is not None else None
            target_lane_xys[b][f"t{step_offset}"] = target_lane_xys_t[b] if target_lane_xys_t is not None else None

        plan_traj_bt5 = np.asarray(plan_result.trajectory_world_bt5, dtype=np.float32)
        replaced_state = apply_ego_replacements_to_expanded_state(
            replaced_state,
            start_t_b=step_offset,
            traj_bt5=plan_traj_bt5,
        )

    world_dt = float(np.asarray(plan_result.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)
    traj_bt5 = sim_state_to_ego_trajectory(replaced_state)

    pred = PredictionBatch(
        start_t_b=jax.numpy.asarray(start_t_b1, dtype=jnp.int32),
        trajectory_world_bt5=jnp.asarray(traj_bt5, dtype=jnp.float32),
        world_t_seconds_bt=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bt=jnp.asarray(world_valid),
        aux=plan_result.aux,
    )
    info = {
        "instructions": instructions,
        "subgoals": subgoals,
        "reward_functions": reward_functions,
        "target_lane_ids": target_lane_ids,
        "target_lane_xys": target_lane_xys,
    }

    return pred, replaced_state, info

def sim_state_to_ego_trajectory(sim_state):
    sdc_indices = get_sdc_indices_for_batched_state(sim_state)
    traj = sim_state.log_trajectory
    ego_traj_bt5 = jnp.stack([
        jnp.take_along_axis(traj.x, sdc_indices[:, None, None], axis=1).squeeze(axis=1),
        jnp.take_along_axis(traj.y, sdc_indices[:, None, None], axis=1).squeeze(axis=1),
        jnp.take_along_axis(traj.yaw, sdc_indices[:, None, None], axis=1).squeeze(axis=1),
        jnp.take_along_axis(traj.vel_x, sdc_indices[:, None, None], axis=1).squeeze(axis=1),
        jnp.take_along_axis(traj.vel_y, sdc_indices[:, None, None], axis=1).squeeze(axis=1),
    ], axis=-1)
    return ego_traj_bt5