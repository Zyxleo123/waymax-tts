from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from glob import glob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diffusion_planner import DiffusionPlanner, PlannerResult  # noqa: E402
from train.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from simulation_utils.utils import _parse_int_csv, _save_json  # noqa: E402
from viz.render import _load_scenario_state_batch_fast, render_videos_batched  # noqa: E402
from waymax import config as waymax_config
from tqdm import tqdm


def _planner_result_to_prediction_batch(result: PlannerResult) -> PredictionBatch:
    return PredictionBatch(
        start_t_b=result.start_t_b,
        trajectories_world_bkt5=result.trajectory_world_bt5[:, None, :, :],
        world_t_seconds_bk=result.world_t_seconds_bt,
        world_t_valid_bk=result.world_t_valid_bt,
        aux=result.aux,
    )


def _infer_goals_from_sim_state(sim_state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infers one goal per world from ego's last valid log position."""
    x_bnt = np.asarray(sim_state.log_trajectory.x)
    y_bnt = np.asarray(sim_state.log_trajectory.y)
    valid_bnt = np.asarray(sim_state.log_trajectory.valid).astype(bool)
    is_sdc_bn = np.asarray(sim_state.object_metadata.is_sdc).astype(bool)

    if is_sdc_bn.ndim != 2:
        raise ValueError(f"Expected is_sdc shape [B,N], got {is_sdc_bn.shape}.")

    batch_size = int(is_sdc_bn.shape[0])
    ego_indices_b = np.argmax(is_sdc_bn.astype(np.int32), axis=1)

    goal_xy_b2 = np.zeros((batch_size, 2), dtype=np.float32)
    goal_t_b = np.zeros((batch_size,), dtype=np.int32)
    for b in range(batch_size):
        if int(np.sum(is_sdc_bn[b])) != 1:
            raise ValueError(
                f"Expected exactly one SDC in world {b}, got {int(np.sum(is_sdc_bn[b]))}."
            )
        ego_idx = int(ego_indices_b[b])
        valid_t = np.flatnonzero(valid_bnt[b, ego_idx])
        t_goal = int(valid_t[-1]) if valid_t.size > 0 else 0
        goal_t_b[b] = t_goal
        goal_xy_b2[b, 0] = float(x_bnt[b, ego_idx, t_goal])
        goal_xy_b2[b, 1] = float(y_bnt[b, ego_idx, t_goal])

    return goal_xy_b2, goal_t_b, ego_indices_b.astype(np.int32)


def _predict_planner_trajectories_with_periodic_replan(
    sim_state,
    planner: DiffusionPlanner,
    goal_xy: jax.Array,
    *,
    rng_key: jax.Array,
    replan_interval_steps: int,
) -> PredictionBatch:
    if int(replan_interval_steps) <= 0:
        raise ValueError(f"replan_interval_steps must be > 0, got {replan_interval_steps}.")

    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b = np.zeros((num_worlds,), dtype=np.int32)

    rng_key, key_initial = jax.random.split(rng_key)
    initial_result = planner.plan_trajectory(sim_state, rng=key_initial, timestep=0, goal=goal_xy)
    initial_pred = _planner_result_to_prediction_batch(initial_result)

    initial_traj_bl5 = np.asarray(initial_result.trajectory_world_bt5, dtype=np.float32)
    model_horizon_len = int(initial_traj_bl5.shape[1])

    traj_bl5 = np.zeros((num_worlds, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_bl5[:, :init_copy_len, :] = initial_traj_bl5[:, :init_copy_len, :]

    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in tqdm(range(interval, max_steps, interval)):
        replaced_state = _apply_ego_replacements_to_expanded_state(
            sim_state,
            start_t_bk=start_t_b,
            traj_bkl5=traj_bl5,
        )
        rng_key, key_replan = jax.random.split(rng_key)
        repl_result = planner.plan_trajectory(
            replaced_state,
            rng=key_replan,
            timestep=int(step_offset),
            goal=goal_xy
        )

        repl_traj_bl5 = np.asarray(repl_result.trajectory_world_bt5, dtype=np.float32)
        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            copy_len = min(remaining, int(repl_traj_bl5.shape[1]))
            traj_bl5[:, step_offset : step_offset + copy_len, :] = repl_traj_bl5[:, :copy_len, :]

    world_dt = np.asarray(initial_pred.aux["world_dt_seconds"], dtype=np.float32).reshape(num_worlds)
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt[:, None]
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)

    return PredictionBatch(
        start_t_b=jnp.zeros((num_worlds,), dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_bl5[:, None, :, :], dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial_pred.aux,
    )


def _evaluate_goal_reaching(
    pred: PredictionBatch,
    goal_xy_b2: np.ndarray,
    *,
    goal_threshold_m: float,
) -> dict[str, np.ndarray]:
    traj_blt2 = np.asarray(pred.trajectories_world_bkt5[:, 0, :, :2], dtype=np.float32)
    dists_blt = np.linalg.norm(traj_blt2 - goal_xy_b2[:, None, :], axis=-1)

    min_dist_b = np.min(dists_blt, axis=1)
    final_dist_b = dists_blt[:, -1]
    reached_b = min_dist_b <= float(goal_threshold_m)

    reached_step_b = np.full((traj_blt2.shape[0],), -1, dtype=np.int32)
    for b in range(traj_blt2.shape[0]):
        hit_steps = np.flatnonzero(dists_blt[b] <= float(goal_threshold_m))
        if hit_steps.size > 0:
            reached_step_b[b] = int(hit_steps[0])

    return {
        "reached": reached_b,
        "reached_step": reached_step_b,
        "min_goal_distance_m": min_dist_b,
        "final_goal_distance_m": final_dist_b,
    }


def run(args) -> list[dict[str, Any]]:
    inference = load_model_for_inference(
        args.checkpoint_path,
        use_ema=bool(args.use_ema),
        metadata_path=args.metadata_path,
        seed=int(args.seed),
        skip_checkpoint_load=bool(args.skip_checkpoint_load),
    )
    policy = nnx.merge(
        inference.graphdef,
        inference.params_state,
        inference.nonparam_state,
    )

    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    tested_scenarios, curr_scenario_index = 0, 0
    reached_all, success_all = 0, 0
    tfrecord_paths = sorted(glob(str(Path(args.tfrecord_dir) / "*")))
    tfrecord_path = tfrecord_paths.pop(0)

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(args.num_worlds,),
        shuffle_seed=0,
    )
    while tested_scenarios < int(args.num_scenarios):
        scenario_indices = curr_scenario_index + np.arange(int(args.num_worlds))
        try:
            sim_state, _ = _load_scenario_state_batch_fast(ds_cfg, scenario_indices)
        except Exception as e:
            print(f"Scenarios in {os.path.basename(tfrecord_path)} are all tested, moving to next tfrecord.")
            tfrecord_path = tfrecord_paths.pop(0)
            curr_scenario_index = 0
            continue

        print(f"Testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}...")

        planner = DiffusionPlanner(
            policy,
            inference.preprocess_cfg,
            population_size=int(args.population_size),
            num_worlds=len(scenario_indices),
        )

        rng_key = jax.random.PRNGKey(int(args.seed))
        goal_xy_b2, goal_t_b, ego_idx_b = _infer_goals_from_sim_state(sim_state)
        pred = _predict_planner_trajectories_with_periodic_replan(
            sim_state,
            planner,
            goal_xy_b2,
            rng_key=rng_key,
            replan_interval_steps=int(args.replan_interval_steps),
        )

        
        goal_eval = _evaluate_goal_reaching(
            pred,
            goal_xy_b2,
            goal_threshold_m=float(args.goal_threshold_m),
        )
        

        metric_names = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
        rollout = rollout_predicted_trajectories_with_metrics(
            sim_state,
            pred,
            metric_names=metric_names,
            rng_key=rng_key,
        )

        overlap = np.zeros((len(scenario_indices),), dtype=bool)
        offroad = np.zeros((len(scenario_indices),), dtype=bool)
        if "overlap" in rollout.metric_timeseries:
            overlap_timeseries = np.asarray(rollout.metric_timeseries["overlap"])
            overlap = overlap_timeseries.sum(axis=(1, 2)) < 0
        if "offroad" in rollout.metric_timeseries:
            offroad_timeseries = np.asarray(rollout.metric_timeseries["offroad"])
            offroad = offroad_timeseries.sum(axis=(1, 2)) < 0

        success = goal_eval["reached"] & (~overlap) & (~offroad)
        reached_all += int(np.sum(goal_eval["reached"]))
        success_all += int(np.sum(success))
        pred_traj = np.asarray(pred.trajectories_world_bkt5)
        start_t = np.asarray(pred.start_t_b, dtype=np.int32)

        video_requests = [
            (str(tfrecord_path), int(idx)) for idx in scenario_indices
        ]
        video_paths = render_videos_batched(
            tfrecord_scenarios=video_requests,
            target_vehicles=[None] * len(scenario_indices),
            output_dir=output_dir.as_posix(),
            fps=int(args.fps),
            num_frames=int(args.num_frames),
            max_num_objects=int(args.max_num_objects),
            width=int(args.width),
            height=int(args.height),
            px_per_meter=float(args.px_per_meter),
            show_agent_id=bool(args.show_agent_id),
            use_log_traj=True,
            ego_start_times=[int(start_t[i]) for i in range(len(scenario_indices))],
            ego_trajectories=[pred_traj[i, 0] for i in range(len(scenario_indices))],
            front_x=float(args.front_x),
            back_x=float(args.back_x),
            front_y=float(args.front_y),
            back_y=float(args.back_y),
            align_ego_heading_up=True,
            goal_xy=np.asarray(goal_xy_b2, dtype=np.float32),
        )
        for i, scenario_idx in enumerate(scenario_indices):
            result = {
                "scenario_idx": int(scenario_idx),
                "ego_idx": int(ego_idx_b[i]),
                "goal_xy": [float(goal_xy_b2[i, 0]), float(goal_xy_b2[i, 1])],
                "goal_timestep": int(goal_t_b[i]),
                "goal_threshold_m": float(args.goal_threshold_m),
                "goal_reached": bool(goal_eval["reached"][i]),
                "reached_timestep": int(goal_eval["reached_step"][i]),
                "min_goal_distance_m": float(goal_eval["min_goal_distance_m"][i]),
                "final_goal_distance_m": float(goal_eval["final_goal_distance_m"][i]),
                "overlap": bool(overlap[i]),
                "offroad": bool(offroad[i]),
                "success": bool(success[i]),
                "video_path": video_paths[i].as_posix(),
                "tfrecord": str(tfrecord_path),
            }
            result_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.json"
            _save_json(result_path, result)
        tested_scenarios += len(scenario_indices)
        curr_scenario_index += len(scenario_indices)

    summary = {
        "num_scenarios": tested_scenarios,
        "num_goal_reached": int(reached_all),
        "num_success": int(success_all),
        "goal_reach_rate": float(reached_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
        "success_rate": float(success_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
        "goal_threshold_m": float(args.goal_threshold_m),
    }
    _save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
