from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diffusion_planner_v3 import (
    DiffusionPlanner,
    PlannerResult,
)
from train.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from train.preprocess import preprocess_simulator_state
from viz.render import _load_scenario_state_fast, render_videos_batched  # noqa: E402
from viz import viz as viz_module  # noqa: E402
import time
from test.utils import _parse_args, _planner_result_to_prediction_batch, _predict_planner_trajectories_with_periodic_replan, _save_json, _summarize_metric, _save_population_xy_score_image


def _planner_result_to_prediction_batch(result: PlannerResult) -> PredictionBatch:
    return PredictionBatch(
        start_t_b=result.start_t_b,
        trajectories_world_bkt5=result.trajectory_world_bt5[:, None, :, :],
        world_t_seconds_bk=result.world_t_seconds_bt,
        world_t_valid_bk=result.world_t_valid_bt,
        aux=result.aux,
    )


def _predict_planner_trajectories_with_periodic_replan(
    sim_state,
    planner: DiffusionPlanner,
    *,
    debug_label: str,
    output_dir: Path,
    scenario_index: int,
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

    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b1 = np.zeros((1,), dtype=np.int32)
    planner_debug: list[dict[str, Any]] = []

    rng_key, key_initial = jax.random.split(rng_key)
    init_pred_start = time.time()

    initial_current_lanes: list[np.ndarray] = []
    for world_idx in range(int(sim_state.log_trajectory.x.shape[0])):
        planner.scorers[world_idx].update_lane_points(sim_state, world_idx=world_idx)
        # planner.scorers[world_idx].update_objects(sim_state, timestep=0, world_idx=world_idx)
        lane_pts = planner.scorers[world_idx].get_right_lane(sim_state, timestep=0, world_idx=world_idx)
        if lane_pts is None:
            lane_pts = planner.scorers[world_idx].get_left_lane(sim_state, timestep=0, world_idx=world_idx)
        if lane_pts is None:
            lane_pts = planner.scorers[world_idx].get_current_lane(sim_state, timestep=0, world_idx=world_idx)
        initial_current_lanes.append(np.asarray(lane_pts, dtype=np.float32))
        planner.scorers[world_idx].add_metric("follow_lane", target=lane_pts)
        vehicle_front = planner.scorers[world_idx].get_vehicle_front(sim_state, timestep=0, world_idx=world_idx)
        print(vehicle_front)
    if vehicle_front is None:
        return None, None, None, None

    initial_result = planner.plan_trajectory(
        sim_state,
        rng=key_initial,
        elite_size=int(elite_size),
        num_iterations=int(num_iterations),
        timestep=0,
    )
    _save_population_xy_score_image(
        sim_state=sim_state,
        timestep=0,
        current_world_bkt5=np.asarray(initial_result.current_world_bkt5),
        current_scores_bk=np.asarray(initial_result.current_scores_bk),
        output_path=output_dir / "population_xy_scores" / f"scenario_{int(scenario_index):05d}" / f"step_{0:04d}.png",
        title=f"{debug_label} step=0 population xy (color=score)",
        front_x=float(front_x),
        back_x=float(back_x),
        front_y=float(front_y),
        back_y=float(back_y),
        current_lanes=initial_current_lanes,
    )
    init_pred_end = time.time()
    # print("initial_plan_time", init_pred_end - init_pred_start)

    initial_pred = _planner_result_to_prediction_batch(initial_result)
    initial_traj_bl5 = np.asarray(initial_result.trajectory_world_bt5, dtype=np.float32)
    model_horizon_len = int(initial_traj_bl5.shape[1])

    traj_b1l5 = np.zeros((1, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_b1l5[:, :init_copy_len, :] = initial_traj_bl5[:, :init_copy_len, :]
    current_score_b = np.asarray(initial_result.best_score_b, dtype=np.float32)
    planner_debug.append(
        {
            "step_offset": 0,
            "best_score_b": current_score_b.tolist(),
        }
    )

    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in range(interval, max_steps, interval):
        for world_idx in range(int(sim_state.log_trajectory.x.shape[0])):
            if planner.scorers[world_idx].on_target_lane(sim_state, timestep=step_offset, world_idx=world_idx):
                current_speed = planner.scorers[world_idx].get_current_speed(sim_state, timestep=step_offset, world_idx=world_idx)
                target_speed = 1.5 * float(current_speed)
                planner.scorers[world_idx].add_metric("set_speed", target=target_speed)

        replaced_state = _apply_ego_replacements_to_expanded_state(
            sim_state,
            start_t_bk=start_t_b1,
            traj_bkl5=traj_b1l5,
        )
        rng_key, key_replan = jax.random.split(rng_key)
        repl_start = time.time()

        repl_result = planner.plan_trajectory(
            replaced_state,
            rng=key_replan,
            elite_size=int(elite_size),
            num_iterations=int(num_iterations),
            timestep=int(step_offset),
        )
        _save_population_xy_score_image(
            sim_state=replaced_state,
            timestep=int(step_offset),
            current_world_bkt5=np.asarray(repl_result.current_world_bkt5),
            current_scores_bk=np.asarray(repl_result.current_scores_bk),
            output_path=output_dir / "population_xy_scores" / f"scenario_{int(scenario_index):05d}" / f"step_{int(step_offset):04d}.png",
            title=f"{debug_label} step={int(step_offset)} population xy (color=score)",
            front_x=float(front_x),
            back_x=float(back_x),
            front_y=float(front_y),
            back_y=float(back_y),
            current_lanes=initial_current_lanes,
        )
        repl_end = time.time()
        # print("replan_time",
        #     f"step_offset={int(step_offset)}",
        #     f"time={repl_end - repl_start:.3f}s",
        # )

        repl_score_b = np.asarray(repl_result.best_score_b, dtype=np.float32)
        repl_traj_b1l5 = np.asarray(repl_result.trajectory_world_bt5, dtype=np.float32)

        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            copy_len = min(remaining, int(repl_traj_b1l5.shape[1]))
            traj_b1l5[:, step_offset : step_offset + copy_len, :] = repl_traj_b1l5[:, :copy_len, :]

        current_score_b = repl_score_b
        planner_debug.append(
            {
                "step_offset": int(step_offset),
                "best_score_b": repl_score_b.tolist()
            }
        )

    world_dt = float(np.asarray(initial_pred.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((1, episode_num_steps), dtype=bool)

    pred = PredictionBatch(
        start_t_b=jnp.zeros((1,), dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_b1l5[:, None, :, :], dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial_pred.aux,
    )
    return pred, current_score_b[:, None], planner_debug, vehicle_front


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    planner = DiffusionPlanner(
        policy,
        inference.preprocess_cfg,
        population_size=int(args.population_size),
        resample_timesteps=int(args.resample_timesteps),
    )

    from waymax import config as waymax_config

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(args.tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )

    base_rng_key = jax.random.PRNGKey(int(args.seed))

    for scenario_index in range(int(args.num_episodes)):
        start_from = 0
        source_scenario_index = start_from + scenario_index
        try:
            sim_state = _load_scenario_state_fast(ds_cfg, int(source_scenario_index))
        except ValueError as exc:
            if "is out of range" in str(exc):
                break
            raise

        rng_key = jax.random.fold_in(base_rng_key, int(scenario_index))
        pred, final_scores, planner_debug, target_idx = _predict_planner_trajectories_with_periodic_replan(
            sim_state,
            planner,
            debug_label=f"scenario={scenario_index}",
            output_dir=output_dir,
            scenario_index=int(scenario_index),
            rng_key=rng_key,
            elite_size=int(args.elite_size),
            num_iterations=int(args.num_iterations),
            replan_interval_steps=int(args.replan_interval_steps),
            front_x=float(args.front_x),
            back_x=float(args.back_x),
            front_y=float(args.front_y),
            back_y=float(args.back_y),
        )

        pred_traj = np.asarray(pred.trajectories_world_bkt5)
        start_t = np.asarray(pred.start_t_b, dtype=np.int32)
        best_sample_index = np.zeros((pred_traj.shape[0],), dtype=np.int32)
        best_score = np.asarray(final_scores[:, 0], dtype=np.float32)


if __name__ == "__main__":
    main()
