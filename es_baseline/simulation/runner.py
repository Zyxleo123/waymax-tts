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

from data.scenario_loader import load_scenario_state_batch_fast
from lane_graph.lane_graph_utils import LaneGraphLoader, lane_graph_zip_path_for_tfrecord
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import LaneGraphShardStore
from simulation.evaluation_utils import (
    rollout_predicted_trajectories_with_metrics,
    infer_goals_from_sim_state, 
    check_goal_reaching, 
    check_traffic_light_violation
)
from simulation.planning_utils import (
    predict_planner_trajectories_with_periodic_replan,
    get_sdc_indices_for_batched_state,
    sim_state_to_ego_trajectory,
)
from planner.abstract_planner import AbstractPlanner, PlannerResult

from simulation.utils import _save_json  # noqa: E402
from viz.render import render_videos_batched
from waymax import config as waymax_config
from tqdm import tqdm


def run(args, planner: AbstractPlanner, log_replay=False) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    tested_scenarios, curr_scenario_index = 0, 0
    reached_all, success_all = 0, 0
    tfrecord_paths = sorted(glob(str(Path(args.tfrecord_dir) / "*")))
    tfrecord_path = tfrecord_paths.pop(0)
    # lane_graph_loader = LaneGraphLoader(args.lane_graph_dir)
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(args.num_worlds,),
        shuffle_seed=0,
    )
    if args.scenario_indices is not None:
        target_scenario_indices = [int(idx) for idx in args.scenario_indices.split(",")]
    else:
        target_scenario_indices = None

    lane_graph_zip_path = None
    while tested_scenarios < int(args.num_scenarios):
        if target_scenario_indices is not None:
            scenario_indices = target_scenario_indices[tested_scenarios : tested_scenarios + int(args.num_worlds)]
            if not scenario_indices:
                break
        else:
            scenario_indices = curr_scenario_index + np.arange(int(args.num_worlds))
        try:
            sim_state, _ = load_scenario_state_batch_fast(ds_cfg, scenario_indices)
        except Exception as e:
            print(f"Error loading scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}: {e}")
            print(f"Scenarios in {os.path.basename(tfrecord_path)} are all tested, moving to next tfrecord.")
            tfrecord_path = tfrecord_paths.pop(0)
            curr_scenario_index = 0
            ds_cfg = dataclasses.replace(
                waymax_config.WOD_1_3_1_TRAINING,
                path=str(tfrecord_path),
                max_num_objects=int(args.max_num_objects),
                batch_dims=(args.num_worlds,),
                shuffle_seed=0,
            )
            continue

        print(f"Testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}...")

        # lane_graphs = lane_graph_loader.get_lane_graph_for_scenarios(tfrecord_path, scenario_indices)
        rng_key = jax.random.PRNGKey(int(args.seed))
        goal_xy_b2, goal_t_b, ego_idx_b = infer_goals_from_sim_state(sim_state)

        if log_replay:
            replaced_state = sim_state
            pred_traj = np.asarray(sim_state_to_ego_trajectory(sim_state))
            start_t = np.full((len(scenario_indices),), int(args.start_timestep), dtype=np.int32)
            instructions = [dict() for _ in range(len(scenario_indices))]
            subgoals = [dict() for _ in range(len(scenario_indices))]
            reward_functions = [dict() for _ in range(len(scenario_indices))]
            target_lane_ids = [None for _ in range(len(scenario_indices))]
            target_lane_xys = [None for _ in range(len(scenario_indices))]
        else:
            if args.planner == "es" or args.planner == "temporal-vla":
                if args.lane_graph_dir:
                    requested_zip_path = lane_graph_zip_path_for_tfrecord(str(tfrecord_path), args.lane_graph_dir)
                    if lane_graph_zip_path != requested_zip_path:
                        lane_graph_zip_path = requested_zip_path
                        if lane_graph_zip_path.exists():
                            lane_graph_store = LaneGraphShardStore(lane_graph_zip_path.as_posix())
                        else:
                            lane_graph_store = None
                            raise FileNotFoundError(f"Lane graph zip file {lane_graph_zip_path} not found for tfrecord {tfrecord_path}.")
                lane_graphs = []
                for world_idx in range(len(scenario_indices)):
                    graph = lane_graph_store.get_by_record_index(int(scenario_indices[world_idx]))
                    planner.scorers[world_idx].set_lane_graph(graph)
                    lane_graphs.append(graph)
                    scorer_helpers.update_lane_points(
                        planner.scorers[world_idx], sim_state, world_idx
                    )
            # SAC / instrumented ES: map world_idx → scenario index for offline banks.
            if hasattr(planner, "batch_scenario_indices"):
                planner.batch_scenario_indices = [int(i) for i in scenario_indices]
            pred, replaced_state, info = predict_planner_trajectories_with_periodic_replan(
                args,
                sim_state,
                goal_xy_b2,
                planner,
                start_timestep=int(args.start_timestep),
                replan_interval_steps=int(args.replan_interval_steps),
                instruction_interval_steps=int(args.instruction_interval_steps),
                use_subgoal=args.use_subgoal,
                rng_key=rng_key,
            )
            pred_traj = np.asarray(pred.trajectory_world_bt5)
            start_t = np.asarray(pred.start_t_b, dtype=np.int32)
            instructions = info["instructions"]
            subgoals = info["subgoals"]
            reward_functions = info["reward_functions"]
            target_lane_ids = info["target_lane_ids"]
            

        rollout = rollout_predicted_trajectories_with_metrics(
            replaced_state,
            rng_key=rng_key,
        )
        # tl_violation_eval = check_traffic_light_violation(
        #     replaced_state,
        #     lane_graphs,
        # )
        # tl_violation = np.asarray(tl_violation_eval["violation"])
        # tl_violation_step = np.asarray(tl_violation_eval["violation_step"])
        goal_eval = check_goal_reaching(
            replaced_state,
            goal_xy_b2,
            start_timestep=int(args.start_timestep),
        )
        goal_reached = np.asarray(goal_eval["reached"])
        goal_reached_step = np.asarray(goal_eval["reached_step"])

        overlap_timeseries = np.asarray(rollout["metric_timeseries"]["overlap"][int(args.start_timestep):])
        offroad_timeseries = np.asarray(rollout["metric_timeseries"]["offroad"][int(args.start_timestep):])

        overlap = np.zeros((len(scenario_indices),), dtype=bool)
        offroad = np.zeros((len(scenario_indices),), dtype=bool)
        for world_idx in range(len(scenario_indices)):
            episode_length = goal_reached_step[world_idx] + 1 if goal_reached[world_idx] else overlap_timeseries.shape[0]
            if "overlap" in rollout["metric_timeseries"]:
                overlap[world_idx] = overlap_timeseries[:episode_length, world_idx].sum() > 0
            if "offroad" in rollout["metric_timeseries"]:
                offroad[world_idx] = offroad_timeseries[:episode_length, world_idx].sum() > 0
        # tl_violation = tl_violation & (tl_violation_step < goal_reached_step)

        success = goal_reached & (~overlap) & (~offroad)
        reached_all += int(np.sum(goal_eval["reached"]))
        success_all += int(np.sum(success))

        if args.save_trajectory:
            for i, scenario_idx in enumerate(scenario_indices):
                traj_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "predicted_trajectory": pred_traj[i],
                    "start_timestep": int(start_t[i]),
                    "goal_xy": np.asarray(goal_xy_b2[i]),
                }
                traj_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.trajectory.npz"
                np.savez(traj_path, **traj_data)
        if args.save_instruction:
            for i, scenario_idx in enumerate(scenario_indices):
                instruction_data = {
                    "scenario_idx": int(scenario_idx),
                    "ego_idx": int(ego_idx_b[i]),
                    "instructions": instructions[i],
                    "subgoals": subgoals[i] if args.use_subgoal else None,
                    "reward_functions": reward_functions[i],
                    "target_lane_ids": target_lane_ids[i]
                }
                instruction_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.instructions.json"
                _save_json(instruction_path, instruction_data)

        if args.visualize_mode == "all":
            visualize_indices = np.arange(len(scenario_indices))
            video_requests = [
                (str(tfrecord_path), int(idx)) for idx in scenario_indices
            ]
        elif args.visualize_mode == "success":
            visualize_indices = np.where(success)[0]
            video_requests = [
                (str(tfrecord_path), int(scenario_indices[i]))
                for i in range(len(scenario_indices)) if success[i]
            ]
        elif args.visualize_mode == "failure":
            visualize_indices = np.where(~success)[0]
            video_requests = [
                (str(tfrecord_path), int(scenario_indices[i]))
                for i in range(len(scenario_indices)) if not success[i]
            ]
        else:
            visualize_indices = []
        video_requests = [
            (str(tfrecord_path), int(scenario_indices[i]))
            for i in visualize_indices
        ]
        ego_start_times = [0 for i in visualize_indices]
        ego_trajectories = [np.asarray(pred_traj[i]) for i in visualize_indices]
        # ego_indices = get_sdc_indices_for_batched_state(replaced_state)
        # ego_trajectories = [
        #     np.asarray(
        #         jnp.stack(
        #             [
        #                 replaced_state.log_trajectory.x[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.y[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.yaw[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.vel_x[i, int(ego_indices[i]), :],
        #                 replaced_state.log_trajectory.vel_y[i, int(ego_indices[i]), :],
        #             ], axis=-1
        #         )
        #     ) for i in visualize_indices
        # ]
        goal_xys = [np.asarray(goal_xy_b2[i]) for i in visualize_indices]
        if video_requests:
            video_paths = render_videos_batched(
                tfrecord_scenarios=video_requests,
                target_vehicles=[None] * len(scenario_indices),
                output_dir=output_dir.as_posix(),
                ego_start_times=ego_start_times,
                ego_trajectories=ego_trajectories,
                goal_xy=goal_xys,
            )
        for i, scenario_idx in enumerate(scenario_indices):
            result = {
                "scenario_idx": int(scenario_idx),
                "ego_idx": int(ego_idx_b[i]),
                "goal_xy": [float(goal_xy_b2[i, 0]), float(goal_xy_b2[i, 1])],
                "goal_timestep": int(goal_t_b[i]),
                "goal_reached": bool(goal_eval["reached"][i]),
                "reached_timestep": int(goal_eval["reached_step"][i]),
                "min_goal_distance_m": float(goal_eval["min_goal_distance_m"][i]),
                "final_goal_distance_m": float(goal_eval["final_goal_distance_m"][i]),
                "overlap": bool(overlap[i]),
                "offroad": bool(offroad[i]),
                "success": bool(success[i]),
                "tfrecord": str(tfrecord_path),
            }
            result_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.json"
            _save_json(result_path, result)
        tested_scenarios += len(scenario_indices)
        curr_scenario_index += len(scenario_indices)
        print(f"Finished testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}.")
        print(f"Current success rate: {success_all}/{tested_scenarios} = {success_all / tested_scenarios:.4f}")
        print("" + "-" * 50)

    summary = {
        "num_scenarios": tested_scenarios,
        "num_goal_reached": int(reached_all),
        "num_success": int(success_all),
        "goal_reach_rate": float(reached_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
        "success_rate": float(success_all / tested_scenarios) if tested_scenarios > 0 else 0.0,
    }
    _save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
