
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

from model.diffusion_planner import DiffusionPlanner, PlannerResult
from model.diffusion_es_planner import DiffusionESPlanner, ESPlannerResult
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import LaneGraphShardStore
from train.infer import (
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from simulation_utils.utils import _parse_int_csv, _save_json  # noqa: E402
from viz.render import _load_scenario_state_batch_fast, render_videos_batched  # noqa: E402
from waymax import config as waymax_config
from tqdm import tqdm
from .utils import check_ego_reached_target_lane, check_traffic_light_violation, _predict_planner_trajectories_with_periodic_replan


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


def _lane_graph_zip_path_for_tfrecord(tfrecord_path: str, lane_graph_dir: str) -> Path:
    """Maps scenario tfrecord filename to lanegraph shard zip filename."""
    base = os.path.basename(tfrecord_path)
    return Path(lane_graph_dir) / f"{base}.lanegraph.zip"


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
    lane_graph_dir = getattr(args, "lane_graph_dir", None)
    lane_graph_store: LaneGraphShardStore | None = None
    lane_graph_zip_path: Path | None = None

    while tested_scenarios < int(args.num_scenarios):
        # scenario_indices = curr_scenario_index + np.arange(int(args.num_worlds))
        scenario_indices = [8]
        try:
            sim_state, _ = _load_scenario_state_batch_fast(ds_cfg, scenario_indices)
        except Exception as e:
            print(f"Scenarios in {os.path.basename(tfrecord_path)} are all tested, moving to next tfrecord.")
            tfrecord_path = tfrecord_paths.pop(0)
            curr_scenario_index = 0
            continue

        print(f"Testing scenarios {scenario_indices} from {os.path.basename(tfrecord_path)}...")

        rng_key = jax.random.PRNGKey(int(args.seed))
        goal_xy_b2, goal_t_b, ego_idx_b = _infer_goals_from_sim_state(sim_state)


        planner = DiffusionESPlanner(
            policy,
            inference.preprocess_cfg,
            population_size=int(args.population_size),
            resample_timesteps=int(args.resample_timesteps),
            num_worlds=len(scenario_indices),
            metrics=["collision", "offroad", "tl_violation"],
        )
        target_lanes = []
        if lane_graph_dir:
            requested_zip_path = _lane_graph_zip_path_for_tfrecord(str(tfrecord_path), lane_graph_dir)
            if lane_graph_zip_path != requested_zip_path:
                lane_graph_zip_path = requested_zip_path
                if lane_graph_zip_path.exists():
                    lane_graph_store = LaneGraphShardStore(lane_graph_zip_path.as_posix())
                else:
                    lane_graph_store = None
                    print(
                        f"[warn] lane graph shard zip not found for tfrecord: "
                        f"{lane_graph_zip_path.as_posix()}"
                    )
        targeted_indices = []
        lane_graphs = []
        for world_idx in range(len(scenario_indices)):
            if lane_graph_store is not None:
                graph = lane_graph_store.get_by_record_index(int(scenario_indices[world_idx]))
                if graph is not None:
                    planner.scorers[world_idx].set_lane_graph(graph)
                    lane_graphs.append(graph)
                else:
                    raise ValueError(f"Lane graph not found in shard for scenario index {scenario_indices[world_idx]} in tfrecord {tfrecord_path}.")
            scorer_helpers.update_lane_points(
                planner.scorers[world_idx], sim_state, world_idx
            )
            if args.task == "change_lane_right":
                target_lane = scorer_helpers.get_right_lane(
                    planner.scorers[world_idx],
                    sim_state,
                    timestep=0,
                    world_idx=world_idx,
                )
            elif args.task == "change_lane_left":
                target_lane = scorer_helpers.get_left_lane(
                    planner.scorers[world_idx],
                    sim_state,
                    timestep=0,
                    world_idx=world_idx,
                )
            else:
                target_lane = None
            target_lanes.append(target_lane)
            if target_lane is not None:
                planner.add_metric("follow_lane", target=target_lane, world_idx=world_idx, weight=1.0)
                targeted_indices.append(world_idx)
        print(f"Added follow_lane metric for worlds with indices: {targeted_indices}")
        
        pred, replaced_state = _predict_planner_trajectories_with_periodic_replan(
            sim_state,
            planner,
            goal_xy_b2,
            goal_t_b=goal_t_b,
            rng_key=rng_key,
            replan_interval_steps=int(args.replan_interval_steps),
            es_cfg=args,
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

        tl_violation_timeseries = check_traffic_light_violation(
            replaced_state,
            lane_graphs,
        )
        tl_violation = tl_violation_timeseries.sum(axis=1) > 0
        overlap = np.zeros((len(scenario_indices),), dtype=bool)
        offroad = np.zeros((len(scenario_indices),), dtype=bool)
        if "overlap" in rollout.metric_timeseries:
            overlap_timeseries = np.asarray(rollout.metric_timeseries["overlap"])
            overlap = overlap_timeseries.sum(axis=(1, 2)) < 0
        if "offroad" in rollout.metric_timeseries:
            offroad_timeseries = np.asarray(rollout.metric_timeseries["offroad"])
            offroad = offroad_timeseries.sum(axis=(1, 2)) < 0

        pred_traj = np.asarray(pred.trajectories_world_bkt5)
        start_t = np.asarray(pred.start_t_b, dtype=np.int32)

        target_exists = np.array([tl is not None for tl in target_lanes], dtype=bool)
        on_target = check_ego_reached_target_lane(pred_traj, target_lanes, world_indices=range(len(scenario_indices)))

        success = on_target & (~overlap) & (~offroad) & (~tl_violation)
        reached_all += int(np.sum(on_target))
        success_all += int(np.sum(success))

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
                "overlap": bool(overlap[i]),
                "offroad": bool(offroad[i]),
                "success": bool(success[i]),
                "reached": bool(on_target[i]),
                "target_exists": bool(target_exists[i]),
                "tl_violation": bool(tl_violation[i]),
                "video_path": video_paths[i].as_posix(),
                "tfrecord": str(tfrecord_path),
            }
            result_path = output_dir / f"{os.path.basename(tfrecord_path)}.scenario_{scenario_idx:03d}.json"
            _save_json(result_path, result)
            if args.save_traj and bool(success[i]):
                data_path = output_dir / "traj" / f"{Path(tfrecord_path).name}.scenario_{scenario_idx:03d}.npz"
                os.makedirs(data_path.parent, exist_ok=True)
                np.savez(
                    data_path,
                    trajectories_world_bkt5=pred_traj[i],
                )

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
        "goal_threshold_m": float(args.goal_threshold_m),
    }
    _save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
