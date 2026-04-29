import os
import json
import numpy as np
from pathlib import Path
from typing import Any
import math
import jax
import jax.numpy as jnp
from model.diffusion_planner import DiffusionPlanner, PlannerResult
from model.diffusion_es_planner import DiffusionESPlanner, ESPlannerResult
from train.infer import (
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from tqdm import tqdm


def _planner_result_to_prediction_batch(result: PlannerResult | ESPlannerResult) -> PredictionBatch:
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
    goal_xy: jax.Array,
    goal_t_b: jax.Array,
    *,
    rng_key: jax.Array,
    replan_interval_steps: int,
    es_cfg: Any,
) -> tuple[PredictionBatch, Any]:
    if int(replan_interval_steps) <= 0:
        raise ValueError(f"replan_interval_steps must be > 0, got {replan_interval_steps}.")

    num_worlds = int(sim_state.log_trajectory.x.shape[0])
    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b = np.zeros((num_worlds,), dtype=np.int32)

    rng_key, key_initial = jax.random.split(rng_key)
    initial_result = planner.plan_trajectory(
        sim_state,
        rng=key_initial, 
        timestep=0, 
        goal=goal_xy,
        goal_t=goal_t_b,
        mask_goal=es_cfg.mask_goal,
        elite_size=int(es_cfg.elite_size),
        num_iterations=int(es_cfg.num_iterations)
    )
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
            goal=goal_xy,
            goal_t=goal_t_b,
            mask_goal=es_cfg.mask_goal,
            elite_size=int(es_cfg.elite_size),
            num_iterations=int(es_cfg.num_iterations)
        )

        repl_traj_bl5 = np.asarray(repl_result.trajectory_world_bt5, dtype=np.float32)
        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            copy_len = min(remaining, int(repl_traj_bl5.shape[1]))
            traj_bl5[:, step_offset : step_offset + copy_len, :] = repl_traj_bl5[:, :copy_len, :]
    replaced_state = _apply_ego_replacements_to_expanded_state(
        sim_state,
        start_t_bk=start_t_b,
        traj_bkl5=traj_bl5,
    )
    world_dt = np.asarray(initial_pred.aux["world_dt_seconds"], dtype=np.float32).reshape(num_worlds)
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt[:, None]
    world_valid = np.ones((num_worlds, episode_num_steps), dtype=bool)

    return PredictionBatch(
        start_t_b=jnp.zeros((num_worlds,), dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_bl5[:, None, :, :], dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial_pred.aux,
    ), replaced_state



def check_ego_reached_target_lane(pred_traj, target_lanes, world_indices=None, dist_threshold=1.0, heading_threshold=np.pi/6):
    """
    pred_traj: np.ndarray, shape [B, 1, T, 5] or [B, T, 5] (final predicted trajectories)
    planner: planner object with .scorers[world_idx] supporting lane graph queries
    sim_state: simulation state (for scorer heading extraction)
    target_lanes: list of lane polylines (each [N, 4] or [N, >=4]), one per world
    world_indices: list of world indices (default: range(B))
    dist_threshold: float, max distance in meters
    heading_threshold: float, max heading difference in radians
    Returns: np.ndarray of bool, shape [B], True if reached
    """
    if pred_traj.ndim == 4:
        # [B, 1, T, 5] -> [B, T, 5]
        pred_traj = pred_traj[:, 0]
    B = pred_traj.shape[0]
    if world_indices is None:
        world_indices = list(range(B))
    reached = np.zeros((B,), dtype=bool)
    for i, world_idx in enumerate(world_indices):
        final_xy = pred_traj[i, -1, :2]  # [2]
        final_heading = pred_traj[i, -1, 2] if pred_traj.shape[-1] > 2 else None
        lane_poly = target_lanes[i]
        if lane_poly is None or lane_poly.shape[0] == 0:
            reached[i] = False
            continue
        lane_xy = np.asarray(lane_poly[:, :2], dtype=np.float32)  # [N,2]
        lane_dir = np.asarray(lane_poly[:, 2:4], dtype=np.float32)  # [N,2]
        # Find closest point on lane
        dists = np.linalg.norm(lane_xy - final_xy[None, :], axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        if min_dist > dist_threshold:
            reached[i] = False
            continue
        # Heading: get lane direction at closest point
        lane_vec = lane_dir[min_idx]
        lane_yaw = math.atan2(lane_vec[1], lane_vec[0])
        # Ego heading: if available from pred_traj, else estimate from last two points
        if final_heading is not None:
            ego_yaw = float(final_heading)
        else:
            if pred_traj.shape[1] >= 2:
                prev_xy = pred_traj[i, -2, :2]
                ego_vec = final_xy - prev_xy
                ego_yaw = math.atan2(ego_vec[1], ego_vec[0])
            else:
                reached[i] = False
                continue
        # Heading difference
        heading_diff = abs((ego_yaw - lane_yaw + np.pi) % (2 * np.pi) - np.pi)
        if heading_diff > heading_threshold:
            reached[i] = False
            continue
        reached[i] = True
    return reached


def _lane_start_pose_from_lane_graph(lane_graph, lane_id: int) -> np.ndarray | None:
    """Returns [x, y, heading] of a lane's start point from lane graph."""
    lane_range = lane_graph.lane_id_to_node_range.get(int(lane_id))
    if lane_range is None:
        return None
    start, end = int(lane_range[0]), int(lane_range[1])
    nodes_xyz = np.asarray(lane_graph.nodes_xyz, dtype=np.float32)
    if start < 0 or end > int(nodes_xyz.shape[0]):
        return None

    start_xy = nodes_xyz[start, :2]
    if end - start >= 2:
        next_xy = nodes_xyz[start + 1, :2]
        vec = next_xy - start_xy
    else:
        vec = np.asarray([1.0, 0.0], dtype=np.float32)

    if float(np.linalg.norm(vec)) <= 1e-6:
        heading = 0.0
    else:
        heading = float(math.atan2(float(vec[1]), float(vec[0])))
    return np.asarray([float(start_xy[0]), float(start_xy[1]), heading], dtype=np.float32)


def check_traffic_light_violation(
    sim_state,
    lane_graphs,
    dist_threshold: float = 2.0,
    stop_state: int = 4,
) -> np.ndarray:
    """Checks per-world red-light stopline violation from logged ego trajectory.

    Args:
        sim_state: Waymax simulation state with batched worlds.
        lane_graphs: List aligned with world index. Each entry should provide
            lane_id_to_node_range and nodes_xyz.
        dist_threshold: Max distance (m) to treat red-light lane start point as valid.
        stop_state: Traffic-light state integer for STOP/red (default=4).

    Returns:
        np.ndarray[bool] with shape [batch_size, horizon], True means violation detected.
    """

    is_sdc_bn = np.asarray(sim_state.object_metadata.is_sdc).astype(bool)
    ego_indices_b = np.argmax(is_sdc_bn.astype(np.int32), axis=1)
    
    obj_xy_bnt2 = np.asarray(sim_state.log_trajectory.xy, dtype=np.float32)

    tl_xy_bnt2 = np.asarray(sim_state.log_traffic_light.xy, dtype=np.float32)
    tl_state_blt = np.asarray(sim_state.log_traffic_light.state)
    tl_lane_ids_blt = np.asarray(sim_state.log_traffic_light.lane_ids)
    tl_valid_blt = np.asarray(sim_state.log_traffic_light.valid).astype(bool)
    batch_size = obj_xy_bnt2.shape[0]
    horizon = obj_xy_bnt2.shape[2]
    violation_bt = np.zeros((batch_size, horizon), dtype=bool)

    for world_idx in range(batch_size):
        lane_graph = lane_graphs[world_idx]
        if lane_graph is None:
            continue

        ego_idx = int(ego_indices_b[world_idx])
        ego_xy_t2 = obj_xy_bnt2[world_idx, ego_idx]

        tl_xy_lt2 = tl_xy_bnt2[world_idx]
        tl_state_lt = tl_state_blt[world_idx]
        tl_lane_ids_lt = tl_lane_ids_blt[world_idx]
        tl_valid_lt = tl_valid_blt[world_idx]

        if not np.any(tl_valid_lt):
            continue

        distance_ego2tl_lt = np.linalg.norm(ego_xy_t2[None, :, :] - tl_xy_lt2, axis=-1)
        distance_ego2tl_masked_lt = np.where(tl_valid_lt, distance_ego2tl_lt, np.inf)
        argmin_dist_ego2tl_t = np.argmin(distance_ego2tl_masked_lt, axis=0)
        step_idx_t = np.arange(horizon)
        valid_nearest_t = tl_valid_lt[argmin_dist_ego2tl_t, step_idx_t]
        target_tl_lane_ids_t = tl_lane_ids_lt[argmin_dist_ego2tl_t, step_idx_t]

        lane_pick_t = np.flatnonzero(valid_nearest_t & (target_tl_lane_ids_t > 0))
        if lane_pick_t.size == 0:
            continue

        target_tl_lane_start_pose = _lane_start_pose_from_lane_graph(
            lane_graph, int(target_tl_lane_ids_t[int(lane_pick_t[0])])
        )
        if target_tl_lane_start_pose is None:
            continue

        distance_ego2tllane_t = np.linalg.norm(ego_xy_t2 - target_tl_lane_start_pose[None, :2], axis=-1)
        
        red_mask_t = (
            valid_nearest_t
            & (tl_state_lt[argmin_dist_ego2tl_t, step_idx_t] == int(stop_state))
            & (distance_ego2tllane_t <= float(dist_threshold))
        )
        current_signed_t = (
            (ego_xy_t2[:-1] - target_tl_lane_start_pose[None, :2]) 
            @ np.array([np.cos(target_tl_lane_start_pose[2]), np.sin(target_tl_lane_start_pose[2])], dtype=np.float32)
        )
        next_signed_t = (
            (ego_xy_t2[1:] - target_tl_lane_start_pose[None, :2]) 
            @ np.array([np.cos(target_tl_lane_start_pose[2]), np.sin(target_tl_lane_start_pose[2])], dtype=np.float32)
        )
        crossed_red_t = red_mask_t[:-1] & (current_signed_t < 0.0) & (next_signed_t >= 0.0)
        violation_bt[world_idx, 1:horizon] = crossed_red_t

    return violation_bt

def _parse_int_csv(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in values]


def _summarize_metric(value_bkt: np.ndarray, valid_bkt: np.ndarray) -> dict[str, Any]:
    batch_size, num_samples, _ = value_bkt.shape
    per_entry: list[list[dict[str, Any]]] = []
    for b in range(batch_size):
        scenario_entries: list[dict[str, Any]] = []
        for k in range(num_samples):
            valid = valid_bkt[b, k].astype(bool)
            if np.any(valid):
                vals = value_bkt[b, k][valid]
                final = float(vals[-1])
                mean = float(vals.mean())
                max_value = float(vals.max())
                valid_steps = int(valid.sum())
            else:
                final = float("nan")
                mean = float("nan")
                max_value = float("nan")
                valid_steps = 0
            scenario_entries.append(
                {
                    "final": final,
                    "mean": mean,
                    "max": max_value,
                    "valid_steps": valid_steps,
                }
            )
        per_entry.append(scenario_entries)
    return {"per_scenario_sample": per_entry}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _lane_graph_zip_path_for_tfrecord(tfrecord_path: str, lane_graph_dir: str) -> Path:
    """Maps scenario tfrecord filename to lanegraph shard zip filename."""
    base = os.path.basename(tfrecord_path)
    return Path(lane_graph_dir) / f"{base}.lanegraph.zip"