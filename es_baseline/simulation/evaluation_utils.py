from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from lane_graph.lane_graph_utils import lane_start_pose_from_lane_graph
from simulation.planning_utils import PredictionBatch

_DEFAULT_METRIC_NAMES = ("overlap", "offroad")


def _build_rollout_env_and_actor(
    metric_names: tuple[str, ...], max_num_objects: int
):
    from waymax import agents
    from waymax import config as waymax_config
    from waymax import dynamics as waymax_dynamics
    from waymax import env as waymax_env

    state_dynamics = waymax_dynamics.StateDynamics()
    planning_dynamics = waymax_env.PlanningAgentDynamics(state_dynamics)
    env_cfg = waymax_config.EnvironmentConfig(
        max_num_objects=int(max_num_objects),
        controlled_object=waymax_config.ObjectType.SDC,
        metrics=waymax_config.MetricsConfig(metrics_to_run=metric_names),
    )
    env_obj = waymax_env.PlanningAgentEnvironment(
        dynamics_model=state_dynamics,
        config=env_cfg,
    )
    actor = agents.create_expert_actor(
        dynamics_model=planning_dynamics,
        is_controlled_func=lambda state: state.object_metadata.is_sdc,
    )
    return env_obj, actor

def _run_batched_rollout(replaced_state, env_obj, actor, rng_key, rollout_num_steps: int):
    from waymax import env as waymax_env

    return waymax_env.rollout(
        scenario=replaced_state,
        actor=actor,
        env=env_obj,
        rng=rng_key,
        rollout_num_steps=rollout_num_steps,
    )


def rollout_predicted_trajectories_with_metrics(
    sim_state,
    *,
    metric_names: tuple[str, ...] = _DEFAULT_METRIC_NAMES,
    rng_key: jax.Array | None = None,
):
    length = int(sim_state.log_trajectory.x.shape[-1])
    env_obj, actor = _build_rollout_env_and_actor(
        metric_names, int(sim_state.log_trajectory.x.shape[1])
    )
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    rollout_out = _run_batched_rollout(
        sim_state, env_obj, actor, key, length - 1
    )

    metric_timeseries: dict[str, jax.Array] = {}
    metric_valid: dict[str, jax.Array] = {}
    for metric_name in metric_names:
        if metric_name not in rollout_out.metrics:
            raise ValueError(
                f"Metric '{metric_name}' missing in rollout output; available={list(rollout_out.metrics.keys())}."
            )
        metric_result = rollout_out.metrics[metric_name]
        value_bt = jnp.transpose(metric_result.value, (0, 1))
        valid_bt = jnp.transpose(metric_result.valid, (0, 1))
        metric_timeseries[metric_name] = value_bt
        metric_valid[metric_name] = valid_bt

    return {
        "metric_timeseries": metric_timeseries,
        "metric_valid": metric_valid,
        "metric_names": metric_names,
    }


def infer_goals_from_sim_state(sim_state) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Infers one goal per world from ego's last valid log position."""
    x_bnt = jnp.asarray(sim_state.log_trajectory.x)
    y_bnt = jnp.asarray(sim_state.log_trajectory.y)
    valid_bnt = jnp.asarray(sim_state.log_trajectory.valid).astype(bool)
    is_sdc_bn = jnp.asarray(sim_state.object_metadata.is_sdc).astype(bool)

    if is_sdc_bn.ndim != 2:
        raise ValueError(f"Expected is_sdc shape [B,N], got {is_sdc_bn.shape}.")

    batch_size = int(is_sdc_bn.shape[0])
    ego_indices_b = jnp.argmax(is_sdc_bn.astype(jnp.int32), axis=1)

    goal_xy_b2 = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    goal_t_b = jnp.zeros((batch_size,), dtype=jnp.int32)
    for b in range(batch_size):
        if int(jnp.sum(is_sdc_bn[b])) != 1:
            raise ValueError(
                f"Expected exactly one SDC in world {b}, got {int(jnp.sum(is_sdc_bn[b]))}."
            )
        ego_idx = int(ego_indices_b[b])
        valid_t = jnp.flatnonzero(valid_bnt[b, ego_idx])
        t_goal = int(valid_t[-1]) if valid_t.size > 0 else 0
        goal_t_b = goal_t_b.at[b].set(jnp.int32(t_goal))
        goal_xy_b2 = goal_xy_b2.at[b].set(
            jnp.array([x_bnt[b, ego_idx, t_goal], y_bnt[b, ego_idx, t_goal]], dtype=jnp.float32)
        )

    return goal_xy_b2, goal_t_b, ego_indices_b.astype(jnp.int32)


def check_goal_reaching(
    sim_state,
    goal_xy_b2: jnp.ndarray,
    start_timestep: int = 10,
    *,
    goal_threshold_m: float = 2.0,
) -> dict[str, jnp.ndarray]:
    if jnp.asarray(goal_xy_b2).ndim != 2:
        raise ValueError(f"Expected goal_xy shape [B,2], got {jnp.asarray(goal_xy_b2).shape}.")

    is_sdc_bn = jnp.asarray(sim_state.object_metadata.is_sdc).astype(bool)
    ego_indices_b = jnp.argmax(is_sdc_bn.astype(jnp.int32), axis=1)

    obj_xy_bnt2 = jnp.asarray(sim_state.log_trajectory.xy, dtype=jnp.float32)
    ego_xy_bt2 = jnp.take_along_axis(
        obj_xy_bnt2, ego_indices_b[:, None, None, None], axis=1
    ).squeeze(axis=1)

    dists_bt = jnp.linalg.norm(ego_xy_bt2[:, start_timestep:] - goal_xy_b2[:, None, :], axis=-1)

    min_dist_b = jnp.min(dists_bt, axis=1)
    final_dist_b = dists_bt[:, -1]
    mask_bl = dists_bt <= float(goal_threshold_m)

    reached_b = jnp.any(mask_bl, axis=1)
    # argmax returns 0 if no True values; guard with reached_b to set -1 when not reached
    first_hit_b = jnp.argmax(mask_bl, axis=1).astype(jnp.int32) + start_timestep
    reached_step_b = jnp.where(reached_b, first_hit_b, -1).astype(jnp.int32)

    return {
        "reached": reached_b,
        "reached_step": reached_step_b,
        "min_goal_distance_m": min_dist_b,
        "final_goal_distance_m": final_dist_b,
    }



def check_traffic_light_violation(
    sim_state,
    lane_graphs,
    dist_threshold: float = 2.0,
    stop_state: int = 4,
) -> dict[str, jnp.ndarray]:
    """Checks per-world red-light stopline violation from logged ego trajectory.

    Args:
        sim_state: Waymax simulation state with batched worlds.
        lane_graphs: List aligned with world index. Each entry should provide
            lane_id_to_node_range and nodes_xyz.
        dist_threshold: Max distance (m) to treat red-light lane start point as valid.
        stop_state: Traffic-light state integer for STOP/red (default=4).

    Returns:
        Dict with keys:
        - "violation": bool array [batch_size, horizon], True means violation detected.
        - "violation_step": int32 array [batch_size], first violation step per world, or -1.
    """

    is_sdc_bn = jnp.asarray(sim_state.object_metadata.is_sdc).astype(bool)
    ego_indices_b = jnp.argmax(is_sdc_bn.astype(jnp.int32), axis=1)

    obj_xy_bnt2 = jnp.asarray(sim_state.log_trajectory.xy, dtype=jnp.float32)

    tl_xy_bnt2 = jnp.asarray(sim_state.log_traffic_light.xy, dtype=jnp.float32)
    tl_state_blt = jnp.asarray(sim_state.log_traffic_light.state)
    tl_lane_ids_blt = jnp.asarray(sim_state.log_traffic_light.lane_ids)
    tl_valid_blt = jnp.asarray(sim_state.log_traffic_light.valid).astype(bool)
    batch_size = int(obj_xy_bnt2.shape[0])
    horizon = int(obj_xy_bnt2.shape[2])
    violation_bt = jnp.zeros((batch_size, horizon), dtype=bool)

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

        if not bool(jnp.any(tl_valid_lt)):
            continue

        distance_ego2tl_lt = jnp.linalg.norm(ego_xy_t2[None, :, :] - tl_xy_lt2, axis=-1)
        distance_ego2tl_masked_lt = jnp.where(tl_valid_lt, distance_ego2tl_lt, jnp.inf)
        argmin_dist_ego2tl_t = jnp.argmin(distance_ego2tl_masked_lt, axis=0)
        step_idx_t = jnp.arange(horizon)
        valid_nearest_t = tl_valid_lt[argmin_dist_ego2tl_t, step_idx_t]
        target_tl_lane_ids_t = tl_lane_ids_lt[argmin_dist_ego2tl_t, step_idx_t]

        lane_pick_t = jnp.flatnonzero(valid_nearest_t & (target_tl_lane_ids_t > 0))
        if lane_pick_t.size == 0:
            continue

        target_tl_lane_start_pose = lane_start_pose_from_lane_graph(
            lane_graph, int(target_tl_lane_ids_t[int(lane_pick_t[0])])
        )
        if target_tl_lane_start_pose is None:
            continue

        target_tl_lane_start_pose = jnp.asarray(target_tl_lane_start_pose, dtype=jnp.float32)
        lane_dir_t = jnp.array(
            [jnp.cos(target_tl_lane_start_pose[2]), jnp.sin(target_tl_lane_start_pose[2])],
            dtype=jnp.float32,
        )

        distance_ego2tllane_t = jnp.linalg.norm(ego_xy_t2 - target_tl_lane_start_pose[None, :2], axis=-1)

        red_mask_t = (
            valid_nearest_t
            & (tl_state_lt[argmin_dist_ego2tl_t, step_idx_t] == int(stop_state))
            & (distance_ego2tllane_t <= float(dist_threshold))
        )
        current_signed_t = jnp.sum((ego_xy_t2[:-1] - target_tl_lane_start_pose[None, :2]) * lane_dir_t[None, :], axis=-1)
        next_signed_t = jnp.sum((ego_xy_t2[1:] - target_tl_lane_start_pose[None, :2]) * lane_dir_t[None, :], axis=-1)
        crossed_red_t = red_mask_t[:-1] & (current_signed_t < 0.0) & (next_signed_t >= 0.0)
        violation_bt = violation_bt.at[world_idx, 1:horizon].set(crossed_red_t)
    violation_b = jnp.any(violation_bt, axis=1)

    violation_step_b = jnp.full((batch_size,), -1, dtype=jnp.int32)
    for world_idx in range(batch_size):
        hit_steps = jnp.flatnonzero(violation_bt[world_idx])
        if hit_steps.size > 0:
            violation_step_b = violation_step_b.at[world_idx].set(jnp.int32(hit_steps[0]))

    return {
        "violation": violation_b,
        "violation_step": violation_step_b,
    }