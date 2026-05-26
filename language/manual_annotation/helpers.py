from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import jax

from waymax.datatypes.roadgraph import MapElementIds
from waymax.datatypes.object_state import ObjectTypeIds
from waymax.datatypes.traffic_lights import TrafficLightStates


CENTERLINE_TYPES = {
    int(MapElementIds.LANE_FREEWAY.value),
    int(MapElementIds.LANE_SURFACE_STREET.value),
}
TRAFFIC_LIGHT_STATE_DICT = {
    int(TrafficLightStates.UNKNOWN): "unknown",
    int(TrafficLightStates.ARROW_STOP): "arrow_stop",
    int(TrafficLightStates.ARROW_CAUTION): "arrow_caution",
    int(TrafficLightStates.ARROW_GO): "arrow_go",
    int(TrafficLightStates.STOP): "stop",
    int(TrafficLightStates.CAUTION): "caution",
    int(TrafficLightStates.GO): "go",
    int(TrafficLightStates.FLASHING_STOP): "flashing_stop",
    int(TrafficLightStates.FLASHING_CAUTION): "flashing_caution",
}
OBJECT_TYPE_DICT = {
    int(ObjectTypeIds.UNSET.value): "unknown",
    int(ObjectTypeIds.VEHICLE.value): "vehicle",
    int(ObjectTypeIds.PEDESTRIAN.value): "pedestrian",
    int(ObjectTypeIds.CYCLIST.value): "cyclist",
    int(ObjectTypeIds.OTHER.value): "other",
}


def _isin(values, test_elements):
    return jnp.isin(values, jnp.asarray(test_elements))


def get_ego_idx(sim_state, world_idx=0):
    ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
    if int(jnp.sum(ego_mask)) != 1:
        raise ValueError(
            f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}."
        )
    return int(jnp.argmax(ego_mask.astype(jnp.int32)))

def get_current_lane_id(
        sim_state,
        timestep,
        world_idx=0, 
        dist_threshold=5.0,
        yaw_threshold=jnp.pi / 4,
    ):
    lane_ids = sim_state.roadgraph_points.ids[world_idx]
    lane_types = sim_state.roadgraph_points.types[world_idx]
    lane_mask = _isin(lane_types, tuple(CENTERLINE_TYPES))
    lane_mask = jnp.logical_and(lane_mask, lane_ids >= 0)
    lane_ids = lane_ids[lane_mask]
    assert sim_state.log_trajectory.x.shape[2] == 91, f"Expected 91 timesteps, got {sim_state.log_trajectory.x.shape[2]}"
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, timestep]
    ego_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, timestep]
    lane_xy = sim_state.roadgraph_points.xy[world_idx][lane_mask]
    lane_dir_x = sim_state.roadgraph_points.dir_x[world_idx][lane_mask]
    lane_dir_y = sim_state.roadgraph_points.dir_y[world_idx][lane_mask]
    lane_yaws = jnp.arctan2(lane_dir_y, lane_dir_x)


    dists = jnp.linalg.norm(lane_xy - ego_xy, axis=-1)
    dist_mask = dists < dist_threshold
    if not jnp.any(dist_mask):
        return -1
    candidate_dists = dists[dist_mask]
    candidate_lane_ids = lane_ids[dist_mask]
    candidate_yaws = lane_yaws[dist_mask]
    yaw_diffs = jnp.abs(jnp.arctan2(jnp.sin(candidate_yaws - ego_yaw), jnp.cos(candidate_yaws - ego_yaw)))
    yaw_mask = yaw_diffs < yaw_threshold
    if not jnp.any(yaw_mask):
        return -1
    candidate_dists = candidate_dists[yaw_mask]
    candidate_lane_ids = candidate_lane_ids[yaw_mask]
    closest_idx = jnp.argmin(candidate_dists)
    return int(candidate_lane_ids[closest_idx])

def get_left_lane_id(lane_graph, lane_id):
    lane_xy = lane_graph.nodes_xyz[:, :2]
    lane_id_to_node_range = lane_graph.lane_id_to_node_range
    left_neighbor_edges = lane_graph.left_neighbor_edges
    lane_start_idx = lane_id_to_node_range.get(int(lane_id))[0]
    left_lane_start_idx = left_neighbor_edges[left_neighbor_edges[:, 0] == lane_start_idx][:, 1]
    if len(left_lane_start_idx) == 0:
        return []
    left_lane_ids = [id for id, node_range in lane_id_to_node_range.items() if node_range[0] in left_lane_start_idx]
    current_lane_start_xy = lane_xy[lane_start_idx]
    def distance_to_current_lane(id):
        return jnp.linalg.norm(current_lane_start_xy - lane_xy[lane_id_to_node_range[id][0]])
    left_lane_ids = sorted(left_lane_ids, key=lambda id: distance_to_current_lane(id))
    return left_lane_ids

def get_right_lane_id(lane_graph, lane_id):
    lane_xy = lane_graph.nodes_xyz[:, :2]
    lane_id_to_node_range = lane_graph.lane_id_to_node_range
    left_neighbor_edges = lane_graph.left_neighbor_edges
    lane_start_idx = lane_id_to_node_range.get(int(lane_id))[0]
    right_lane_start_idx = left_neighbor_edges[left_neighbor_edges[:, 1] == lane_start_idx][:, 0]
    if len(right_lane_start_idx) == 0:
        return []
    right_lane_ids = [id for id, node_range in lane_id_to_node_range.items() if node_range[0] in right_lane_start_idx]
    current_lane_start_xy = lane_xy[lane_start_idx]
    def distance_to_current_lane(id):
        return jnp.linalg.norm(current_lane_start_xy - lane_xy[lane_id_to_node_range[id][0]])
    right_lane_ids = sorted(right_lane_ids, key=lambda id: distance_to_current_lane(id))
    return right_lane_ids

def extend_lane_id(
    lane_graph,
    lane_id,
    n_hops=5,
):
    extended_ids = [lane_id]
    current_lane_ids = [lane_id]
    lane_id_to_node_range = lane_graph.lane_id_to_node_range
    successor_edges = lane_graph.successor_edges
    for _ in range(n_hops):
        new_lane_ids = []
        for current_lane_id in current_lane_ids:
            lane_end_idx = lane_id_to_node_range.get(int(current_lane_id))[1] - 1
            new_lane_start_idx = successor_edges[successor_edges[:, 0] == lane_end_idx][:, 1]
            new_lane_ids += [id for id, node_range in lane_id_to_node_range.items() if node_range[0] in new_lane_start_idx]
        if len(new_lane_ids) == 0:
            break
        extended_ids += new_lane_ids
        current_lane_ids = new_lane_ids
    return extended_ids
    
def check_direction(
    sim_state,
    start_timestep,
    end_timestep,
    world_idx,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, start_timestep:end_timestep]
    ego_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep:end_timestep]
    start_ego_xy = ego_xy[0]
    start_ego_yaw = ego_yaw[0]
    end_ego_xy = ego_xy[-1]
    end_ego_yaw = ego_yaw[-1]
    movement_vector = end_ego_xy - start_ego_xy
    lateral_movement = -movement_vector[0] * jnp.sin(start_ego_yaw) + movement_vector[1] * jnp.cos(start_ego_yaw)
    if lateral_movement < 2.0 and lateral_movement > 1.0:
        return "slight left"
    elif lateral_movement > 2.0:
        return "left"
    elif lateral_movement > -2.0 and lateral_movement < -1.0:
        return "slight right"
    elif lateral_movement < -2.0:
        return "right"
    else:
        return "straight"

def check_speed(
    sim_state,
    start_timestep,
    end_timestep,
    world_idx,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_speed = sim_state.log_trajectory.speed[world_idx][ego_idx, start_timestep:end_timestep]
    start_speed = ego_speed[0]
    end_speed = ego_speed[-1]
    avg_speed = jnp.mean(ego_speed)
    return start_speed, end_speed, avg_speed

def check_turn(
    sim_state,
    lane_graph,
    start_timestep,
    end_timestep,
    start_lane_id,
    end_lane_id,
    direction,
    world_idx,
    yaw_threshold=jnp.pi / 6,
    u_turn_yaw_threshold=jnp.pi / 6 * 5,
):
    lane_xy = lane_graph.nodes_xyz[:, :2]
    lane_id_to_node_range = lane_graph.lane_id_to_node_range
    # ego_idx = get_ego_idx(sim_state, world_idx)
    # ego_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep:end_timestep]
    start_lane_start_idx = lane_id_to_node_range.get(int(start_lane_id))[0]
    start_lane_start_dir = lane_xy[start_lane_start_idx + 1] - lane_xy[start_lane_start_idx]
    start_lane_start_yaw = jnp.arctan2(start_lane_start_dir[1], start_lane_start_dir[0])
    end_lane_end_idx = lane_id_to_node_range.get(int(end_lane_id))[1] - 1
    end_lane_end_dir = lane_xy[end_lane_end_idx] - lane_xy[end_lane_end_idx - 1]
    end_lane_end_yaw = jnp.arctan2(end_lane_end_dir[1], end_lane_end_dir[0])
    yaw_diff = jnp.abs(jnp.arctan2(jnp.sin(end_lane_end_yaw - start_lane_start_yaw), jnp.cos(end_lane_end_yaw - start_lane_start_yaw)))
    if yaw_diff > u_turn_yaw_threshold:
        return "u-turn"
    elif yaw_diff > yaw_threshold:
        return direction
    else:
        return "straight"
    
def check_traffic_light(
    sim_state,
    lane_graph,
    start_timestep,
    start_lane_id,
    world_idx,
):
    lane_id_to_node_range = lane_graph.lane_id_to_node_range
    successor_edges = lane_graph.successor_edges
    lane_end_idx = lane_id_to_node_range.get(int(start_lane_id))[1] - 1
    successor_lane_start_idx = successor_edges[successor_edges[:, 0] == lane_end_idx][:, 1]
    if len(successor_lane_start_idx) == 0:
        return ["no successor lane"]
    successor_lane_ids = [id for id, node_range in lane_id_to_node_range.items() if node_range[0] in successor_lane_start_idx]

    tls = sim_state.log_traffic_light
    tl_state = jnp.asarray(tls.state[world_idx][:, start_timestep])
    tl_lane_ids = jnp.asarray(tls.lane_ids[world_idx][:, start_timestep])
    tl_valid = jnp.asarray(tls.valid[world_idx][:, start_timestep]).astype(jnp.bool_)
    lane_mask = tl_valid & _isin(tl_lane_ids, tuple(successor_lane_ids))
    if not jnp.any(lane_mask):
        return ["no traffic light"]
    tl_state = tl_state[lane_mask]
    tl_state = tl_state.astype(jnp.int32)
    tl_state_str = [TRAFFIC_LIGHT_STATE_DICT.get(int(state), "unknown") for state in tl_state]
    return tl_state_str

def check_risk(
    sim_state,
    start_timestep,
    end_timestep,
    world_idx,
    risk_lateral_threshold=3.0,
    risk_longitudinal_threshold=20.0,
    risk_lane_change_threshold=5.0,
):
    object_xy = sim_state.log_trajectory.xy[world_idx][:, start_timestep:end_timestep]
    object_indices = jnp.arange(object_xy.shape[0])
    object_yaw = sim_state.log_trajectory.yaw[world_idx][:, start_timestep:end_timestep]
    object_types = sim_state.object_metadata.object_types[world_idx]
    object_valid = sim_state.log_trajectory.valid[world_idx][:, start_timestep:end_timestep].astype(bool)
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = object_xy[ego_idx, 0]
    ego_yaw = object_yaw[ego_idx, 0]
    other_mask = jnp.any(object_valid, axis=1) & (jnp.arange(object_xy.shape[0]) != ego_idx)
    other_indices = object_indices[other_mask]
    if not jnp.any(other_mask):
        return []
    other_xy = object_xy[other_mask]
    other_types = object_types[other_mask]
    relative_xy = other_xy - ego_xy
    lateral_dist = -relative_xy[:, :, 0] * jnp.sin(ego_yaw) + relative_xy[:, :, 1] * jnp.cos(ego_yaw)
    longitudinal_dist = relative_xy[:, :, 0] * jnp.cos(ego_yaw) + relative_xy[:, :, 1] * jnp.sin(ego_yaw)
    behind_at_start = (
        (longitudinal_dist[:, 0] < 0)
        & (jnp.abs(lateral_dist[:, 0]) < risk_lateral_threshold)
    )
    risk_left_mask = (
        (lateral_dist[:, 0] > 0.5)
        & (lateral_dist[:, 0] < risk_lane_change_threshold)
        & (longitudinal_dist[:, 0] > -risk_lane_change_threshold)
        & (longitudinal_dist[:, 0] < risk_longitudinal_threshold)
    )
    risk_right_mask = (
        (lateral_dist[:, 0] < -0.5)
        & (lateral_dist[:, 0] > -risk_lane_change_threshold)
        & (longitudinal_dist[:, 0] > -risk_lane_change_threshold)
        & (longitudinal_dist[:, 0] < risk_longitudinal_threshold)
    )

    min_lateral_dist = jnp.min(
        jnp.abs(lateral_dist),
        axis=1,
    )
    
    min_signed_longitudinal_dist = jnp.min(
        longitudinal_dist,
        axis=1,
    )
    risk_front_mask = (
        (min_lateral_dist < risk_lateral_threshold) 
        & (min_signed_longitudinal_dist > 0) 
        & (min_signed_longitudinal_dist < risk_longitudinal_threshold)
        & ~behind_at_start
    )
    risk_front, risk_left, risk_right = [], [], []
    if jnp.any(risk_front_mask):
        risk_front_object_types = other_types[risk_front_mask].astype(jnp.int32)
        risk_front_object_types_str = [OBJECT_TYPE_DICT.get(int(t), "unknown") for t in risk_front_object_types]
        risk_front_object_indices = other_indices[risk_front_mask]
        risk_front_object_min_lateral_dist = min_lateral_dist[risk_front_mask]
        risk_front_object_min_signed_longitudinal_dist = min_signed_longitudinal_dist[risk_front_mask]
        for idx, t, lat_dist, lon_dist in zip(risk_front_object_indices, risk_front_object_types_str, risk_front_object_min_lateral_dist, risk_front_object_min_signed_longitudinal_dist):
            risk_front.append(
                {
                    "object_index": int(idx),
                    "object_type": t,
                    "min_lateral_distance": float(lat_dist),
                    "min_longitudinal_distance": float(lon_dist),
                }
            )
    if jnp.any(risk_left_mask):
        risk_left_object_types = other_types[risk_left_mask].astype(jnp.int32)
        risk_left_object_types_str = [OBJECT_TYPE_DICT.get(int(t), "unknown") for t in risk_left_object_types]
        risk_left_object_indices = other_indices[risk_left_mask]
        risk_left_object_lateral_dist = lateral_dist[risk_left_mask, 0]
        risk_left_object_longitudinal_dist = longitudinal_dist[risk_left_mask, 0]
        for idx, t, lat_dist, lon_dist in zip(risk_left_object_indices, risk_left_object_types_str, risk_left_object_lateral_dist, risk_left_object_longitudinal_dist):
            risk_left.append(
                {
                    "object_index": int(idx),
                    "object_type": t,
                    "lateral_distance": float(lat_dist),
                    "longitudinal_distance": float(lon_dist),
                }
            )
    if jnp.any(risk_right_mask):
        risk_right_object_types = other_types[risk_right_mask].astype(jnp.int32)
        risk_right_object_types_str = [OBJECT_TYPE_DICT.get(int(t), "unknown") for t in risk_right_object_types]
        risk_right_object_indices = other_indices[risk_right_mask]
        risk_right_object_lateral_dist = lateral_dist[risk_right_mask, 0]
        risk_right_object_longitudinal_dist = longitudinal_dist[risk_right_mask, 0]
        for idx, t, lat_dist, lon_dist in zip(risk_right_object_indices, risk_right_object_types_str, risk_right_object_lateral_dist, risk_right_object_longitudinal_dist):
            risk_right.append(
                {
                    "object_index": int(idx),
                    "object_type": t,
                    "lateral_distance": float(lat_dist),
                    "longitudinal_distance": float(lon_dist),
                }
            )
    return {
        "front": risk_front,
        "left": risk_left,
        "right": risk_right,
    }

def get_goal(
    sim_state,
    world_idx,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, -1]
    return (float(ego_xy[0]), float(ego_xy[1]))

def check_goal_is_behind(
    sim_state,
    goal_xy,
    start_timestep,
    world_idx
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = sim_state.log_trajectory.xy[world_idx][ego_idx, start_timestep]
    ego_yaw = sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep]
    relative_xy = jnp.array(goal_xy) - ego_xy
    longitudinal_dist = relative_xy[0] * jnp.cos(ego_yaw) + relative_xy[1] * jnp.sin(ego_yaw)
    return longitudinal_dist < 0