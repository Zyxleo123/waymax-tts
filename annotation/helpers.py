from __future__ import annotations

import numpy as np

from waymax.datatypes.object_state import ObjectTypeIds
from waymax.datatypes.roadgraph import MapElementIds
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


def _to_numpy(value):
	return np.asarray(value)


def _isin(values, test_elements):
	return np.isin(_to_numpy(values), np.asarray(test_elements))


def get_ego_idx(sim_state, world_idx=0):
	ego_mask = _to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
	ego_count = int(np.sum(ego_mask))
	if ego_count != 1:
		raise ValueError(f"Expected exactly one SDC object, got {ego_count}.")
	return int(np.argmax(ego_mask.astype(np.int32)))


def get_ego_mask(sim_state, world_idx=0):
	return _to_numpy(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)


def get_vehicle_mask(sim_state, world_idx):
	object_types = _to_numpy(sim_state.object_metadata.object_types[world_idx])
	return object_types == int(ObjectTypeIds.VEHICLE.value)


def get_pedestrian_mask(sim_state, world_idx):
	object_types = _to_numpy(sim_state.object_metadata.object_types[world_idx])
	return (object_types == int(ObjectTypeIds.PEDESTRIAN.value)) | (object_types == int(ObjectTypeIds.CYCLIST.value))


def get_current_lane_id(
	sim_state,
	timestep,
	world_idx=0,
	dist_threshold=2.0,
	yaw_threshold=np.pi / 6,
):
	lane_ids = _to_numpy(sim_state.roadgraph_points.ids[world_idx])
	lane_types = _to_numpy(sim_state.roadgraph_points.types[world_idx])
	lane_mask = _isin(lane_types, tuple(CENTERLINE_TYPES))
	lane_mask = np.logical_and(lane_mask, lane_ids >= 0)
	lane_ids = lane_ids[lane_mask]
	assert sim_state.log_trajectory.x.shape[2] == 91, f"Expected 91 timesteps, got {sim_state.log_trajectory.x.shape[2]}"
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, timestep])
	ego_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, timestep])
	ego_xy = np.stack([ego_x, ego_y], axis=-1)
	ego_yaw = float(sim_state.log_trajectory.yaw[world_idx][ego_idx, timestep])
	lane_x = _to_numpy(sim_state.roadgraph_points.x[world_idx][lane_mask])
	lane_y = _to_numpy(sim_state.roadgraph_points.y[world_idx][lane_mask])
	lane_xy = np.stack([lane_x, lane_y], axis=-1)
	lane_dir_x = _to_numpy(sim_state.roadgraph_points.dir_x[world_idx][lane_mask])
	lane_dir_y = _to_numpy(sim_state.roadgraph_points.dir_y[world_idx][lane_mask])
	lane_yaws = np.arctan2(lane_dir_y, lane_dir_x)

	dists = np.linalg.norm(lane_xy - ego_xy, axis=-1)
	dist_mask = dists < dist_threshold
	if not np.any(dist_mask):
		return -1
	candidate_dists = dists[dist_mask]
	candidate_lane_ids = lane_ids[dist_mask]
	candidate_yaws = lane_yaws[dist_mask]
	yaw_diffs = np.abs(np.arctan2(np.sin(candidate_yaws - ego_yaw), np.cos(candidate_yaws - ego_yaw)))
	yaw_mask = yaw_diffs < yaw_threshold
	if not np.any(yaw_mask):
		return -1
	closest_idx = int(np.argmin(candidate_dists[yaw_mask]))
	return int(candidate_lane_ids[yaw_mask][closest_idx])


def get_vehicle_lane_id(
	sim_state,
	timestep,
	target_lane_ids,
	world_idx=0,
	dist_threshold=2.0,
	yaw_threshold=np.pi / 6,
):
	num_objects = sim_state.log_trajectory.x[world_idx].shape[0]
	lane_ids = _to_numpy(sim_state.roadgraph_points.ids[world_idx])
	lane_types = _to_numpy(sim_state.roadgraph_points.types[world_idx])
	lane_mask = _isin(lane_types, tuple(CENTERLINE_TYPES))
	lane_mask = np.logical_and(lane_mask, lane_ids >= 0)
	lane_mask = np.logical_and(lane_mask, _isin(lane_ids, tuple(target_lane_ids)))
	lane_ids = lane_ids[lane_mask]
	lane_x = _to_numpy(sim_state.roadgraph_points.x[world_idx][lane_mask])
	lane_y = _to_numpy(sim_state.roadgraph_points.y[world_idx][lane_mask])
	lane_xy = np.stack([lane_x, lane_y], axis=-1)
	lane_dir_x = _to_numpy(sim_state.roadgraph_points.dir_x[world_idx][lane_mask])
	lane_dir_y = _to_numpy(sim_state.roadgraph_points.dir_y[world_idx][lane_mask])
	lane_yaws = np.arctan2(lane_dir_y, lane_dir_x)

	object_x = _to_numpy(sim_state.log_trajectory.x[world_idx][:, timestep])
	object_y = _to_numpy(sim_state.log_trajectory.y[world_idx][:, timestep])
	object_xy = np.stack([object_x, object_y], axis=-1)
	object_yaw = _to_numpy(sim_state.log_trajectory.yaw[world_idx][:, timestep])
	object_valid = _to_numpy(sim_state.log_trajectory.valid[world_idx][:, timestep]).astype(bool)
	object_valid = object_valid & ~get_ego_mask(sim_state, world_idx) & get_vehicle_mask(sim_state, world_idx)
	object_xy = object_xy[object_valid]
	object_yaw = object_yaw[object_valid]
	object_indices = np.arange(num_objects)[object_valid]

	dists = np.linalg.norm(lane_xy[None, :, :] - object_xy[:, None, :], axis=-1)
	dist_mask = dists < dist_threshold
	object_lane_ids = np.full(num_objects, -1, dtype=np.int32)
	for i in range(object_xy.shape[0]):
		candidate_lane_ids = lane_ids[dist_mask[i]]
		candidate_yaws = lane_yaws[dist_mask[i]]
		yaw_diffs = np.abs(np.arctan2(np.sin(candidate_yaws - object_yaw[i]), np.cos(candidate_yaws - object_yaw[i])))
		yaw_mask = yaw_diffs < yaw_threshold
		if not np.any(yaw_mask):
			continue
		candidate_lane_ids = candidate_lane_ids[yaw_mask]
		candidate_dists = dists[i][dist_mask[i]][yaw_mask]
		closest_idx = int(np.argmin(candidate_dists))
		object_lane_ids[object_indices[i]] = int(candidate_lane_ids[closest_idx])
	return object_lane_ids


def get_vehicle_on_target_lane(
	sim_state,
	timestep,
	target_lane_ids,
	object_lane_ids,
	world_idx=0,
	distance_threshold=30.0,
):
	if len(target_lane_ids) == 0:
		return None, None
	object_indices = np.arange(sim_state.log_trajectory.x[world_idx].shape[0])
	object_on_target_lane_mask = _isin(object_lane_ids, tuple(target_lane_ids))
	object_x = _to_numpy(sim_state.log_trajectory.x[world_idx][:, timestep])
	object_y = _to_numpy(sim_state.log_trajectory.y[world_idx][:, timestep])
	object_xy = np.stack([object_x, object_y], axis=-1)
	object_vx = _to_numpy(sim_state.log_trajectory.vel_x[world_idx][:, timestep])
	object_vy = _to_numpy(sim_state.log_trajectory.vel_y[world_idx][:, timestep])
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, timestep])
	ego_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, timestep])
	ego_xy = np.stack([ego_x, ego_y], axis=-1)
	ego_yaw = float(sim_state.log_trajectory.yaw[world_idx][ego_idx, timestep])

	object_rel_pos = object_xy - ego_xy
	object_rel_pos_x = object_rel_pos[:, 0] * np.cos(ego_yaw) + object_rel_pos[:, 1] * np.sin(ego_yaw)
	object_rel_pos_y = -object_rel_pos[:, 0] * np.sin(ego_yaw) + object_rel_pos[:, 1] * np.cos(ego_yaw)
	object_rel_pos = np.stack([object_rel_pos_x, object_rel_pos_y], axis=-1)
	object_speed = np.linalg.norm(np.stack([object_vx, object_vy], axis=-1), axis=-1)

	object_indices = object_indices[object_on_target_lane_mask]
	object_rel_pos = object_rel_pos[object_on_target_lane_mask]
	object_speed = object_speed[object_on_target_lane_mask]
	object_dists = np.linalg.norm(object_rel_pos, axis=-1)
	front_valid_mask = (object_rel_pos[:, 0] > 0) & (object_dists < distance_threshold)
	rear_valid_mask = (object_rel_pos[:, 0] < 0) & (object_dists < distance_threshold)

	if np.any(front_valid_mask):
		front_closest_idx = int(np.argmin(object_dists[front_valid_mask]))
		front = {
			"object_index": int(object_indices[front_valid_mask][front_closest_idx]),
			"relative_position": (
				float(object_rel_pos[front_valid_mask][front_closest_idx, 0]),
				float(object_rel_pos[front_valid_mask][front_closest_idx, 1]),
			),
			"speed": float(object_speed[front_valid_mask][front_closest_idx]),
			"distance": float(object_dists[front_valid_mask][front_closest_idx]),
		}
	else:
		front = None

	if np.any(rear_valid_mask):
		rear_closest_idx = int(np.argmin(object_dists[rear_valid_mask]))
		rear = {
			"object_index": int(object_indices[rear_valid_mask][rear_closest_idx]),
			"relative_position": (
				float(object_rel_pos[rear_valid_mask][rear_closest_idx, 0]),
				float(object_rel_pos[rear_valid_mask][rear_closest_idx, 1]),
			),
			"speed": float(object_speed[rear_valid_mask][rear_closest_idx]),
			"distance": float(object_dists[rear_valid_mask][rear_closest_idx]),
		}
	else:
		rear = None
	return front, rear


def get_left_lane_id(lane_graph, lane_id):
	lane_xy = np.asarray(lane_graph.nodes_xyz)[:, :2]
	lane_id_to_node_range = lane_graph.lane_id_to_node_range
	left_neighbor_edges = np.asarray(lane_graph.left_neighbor_edges)
	lane_start_idx = lane_id_to_node_range.get(int(lane_id))[0]
	left_lane_start_idx = left_neighbor_edges[left_neighbor_edges[:, 0] == lane_start_idx][:, 1]
	if len(left_lane_start_idx) == 0:
		return []
	left_lane_ids = [lane_id_value for lane_id_value, node_range in lane_id_to_node_range.items() if node_range[0] in left_lane_start_idx]
	current_lane_start_xy = lane_xy[lane_start_idx]
	return sorted(
		left_lane_ids,
		key=lambda lane_id_value: np.linalg.norm(current_lane_start_xy - lane_xy[lane_id_to_node_range[lane_id_value][0]]),
	)


def get_right_lane_id(lane_graph, lane_id):
	lane_xy = np.asarray(lane_graph.nodes_xyz)[:, :2]
	lane_id_to_node_range = lane_graph.lane_id_to_node_range
	left_neighbor_edges = np.asarray(lane_graph.left_neighbor_edges)
	lane_start_idx = lane_id_to_node_range.get(int(lane_id))[0]
	right_lane_start_idx = left_neighbor_edges[left_neighbor_edges[:, 1] == lane_start_idx][:, 0]
	if len(right_lane_start_idx) == 0:
		return []
	right_lane_ids = [lane_id_value for lane_id_value, node_range in lane_id_to_node_range.items() if node_range[0] in right_lane_start_idx]
	current_lane_start_xy = lane_xy[lane_start_idx]
	return sorted(
		right_lane_ids,
		key=lambda lane_id_value: np.linalg.norm(current_lane_start_xy - lane_xy[lane_id_to_node_range[lane_id_value][0]]),
	)


def extend_lane_id(
	lane_graph,
	lane_id,
	n_hops=5,
):
	extended_ids = [lane_id]
	current_lane_ids = [lane_id]
	lane_id_to_node_range = lane_graph.lane_id_to_node_range
	successor_edges = np.asarray(lane_graph.successor_edges)
	for _ in range(n_hops):
		new_lane_ids = []
		for current_lane_id in current_lane_ids:
			lane_end_idx = lane_id_to_node_range.get(int(current_lane_id))[1] - 1
			new_lane_start_idx = successor_edges[successor_edges[:, 0] == lane_end_idx][:, 1]
			new_lane_ids += [lane_id_value for lane_id_value, node_range in lane_id_to_node_range.items() if node_range[0] in new_lane_start_idx]
		if len(new_lane_ids) == 0:
			break
		extended_ids += new_lane_ids
		current_lane_ids = new_lane_ids
	current_lane_ids = [lane_id]
	for _ in range(n_hops):
		new_lane_ids = []
		for current_lane_id in current_lane_ids:
			lane_start_idx = lane_id_to_node_range.get(int(current_lane_id))[0]
			new_lane_end_idx = successor_edges[successor_edges[:, 1] == lane_start_idx][:, 0]
			new_lane_ids += [lane_id_value for lane_id_value, node_range in lane_id_to_node_range.items() if node_range[1] - 1 in new_lane_end_idx]
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
	threshold_1=0.05,
	threshold_2=0.2,
):
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, start_timestep:end_timestep])
	ego_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, start_timestep:end_timestep])
	ego_xy = np.stack([ego_x, ego_y], axis=-1)
	ego_yaw = _to_numpy(sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep:end_timestep])
	movement_vector = ego_xy[-1] - ego_xy[0]
	start_ego_yaw = float(ego_yaw[0])
	lateral_movement = -movement_vector[0] * np.sin(start_ego_yaw) + movement_vector[1] * np.cos(start_ego_yaw)
	longitudinal_movement = movement_vector[0] * np.cos(start_ego_yaw) + movement_vector[1] * np.sin(start_ego_yaw)
	movement_tangent = lateral_movement / (np.abs(longitudinal_movement) + 1e-6)
	if -threshold_1 < movement_tangent < threshold_1:
		return "straight"
	if threshold_1 < movement_tangent < threshold_2:
		return "slight left"
	if movement_tangent >= threshold_2:
		return "left"
	if -threshold_2 < movement_tangent < -threshold_1:
		return "slight right"
	return "right"


def check_speed(
	sim_state,
	start_timestep,
	end_timestep,
	world_idx,
):
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_speed = _to_numpy(sim_state.log_trajectory.speed[world_idx][ego_idx, start_timestep:end_timestep])
	return ego_speed[0], ego_speed[-1], np.mean(ego_speed)


def check_turn(
	sim_state,
	lane_graph,
	start_timestep,
	end_timestep,
	start_lane_id,
	end_lane_id,
	direction,
	world_idx,
	yaw_threshold=np.pi / 6,
):
	lane_xy = np.asarray(lane_graph.nodes_xyz)[:, :2]
	lane_id_to_node_range = lane_graph.lane_id_to_node_range
	start_lane_start_idx = lane_id_to_node_range.get(int(start_lane_id))[0]
	start_lane_start_dir = lane_xy[start_lane_start_idx + 1] - lane_xy[start_lane_start_idx]
	start_lane_start_yaw = np.arctan2(start_lane_start_dir[1], start_lane_start_dir[0])
	end_lane_end_idx = lane_id_to_node_range.get(int(end_lane_id))[1] - 1
	end_lane_end_dir = lane_xy[end_lane_end_idx] - lane_xy[end_lane_end_idx - 1]
	end_lane_end_yaw = np.arctan2(end_lane_end_dir[1], end_lane_end_dir[0])
	yaw_diff = np.abs(np.arctan2(np.sin(end_lane_end_yaw - start_lane_start_yaw), np.cos(end_lane_end_yaw - start_lane_start_yaw)))
	return direction if yaw_diff > yaw_threshold else "straight"


def check_traffic_light(
	sim_state,
	lane_graph,
	start_timestep,
	start_lane_id,
	world_idx,
	yaw_threshold=np.pi / 6,
):
	lane_xy = np.asarray(lane_graph.nodes_xyz)[:, :2]
	lane_id_to_node_range = lane_graph.lane_id_to_node_range
	successor_edges = np.asarray(lane_graph.successor_edges)
	lane_end_idx = lane_id_to_node_range.get(int(start_lane_id))[1] - 1
	successor_lane_start_idx = successor_edges[successor_edges[:, 0] == lane_end_idx][:, 1]
	if len(successor_lane_start_idx) == 0:
		return ["no successor lane"]
	successor_lane_ids = [lane_id_value for lane_id_value, node_range in lane_id_to_node_range.items() if node_range[0] in successor_lane_start_idx]

	tls = sim_state.log_traffic_light
	tl_state = _to_numpy(tls.state[world_idx][:, start_timestep])
	tl_lane_ids = _to_numpy(tls.lane_ids[world_idx][:, start_timestep])
	tl_valid = _to_numpy(tls.valid[world_idx][:, start_timestep]).astype(bool)
	lane_mask = tl_valid & _isin(tl_lane_ids, tuple(successor_lane_ids))
	if not np.any(lane_mask):
		return {}
	tl_state = tl_state[lane_mask]
	tl_lane_ids = tl_lane_ids[lane_mask]
	output = {}
	for lane_id, state in zip(tl_lane_ids, tl_state.astype(np.int32)):
		start_idx = lane_id_to_node_range.get(int(lane_id))[0]
		end_idx = lane_id_to_node_range.get(int(lane_id))[1] - 1
		start_yaw = np.arctan2(lane_xy[start_idx + 1, 1] - lane_xy[start_idx, 1], lane_xy[start_idx + 1, 0] - lane_xy[start_idx, 0])
		end_yaw = np.arctan2(lane_xy[end_idx, 1] - lane_xy[end_idx - 1, 1], lane_xy[end_idx, 0] - lane_xy[end_idx - 1, 0])
		yaw_diff = np.arctan2(np.sin(end_yaw - start_yaw), np.cos(end_yaw - start_yaw))
		if yaw_diff < -yaw_threshold:
			turn_direction = "left"
		elif yaw_diff > yaw_threshold:
			turn_direction = "right"
		else:
			turn_direction = "straight"
		output[turn_direction] = TRAFFIC_LIGHT_STATE_DICT.get(int(state), "unknown")
	return output


def check_risk(
	sim_state,
	start_timestep,
	end_timestep,
	world_idx,
	risk_lateral_threshold=3.0,
	risk_longitudinal_threshold=20.0,
):
	object_x = _to_numpy(sim_state.log_trajectory.x[world_idx][:, start_timestep:end_timestep])
	object_y = _to_numpy(sim_state.log_trajectory.y[world_idx][:, start_timestep:end_timestep])
	object_xy = np.stack([object_x, object_y], axis=-1)
	object_yaw = _to_numpy(sim_state.log_trajectory.yaw[world_idx][:, start_timestep:end_timestep])
	object_types = _to_numpy(sim_state.object_metadata.object_types[world_idx])
	object_valid = _to_numpy(sim_state.log_trajectory.valid[world_idx][:, start_timestep:end_timestep]).astype(bool)
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_xy = object_xy[ego_idx, 0]
	ego_yaw = float(object_yaw[ego_idx, 0])
	other_mask = np.any(object_valid, axis=1) & (np.arange(object_xy.shape[0]) != ego_idx)
	if not np.any(other_mask):
		return []
	other_indices = np.arange(object_xy.shape[0])[other_mask]
	other_xy = object_xy[other_mask]
	other_types = object_types[other_mask]
	relative_xy = other_xy - ego_xy
	lateral_dist = -relative_xy[:, :, 0] * np.sin(ego_yaw) + relative_xy[:, :, 1] * np.cos(ego_yaw)
	longitudinal_dist = relative_xy[:, :, 0] * np.cos(ego_yaw) + relative_xy[:, :, 1] * np.sin(ego_yaw)
	behind_at_start = (longitudinal_dist[:, 0] < 0) & (np.abs(lateral_dist[:, 0]) < risk_lateral_threshold)
	min_lateral_dist = np.min(np.abs(lateral_dist), axis=1)
	min_signed_longitudinal_dist = np.min(longitudinal_dist, axis=1)
	risk_front_mask = (
		(min_lateral_dist < risk_lateral_threshold)
		& (min_signed_longitudinal_dist > 0)
		& (min_signed_longitudinal_dist < risk_longitudinal_threshold)
		& ~behind_at_start
	)
	risk_front = []
	if np.any(risk_front_mask):
		risk_front_object_types = other_types[risk_front_mask].astype(np.int32)
		risk_front_object_indices = other_indices[risk_front_mask]
		for idx, type_value, lat_dist, lon_dist in zip(
			risk_front_object_indices,
			[OBJECT_TYPE_DICT.get(int(value), "unknown") for value in risk_front_object_types],
			min_lateral_dist[risk_front_mask],
			min_signed_longitudinal_dist[risk_front_mask],
		):
			risk_front.append(
				{
					"object_index": int(idx),
					"object_type": type_value,
					"min_lateral_distance": float(lat_dist),
					"min_longitudinal_distance": float(lon_dist),
				}
			)
	return risk_front


def get_goal_info(
	sim_state,
	timestep,
	world_idx,
):
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_yaw = float(sim_state.log_trajectory.yaw[world_idx][ego_idx, timestep])
	ego_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, timestep])
	ego_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, timestep])
	ego_xy = np.stack([ego_x, ego_y], axis=-1)
	goal_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, -1])
	goal_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, -1])
	goal_xy = np.stack([goal_x, goal_y], axis=-1)
	rel_goal_xy = goal_xy - ego_xy
	rel_goal_x = rel_goal_xy[0] * np.cos(ego_yaw) + rel_goal_xy[1] * np.sin(ego_yaw)
	rel_goal_y = -rel_goal_xy[0] * np.sin(ego_yaw) + rel_goal_xy[1] * np.cos(ego_yaw)

	lane_ids = _to_numpy(sim_state.roadgraph_points.ids[world_idx])
	lane_types = _to_numpy(sim_state.roadgraph_points.types[world_idx])
	lane_mask = _isin(lane_types, tuple(CENTERLINE_TYPES))
	lane_mask = np.logical_and(lane_mask, lane_ids >= 0)
	lane_ids = lane_ids[lane_mask]
	lane_x = _to_numpy(sim_state.roadgraph_points.x[world_idx][lane_mask])
	lane_y = _to_numpy(sim_state.roadgraph_points.y[world_idx][lane_mask])
	lane_xy = np.stack([lane_x, lane_y], axis=-1)
	dists = np.linalg.norm(lane_xy - goal_xy, axis=-1)
	closest_idx = int(np.argmin(dists))
	return {
		"relative_position": (float(rel_goal_x), float(rel_goal_y)),
		"lane_id": int(lane_ids[closest_idx]),
	}


def check_goal_is_behind(
	sim_state,
	goal_xy,
	start_timestep,
	world_idx,
):
	ego_idx = get_ego_idx(sim_state, world_idx)
	ego_x = _to_numpy(sim_state.log_trajectory.x[world_idx][ego_idx, start_timestep])
	ego_y = _to_numpy(sim_state.log_trajectory.y[world_idx][ego_idx, start_timestep])
	ego_xy = np.stack([ego_x, ego_y], axis=-1)
	ego_yaw = float(sim_state.log_trajectory.yaw[world_idx][ego_idx, start_timestep])
	relative_xy = np.array(goal_xy) - ego_xy
	longitudinal_dist = relative_xy[0] * np.cos(ego_yaw) + relative_xy[1] * np.sin(ego_yaw)
	return longitudinal_dist < 0