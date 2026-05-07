from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from waymax.datatypes.roadgraph import MapElementIds
from waymax.datatypes.object_state import ObjectTypeIds


CENTERLINE_TYPES = {
    int(MapElementIds.LANE_FREEWAY.value),
    int(MapElementIds.LANE_SURFACE_STREET.value),
}


def get_ego_idx(sim_state, world_idx):
    ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
    if int(jnp.sum(ego_mask)) != 1:
        raise ValueError(
            f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}."
        )
    return int(jnp.argmax(ego_mask.astype(jnp.int32)))

def get_vehicle_mask(sim_state, world_idx):
    object_types = jnp.asarray(sim_state.object_metadata.object_types[world_idx])
    vehicle_mask = (object_types == int(ObjectTypeIds.VEHICLE.value))
    return vehicle_mask

def get_pedestrian_mask(sim_state, world_idx):
    object_types = jnp.asarray(sim_state.object_metadata.object_types[world_idx])
    pedestrian_mask = (object_types == int(ObjectTypeIds.PEDESTRIAN.value)) | (object_types == int(ObjectTypeIds.CYCLIST.value))
    return pedestrian_mask


def get_vehicle_current_lane_ids(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
    seed_heading_threshold_rad=jnp.pi / 6.0,
) -> List[Optional[int]]:
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)

    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx][ego_idx], dtype=jnp.float32)
    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    vehicle_mask = jnp.asarray(get_vehicle_mask(sim_state, world_idx)).astype(jnp.bool_)

    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )
    if vehicle_mask.ndim != 1:
        raise ValueError(
            "sim_state.object_metadata.object_types must produce a 1D vehicle mask, "
            f"got {vehicle_mask.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))
    distance = jnp.linalg.norm(object_xy[:, t] - ego_xy[t, :], axis=-1)
    num_objects = int(object_xy.shape[0])
    current_lane_ids: List[Optional[int]] = [None] * num_objects
    closest_lane_point: List[Optional[int]] = [None] * num_objects
    for obj_idx in range(num_objects):
        if not bool(vehicle_mask[obj_idx]) or not bool(object_valid[obj_idx, t]) or distance[obj_idx] > max_distance:
            continue
        seed_idx = get_closest_lane_point(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            target_vehicle=obj_idx,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        closest_lane_point[obj_idx] = seed_idx
        current_lane_ids[obj_idx] = lane_id_from_seed(scorer, seed_idx)

    return current_lane_ids, closest_lane_point

def lane_xy_dir(scorer: Any, lane_id: int) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    if scorer._lane_graph is None or scorer.lane_points is None:
        return None
    node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lane_id))
    if node_range is None:
        return None
    start, end = node_range
    if end <= start:
        return None
    lane_xy = jnp.asarray(scorer.lane_points[start:end, :2], dtype=jnp.float32)
    lane_dir = jnp.asarray(scorer.lane_points[start:end, 2:4], dtype=jnp.float32)
    return lane_xy, lane_dir

def get_closest_lane_point(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    target_vehicle="ego",
    seed_heading_threshold_rad=jnp.pi / 6.0,
):
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    if target_vehicle == "ego":
        target_idx = get_ego_idx(sim_state, world_idx)
    else:
        target_idx = int(target_vehicle)
        if target_idx < 0 or target_idx >= int(object_xy.shape[0]):
            raise IndexError(
                f"target_vehicle index out of range: {target_idx} for num_objects={int(object_xy.shape[0])}"
            )

    target_xy = object_xy[target_idx, t]
    target_heading = object_yaw[target_idx, t]

    lane_points = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
    lane_xy = lane_points[:, :2]
    lane_dir = lane_points[:, 2:4]
    lane_heading = jnp.arctan2(lane_dir[:, 1], lane_dir[:, 0])

    seed_heading_delta = target_heading - lane_heading
    seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
    seed_heading_ok = jnp.abs(seed_heading_delta_wrapped) <= jnp.asarray(
        seed_heading_threshold_rad, dtype=jnp.float32
    )

    dist2 = jnp.sum((lane_xy - target_xy[None, :]) ** 2, axis=-1)
    large = jnp.asarray(1e12, dtype=jnp.float32)
    seed_dist2 = jnp.where(seed_heading_ok, dist2, large)
    if bool(jnp.any(seed_heading_ok)):
        return int(jnp.argmin(seed_dist2))
    return int(jnp.argmin(dist2))

def choose_best_connected_lane(
    scorer: Any,
    lane_id: int,
    candidates: List[int],
    direction: str,
    successor_branch: str = "straight",
) -> Optional[int]:
    if not candidates:
        return None
    cur = lane_xy_dir(scorer, lane_id)
    if cur is None:
        return int(candidates[0])

    cur_xy, cur_dir = cur
    if direction == "forward":
        anchor_xy = cur_xy[-1]
        anchor_dir = cur_dir[-1]
    else:
        anchor_xy = cur_xy[0]
        anchor_dir = cur_dir[0]

    anchor_norm = jnp.maximum(jnp.linalg.norm(anchor_dir), 1e-6)
    anchor_dir = anchor_dir / anchor_norm

    if successor_branch not in {"straight", "left", "right"}:
        successor_branch = "straight"

    best_lane_id = None
    best_primary = float("-inf")
    best_secondary = float("-inf")
    best_dist = float("inf")
    for cand_lane_id in candidates:
        cand = lane_xy_dir(scorer, int(cand_lane_id))
        if cand is None:
            continue
        cand_xy, cand_dir = cand
        if direction == "forward":
            cand_anchor_xy = cand_xy[0]
            cand_anchor_dir = cand_dir[0]
        else:
            cand_anchor_xy = cand_xy[-1]
            cand_anchor_dir = cand_dir[-1]

        cand_norm = jnp.maximum(jnp.linalg.norm(cand_anchor_dir), 1e-6)
        cand_anchor_dir = cand_anchor_dir / cand_norm
        cos_sim = float(jnp.dot(anchor_dir, cand_anchor_dir))
        dist = float(jnp.linalg.norm(cand_anchor_xy - anchor_xy))

        cross_z = float(anchor_dir[0] * cand_anchor_dir[1] - anchor_dir[1] * cand_anchor_dir[0])
        signed_angle = float(np.arctan2(cross_z, cos_sim))

        if direction == "forward":
            if successor_branch == "left":
                primary = signed_angle
                secondary = cos_sim
            elif successor_branch == "right":
                primary = -signed_angle
                secondary = cos_sim
            else:
                primary = cos_sim
                secondary = -abs(signed_angle)
        else:
            primary = cos_sim
            secondary = 0.0

        better = False
        if primary > best_primary + 1e-8:
            better = True
        elif abs(primary - best_primary) <= 1e-8:
            if secondary > best_secondary + 1e-8:
                better = True
            elif abs(secondary - best_secondary) <= 1e-8 and dist < best_dist:
                better = True

        if better:
            best_primary = primary
            best_secondary = secondary
            best_dist = dist
            best_lane_id = int(cand_lane_id)

    return best_lane_id


def lane_chain_ids(
    scorer: Any, center_lane_id: int, n_hops: int, successor_branch: str = "straight"
) -> List[int]:
    if scorer._lane_graph is None:
        return [int(center_lane_id)]

    n_hops = max(int(n_hops), 0)
    center_lane_id = int(center_lane_id)

    prev_ids: List[int] = []
    visited = {center_lane_id}
    cur = center_lane_id
    for _ in range(n_hops):
        cands = scorer._predecessor_lane_ids.get(cur, [])
        cands = [int(x) for x in cands if int(x) not in visited]
        next_lane = choose_best_connected_lane(
            scorer, cur, cands, direction="backward", successor_branch=successor_branch
        )
        if next_lane is None:
            break
        prev_ids.append(int(next_lane))
        visited.add(int(next_lane))
        cur = int(next_lane)

    next_ids: List[int] = []
    cur = center_lane_id
    for _ in range(n_hops):
        cands = scorer._successor_lane_ids.get(cur, [])
        cands = [int(x) for x in cands if int(x) not in visited]
        next_lane = choose_best_connected_lane(
            scorer, cur, cands, direction="forward", successor_branch=successor_branch
        )
        if next_lane is None:
            break
        next_ids.append(int(next_lane))
        visited.add(int(next_lane))
        cur = int(next_lane)

    return list(reversed(prev_ids)) + [center_lane_id] + next_ids


def concat_lane_polylines(polylines: List[jnp.ndarray]) -> jnp.ndarray:
    if not polylines:
        return jnp.empty((0, 4), dtype=jnp.float32)

    merged = []
    for i, p in enumerate(polylines):
        p = jnp.asarray(p, dtype=jnp.float32)
        if int(p.shape[0]) == 0:
            continue
        if i == 0 or not merged:
            merged.append(p)
            continue
        prev = merged[-1]
        if int(prev.shape[0]) == 0:
            merged[-1] = p
            continue
        if float(jnp.linalg.norm(prev[-1, :2] - p[0, :2])) <= 1e-3:
            merged.append(p[1:])
        else:
            merged.append(p)

    if not merged:
        return jnp.empty((0, 4), dtype=jnp.float32)
    return jnp.concatenate(merged, axis=0).astype(jnp.float32)


def load_lane_graph_by_index(scorer: Any, scenario_index: int) -> bool:
    if scorer._lane_graph_shard_store is None:
        return False
    lane_graph = scorer._lane_graph_shard_store.get_by_record_index(int(scenario_index))
    if lane_graph is None:
        return False
    scorer.set_lane_graph(lane_graph)
    return True


def update_lane_points(scorer: Any, sim_state, world_idx=0):
    if scorer._lane_graph is None:
        if scorer._lane_graph_shard_store is not None:
            target_scenario_index: Optional[int] = None
            if scorer._scenario_indices is not None and 0 <= int(world_idx) < len(scorer._scenario_indices):
                target_scenario_index = int(scorer._scenario_indices[int(world_idx)])
            elif scorer._scenario_index is not None:
                target_scenario_index = int(scorer._scenario_index)

            if target_scenario_index is not None:
                _ = load_lane_graph_by_index(scorer, target_scenario_index)

    if scorer._lane_graph is not None:
        if scorer.lane_points is None:
            scorer.set_lane_graph(scorer._lane_graph)
        return
    else:
        raise ValueError("Failed to load lane graph for lane points update, and no fallback method available.")


def lane_polyline_from_lane_id(
    scorer: Any,
    lane_id: int,
    n_hops: int = 0,
    successor_branch: str = "straight",
) -> Optional[jnp.ndarray]:
    if scorer._lane_graph is None:
        return None
    lane_chain = lane_chain_ids(
        scorer, int(lane_id), n_hops=int(n_hops), successor_branch=successor_branch
    )
    lane_chain = list(set(lane_chain))
    polylines: List[jnp.ndarray] = []
    for lane_id_i in lane_chain:
        node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lane_id_i))
        if node_range is None:
            continue
        start, end = node_range
        if end <= start:
            continue
        polylines.append(scorer.lane_points[start:end, :4])
    if not polylines:
        return jnp.empty((0, 4), dtype=jnp.float32)
    return concat_lane_polylines(polylines)


def lane_id_from_seed(scorer: Any, seed_idx: int) -> Optional[int]:
    if scorer._lane_graph is None or scorer.lane_points is None:
        return None
    if seed_idx < 0 or seed_idx >= int(scorer.lane_points.shape[0]):
        return None
    return int(scorer._lane_graph.node_lane_ids[int(seed_idx)])


def get_side_lane_ids(scorer: Any, lane_id: int, side: str) -> List[int]:
    if side == "left":
        return scorer._left_neighbor_lane_ids.get(int(lane_id), [])
    if side == "right":
        return scorer._right_neighbor_lane_ids.get(int(lane_id), [])
    return []


def pick_best_side_lane(
    scorer: Any,
    seed_idx: int,
    candidate_lane_ids: List[int],
    max_side_distance_m: float,
) -> Optional[int]:
    if not candidate_lane_ids or scorer.lane_points is None:
        return None

    lane_xy = jnp.asarray(scorer.lane_points[:, :2], dtype=jnp.float32)
    seed_xy = lane_xy[int(seed_idx)]

    best_lane_id = None
    best_dist2 = float("inf")
    for lane_id in candidate_lane_ids:
        node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lane_id)) if scorer._lane_graph is not None else None
        if node_range is None:
            continue
        start, end = node_range
        if end <= start:
            continue
        lane_pts = lane_xy[start:end]
        d2 = jnp.sum((lane_pts - seed_xy[None, :]) ** 2, axis=-1)
        min_d2 = float(jnp.min(d2))
        if min_d2 <= float(max_side_distance_m) ** 2 and min_d2 < best_dist2:
            best_dist2 = min_d2
            best_lane_id = int(lane_id)

    return best_lane_id


def get_current_lane(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    n_hops: int = 2,
    successor_branch: str = "straight",
):
    del ray_distance_threshold_m

    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return jnp.empty((0, 4), dtype=jnp.float32)

    seed_idx = get_closest_lane_point(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        world_idx=world_idx,
        target_vehicle="ego",
        seed_heading_threshold_rad=seed_heading_threshold_rad,
    )
    lane_id = lane_id_from_seed(scorer, seed_idx)
    if lane_id is None:
        return jnp.empty((0, 4), dtype=jnp.float32)
    lane = lane_polyline_from_lane_id(
        scorer, lane_id, n_hops=n_hops, successor_branch=successor_branch
    )
    if lane is None:
        return jnp.empty((0, 4), dtype=jnp.float32)
    return lane


def get_side_lane(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    side="left",
    max_side_distance_m=5.0,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    n_hops: int = 2,
    successor_branch: str = "straight",
):
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return None

    seed_idx = get_closest_lane_point(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        world_idx=world_idx,
        target_vehicle="ego",
        seed_heading_threshold_rad=seed_heading_threshold_rad,
    )
    cur_lane_id = lane_id_from_seed(scorer, seed_idx)
    if cur_lane_id is None:
        return None

    candidate_lane_ids = get_side_lane_ids(scorer, cur_lane_id, side=side)
    best_lane_id = pick_best_side_lane(
        scorer,
        seed_idx,
        candidate_lane_ids,
        max_side_distance_m=float(max_side_distance_m),
    )
    if best_lane_id is None:
        return None

    return lane_polyline_from_lane_id(
        scorer, best_lane_id, n_hops=n_hops, successor_branch=successor_branch
    )


def get_left_lane(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_side_distance_m=6.0,
    max_direction_delta_rad=jnp.pi / 6.0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    n_hops: int = 2,
    successor_branch: str = "left",
):
    return get_side_lane(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        world_idx=world_idx,
        side="left",
        max_side_distance_m=max_side_distance_m,
        seed_heading_threshold_rad=seed_heading_threshold_rad,
        n_hops=n_hops,
        successor_branch=successor_branch,
    )


def get_right_lane(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_side_distance_m=6.0,
    max_direction_delta_rad=jnp.pi / 6.0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    n_hops: int = 2,
    successor_branch: str = "right",
):
    return get_side_lane(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        world_idx=world_idx,
        side="right",
        max_side_distance_m=max_side_distance_m,
        seed_heading_threshold_rad=seed_heading_threshold_rad,
        n_hops=n_hops,
        successor_branch=successor_branch,
    )


def get_vehicle_front(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    vehicle_current_lane_ids: Optional[List[Optional[int]]] = None,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    vehicle_mask = get_vehicle_mask(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    ego_lane_id = None
    if vehicle_current_lane_ids is not None and ego_idx < len(vehicle_current_lane_ids):
        ego_lane_id = vehicle_current_lane_ids[ego_idx]
    if ego_lane_id is None:
        ego_lane_point = get_closest_lane_point(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            target_vehicle=ego_idx,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        ego_lane_id = lane_id_from_seed(scorer, ego_lane_point)
    if ego_lane_id is None:
        return []
    ego_lane_ids = lane_chain_ids(
        scorer, ego_lane_id, n_hops=3, successor_branch="straight"
    )

    # Precompute ego-lane point indices to limit per-object distance computations
    ego_lane_point_indices: List[int] = []
    if scorer._lane_graph is not None:
        for lid in ego_lane_ids:
            node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lid))
            if node_range is None:
                continue
            start, end = node_range
            if end <= start:
                continue
            ego_lane_point_indices.extend(list(range(start, end)))

    # If lane points exist for ego lanes, prepare their xy array
    lane_points_global = None
    if ego_lane_point_indices:
        lane_points_global = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
        ego_lane_xy = lane_points_global[jnp.asarray(ego_lane_point_indices, dtype=jnp.int32), :2]

    num_objects = int(object_xy.shape[0])
    vehicles_front = []

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx or not bool(vehicle_mask[obj_idx]) or not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        distance = jnp.linalg.norm(rel_xy)
        is_forward = jnp.dot(rel_xy, ego_forward) > 0
        if distance > float(max_distance) or not bool(is_forward):
            continue

        obj_lane_id = None
        # If caller provided current lane ids, use them directly
        if vehicle_current_lane_ids is not None and obj_idx < len(vehicle_current_lane_ids):
            obj_lane_id = vehicle_current_lane_ids[obj_idx]

        # Otherwise, restrict search to ego-lane points only (faster)
        if obj_lane_id is None:
            if ego_lane_point_indices and lane_points_global is not None:
                # compute squared distances to ego-lane points and pick the closest global seed index
                d2 = jnp.sum((ego_lane_xy - obj_xy[None, :]) ** 2, axis=-1)
                min_idx = int(jnp.argmin(d2))
                min_distance = float(d2[min_idx] ** 0.5)
                obj_seed_global = int(ego_lane_point_indices[min_idx])
                # compute lane heading at the chosen ego-lane point and compare heading delta
                lane_dir = lane_points_global[obj_seed_global, 2:4]
                lane_heading = jnp.arctan2(lane_dir[1], lane_dir[0])
                obj_heading = object_yaw[obj_idx, t]
                seed_heading_delta = obj_heading - lane_heading
                seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
                if jnp.abs(seed_heading_delta_wrapped) > jnp.asarray(seed_heading_threshold_rad, dtype=jnp.float32) \
                    or min_distance > 2.0:
                    # heading difference too large -> treat as not on ego lane
                    obj_lane_id = -1
                else:
                    obj_lane_id = lane_id_from_seed(scorer, obj_seed_global)
            else:
                # fallback to global closest search
                obj_seed_idx = get_closest_lane_point(
                    scorer=scorer,
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                    target_vehicle=obj_idx,
                    seed_heading_threshold_rad=seed_heading_threshold_rad,
                )
                obj_lane_id = lane_id_from_seed(scorer, obj_seed_idx)

        if obj_lane_id in ego_lane_ids:
            vehicles_front.append((obj_idx, distance))

    return vehicles_front


def get_vehicle_behind(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    vehicle_current_lane_ids: Optional[List[Optional[int]]] = None,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    vehicle_mask = get_vehicle_mask(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    ego_lane_id = None
    if vehicle_current_lane_ids is not None and ego_idx < len(vehicle_current_lane_ids):
        ego_lane_id = vehicle_current_lane_ids[ego_idx]
    if ego_lane_id is None:
        ego_lane_point = get_closest_lane_point(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            target_vehicle=ego_idx,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        ego_lane_id = lane_id_from_seed(scorer, ego_lane_point)
    if ego_lane_id is None:
        return []
    ego_lane_ids = lane_chain_ids(
        scorer, ego_lane_id, n_hops=3, successor_branch="straight"
    )
    # Precompute ego-lane point indices to limit per-object distance computations
    ego_lane_point_indices: List[int] = []
    if scorer._lane_graph is not None:
        for lid in ego_lane_ids:
            node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lid))
            if node_range is None:
                continue
            start, end = node_range
            if end <= start:
                continue
            ego_lane_point_indices.extend(list(range(start, end)))

    lane_points_global = None
    if ego_lane_point_indices:
        lane_points_global = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
        ego_lane_xy = lane_points_global[jnp.asarray(ego_lane_point_indices, dtype=jnp.int32), :2]

    num_objects = int(object_xy.shape[0])
    vehicles_behind = []

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx or not bool(vehicle_mask[obj_idx]) or not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        distance = jnp.linalg.norm(rel_xy)
        is_behind = jnp.dot(rel_xy, ego_forward) < 0
        if distance > float(max_distance) or not bool(is_behind):
            continue

        obj_lane_id = None
        if vehicle_current_lane_ids is not None and obj_idx < len(vehicle_current_lane_ids):
            obj_lane_id = vehicle_current_lane_ids[obj_idx]

        if obj_lane_id is None:
            if ego_lane_point_indices and lane_points_global is not None:
                d2 = jnp.sum((ego_lane_xy - obj_xy[None, :]) ** 2, axis=-1)
                min_idx = int(jnp.argmin(d2))
                min_distance = float(d2[min_idx] ** 0.5)
                obj_seed_global = int(ego_lane_point_indices[min_idx])
                lane_dir = lane_points_global[obj_seed_global, 2:4]
                lane_heading = jnp.arctan2(lane_dir[1], lane_dir[0])
                obj_heading = object_yaw[obj_idx, t]
                seed_heading_delta = obj_heading - lane_heading
                seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
                if jnp.abs(seed_heading_delta_wrapped) > jnp.asarray(seed_heading_threshold_rad, dtype=jnp.float32) \
                    or min_distance > 2.0:
                    obj_lane_id = -1
                else:
                    obj_lane_id = lane_id_from_seed(scorer, obj_seed_global)
            else:
                obj_seed_idx = get_closest_lane_point(
                    scorer=scorer,
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                    target_vehicle=obj_idx,
                    seed_heading_threshold_rad=seed_heading_threshold_rad,
                )
                obj_lane_id = lane_id_from_seed(scorer, obj_seed_idx)

        if obj_lane_id in ego_lane_ids:
            vehicles_behind.append((obj_idx, distance))

    return vehicles_behind


def get_vehicle_left(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    vehicle_current_lane_ids: Optional[List[Optional[int]]] = None,
    closest_lane_points: Optional[List[Optional[int]]] = None,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    vehicle_mask = get_vehicle_mask(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    ego_lane_id = None
    if vehicle_current_lane_ids is not None and ego_idx < len(vehicle_current_lane_ids):
        ego_lane_point = closest_lane_points[ego_idx]
        ego_lane_id = vehicle_current_lane_ids[ego_idx]
    else:
        ego_lane_point = get_closest_lane_point(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            target_vehicle=ego_idx,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        ego_lane_id = lane_id_from_seed(scorer, ego_lane_point)
    if ego_lane_id is None:
        return []
    
    left_lane_cand_ids = get_side_lane_ids(scorer, ego_lane_id, side="left")
    left_lane_id = pick_best_side_lane(
        scorer,
        seed_idx=ego_lane_point,
        candidate_lane_ids=left_lane_cand_ids,
        max_side_distance_m=float(max_distance),
    )
    left_lane_ids = lane_chain_ids(
        scorer, left_lane_id, n_hops=3, successor_branch="straight"
    ) if left_lane_id is not None else []
    # Precompute left-lane point indices for faster per-object checks
    left_lane_point_indices: List[int] = []
    if scorer._lane_graph is not None and left_lane_ids:
        for lid in left_lane_ids:
            node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lid))
            if node_range is None:
                continue
            start, end = node_range
            if end <= start:
                continue
            left_lane_point_indices.extend(list(range(start, end)))

    lane_points_global = None
    if left_lane_point_indices:
        lane_points_global = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
        left_lane_xy = lane_points_global[jnp.asarray(left_lane_point_indices, dtype=jnp.int32), :2]

    num_objects = int(object_xy.shape[0])
    vehicles_left = []

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx or not bool(vehicle_mask[obj_idx]) or not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        distance = jnp.linalg.norm(rel_xy)
        if distance > float(max_distance):
            continue

        obj_lane_id = None
        if vehicle_current_lane_ids is not None and obj_idx < len(vehicle_current_lane_ids):
            obj_seed_idx = closest_lane_points[obj_idx]
            obj_lane_id = vehicle_current_lane_ids[obj_idx]

        if obj_lane_id is None:
            if left_lane_point_indices and lane_points_global is not None:
                d2 = jnp.sum((left_lane_xy - obj_xy[None, :]) ** 2, axis=-1)
                min_idx = int(jnp.argmin(d2))
                min_distance = float(d2[min_idx] ** 0.5)
                obj_seed_global = int(left_lane_point_indices[min_idx])
                lane_dir = lane_points_global[obj_seed_global, 2:4]
                lane_heading = jnp.arctan2(lane_dir[1], lane_dir[0])
                obj_heading = object_yaw[obj_idx, t]
                seed_heading_delta = obj_heading - lane_heading
                seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
                if jnp.abs(seed_heading_delta_wrapped) > jnp.asarray(seed_heading_threshold_rad, dtype=jnp.float32) or min_distance > 2.0:
                    obj_lane_id = -1
                else:
                    obj_lane_id = lane_id_from_seed(scorer, obj_seed_global)
            else:
                obj_seed_idx = get_closest_lane_point(
                    scorer=scorer,
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                    target_vehicle=obj_idx,
                    seed_heading_threshold_rad=seed_heading_threshold_rad,
                )
                obj_lane_id = lane_id_from_seed(scorer, obj_seed_idx)

        if obj_lane_id in left_lane_ids:
            vehicles_left.append((obj_idx, distance))

    return vehicles_left


def get_vehicle_right(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    vehicle_current_lane_ids: Optional[List[Optional[int]]] = None,
    closest_lane_points: Optional[List[Optional[int]]] = None,
):
    ego_idx = get_ego_idx(sim_state, world_idx)
    vehicle_mask = get_vehicle_mask(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    
    ego_lane_id = None
    if vehicle_current_lane_ids is not None and ego_idx < len(vehicle_current_lane_ids):
        ego_lane_point = closest_lane_points[ego_idx]
        ego_lane_id = vehicle_current_lane_ids[ego_idx]
    else:
        ego_lane_point = get_closest_lane_point(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            target_vehicle=ego_idx,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        ego_lane_id = lane_id_from_seed(scorer, ego_lane_point)
    if ego_lane_id is None:
        return []
    
    right_lane_cand_ids = get_side_lane_ids(scorer, ego_lane_id, side="right")
    right_lane_id = pick_best_side_lane(
        scorer,
        seed_idx=ego_lane_point,
        candidate_lane_ids=right_lane_cand_ids,
        max_side_distance_m=float(max_distance),
    )
    right_lane_ids = lane_chain_ids(
        scorer, right_lane_id, n_hops=3, successor_branch="straight"
    ) if right_lane_id is not None else []
    # Precompute right-lane point indices for faster per-object checks
    right_lane_point_indices: List[int] = []
    if scorer._lane_graph is not None and right_lane_ids:
        for lid in right_lane_ids:
            node_range = scorer._lane_graph.lane_id_to_node_range.get(int(lid))
            if node_range is None:
                continue
            start, end = node_range
            if end <= start:
                continue
            right_lane_point_indices.extend(list(range(start, end)))

    lane_points_global = None
    if right_lane_point_indices:
        lane_points_global = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
        right_lane_xy = lane_points_global[jnp.asarray(right_lane_point_indices, dtype=jnp.int32), :2]

    num_objects = int(object_xy.shape[0])
    vehicles_right = []

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx or not bool(vehicle_mask[obj_idx]) or not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        distance = jnp.linalg.norm(rel_xy)
        if distance > float(max_distance):
            continue

        obj_lane_id = None
        if vehicle_current_lane_ids is not None and obj_idx < len(vehicle_current_lane_ids):
            obj_seed_idx = closest_lane_points[obj_idx]
            obj_lane_id = vehicle_current_lane_ids[obj_idx]
        else:
            if right_lane_point_indices and lane_points_global is not None:
                d2 = jnp.sum((right_lane_xy - obj_xy[None, :]) ** 2, axis=-1)
                min_idx = int(jnp.argmin(d2))
                min_distance = float(d2[min_idx] ** 0.5)
                obj_seed_global = int(right_lane_point_indices[min_idx])
                lane_dir = lane_points_global[obj_seed_global, 2:4]
                lane_heading = jnp.arctan2(lane_dir[1], lane_dir[0])
                obj_heading = object_yaw[obj_idx, t]
                seed_heading_delta = obj_heading - lane_heading
                seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
                if jnp.abs(seed_heading_delta_wrapped) > jnp.asarray(seed_heading_threshold_rad, dtype=jnp.float32) or min_distance > 2.0:
                    obj_lane_id = -1
                else:
                    obj_lane_id = lane_id_from_seed(scorer, obj_seed_global)
            else:
                obj_seed_idx = get_closest_lane_point(
                    scorer=scorer,
                    sim_state=sim_state,
                    timestep=timestep,
                    world_idx=world_idx,
                    target_vehicle=obj_idx,
                    seed_heading_threshold_rad=seed_heading_threshold_rad,
                )
                obj_lane_id = lane_id_from_seed(scorer, obj_seed_idx)

        if obj_lane_id in right_lane_ids:
            vehicles_right.append((obj_idx, distance))

    return vehicles_right

def get_pedestrian_front(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    max_distance=30,
):
    ego_idx = get_ego_idx(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    pedestrian_mask = jnp.asarray(get_pedestrian_mask(sim_state, world_idx)).astype(jnp.bool_)

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)

    num_objects = int(object_xy.shape[0])
    pedestrians_front = []

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx:
            continue
        if not bool(pedestrian_mask[obj_idx]):
            continue
        if not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        distance = jnp.linalg.norm(rel_xy)
        if distance > float(max_distance):
            continue
        if float(jnp.dot(rel_xy, ego_forward)) <= 0.0:
            continue

        pedestrians_front.append((obj_idx, distance))

    return pedestrians_front


def get_traffic_light_state_ahead(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    successor_branch: str = "straight",
):
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return -1
    if scorer._lane_graph is None:
        return -1

    ego_idx = get_ego_idx(sim_state, world_idx)
    ego_seed_idx = get_closest_lane_point(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        world_idx=world_idx,
        target_vehicle=ego_idx,
        seed_heading_threshold_rad=seed_heading_threshold_rad,
    )
    ego_lane_id = lane_id_from_seed(scorer, ego_seed_idx)
    if ego_lane_id is None:
        return -1

    successor_lane_id = choose_best_connected_lane(
        scorer=scorer,
        lane_id=ego_lane_id,
        candidates=[int(x) for x in scorer._successor_lane_ids.get(int(ego_lane_id), [])],
        direction="forward",
        successor_branch=successor_branch,
    )
    if successor_lane_id is None:
        return -1

    tls = sim_state.log_traffic_light
    tl_state = jnp.asarray(tls.state)
    tl_lane_ids = jnp.asarray(tls.lane_ids)
    tl_valid = jnp.asarray(tls.valid).astype(jnp.bool_)

    if tl_state.ndim == 3:
        tl_state = tl_state[world_idx]
        tl_lane_ids = tl_lane_ids[world_idx]
        tl_valid = tl_valid[world_idx]
    elif tl_state.ndim != 2:
        return -1

    if tl_state.ndim != 2 or tl_lane_ids.ndim != 2 or tl_valid.ndim != 2:
        return -1

    max_t = int(tl_state.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    lane_mask = tl_valid[:, t] & (jnp.asarray(tl_lane_ids[:, t]) == jnp.asarray(successor_lane_id))
    if not bool(jnp.any(lane_mask)):
        return -1

    matching_states = jnp.asarray(tl_state[:, t])[lane_mask]
    if int(matching_states.shape[0]) == 0:
        return -1
    return int(matching_states[0])


def is_ahead_of(
    scorer: Any,
    sim_state,
    timestep,
    target_vehicle,
    world_idx=0,
    min_longitudinal_distance_m=0.0,
):
    ego_idx = get_ego_idx(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

    if target_vehicle is None:
        return False

    target_idx = int(target_vehicle)
    if target_idx == ego_idx:
        return False

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))
    if not bool(object_valid[ego_idx, t]) or not bool(object_valid[target_idx, t]):
        return False

    ego_xy = object_xy[ego_idx, t]
    target_xy = object_xy[target_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    target_heading = object_yaw[target_idx, t]

    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    target_forward = jnp.asarray([jnp.cos(target_heading), jnp.sin(target_heading)], dtype=jnp.float32)
    relative_xy = target_xy - ego_xy
    ego_to_target = jnp.dot(relative_xy, ego_forward)
    target_to_ego = jnp.dot(-relative_xy, target_forward)
    return bool(
        ego_to_target < jnp.asarray(min_longitudinal_distance_m, dtype=jnp.float32)
    ) and bool(
        target_to_ego > jnp.asarray(min_longitudinal_distance_m, dtype=jnp.float32)
    )


def is_behind_of(
    scorer: Any,
    sim_state,
    timestep,
    target_vehicle,
    world_idx=0,
    min_longitudinal_distance_m=0.0,
):
    return not is_ahead_of(
        scorer=scorer,
        sim_state=sim_state,
        timestep=timestep,
        target_vehicle=target_vehicle,
        world_idx=world_idx,
        min_longitudinal_distance_m=min_longitudinal_distance_m,
    )


def on_target_lane(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    threshold_distance_m=1.0,
):
    if scorer.target_lane is None:
        return False

    target_lane = jnp.asarray(scorer.target_lane, dtype=jnp.float32)
    if target_lane.ndim != 2 or target_lane.shape[0] == 0 or target_lane.shape[1] < 2:
        raise ValueError(
            "target_lane must have shape [num_points, >=2] with x,y columns, "
            f"got {target_lane.shape}"
        )

    ego_idx = get_ego_idx(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))
    if not bool(object_valid[ego_idx, t]):
        return False

    ego_xy = object_xy[ego_idx, t]
    target_xy = target_lane[:, :2]
    min_dist2 = jnp.min(jnp.sum((target_xy - ego_xy[None, :]) ** 2, axis=-1))
    return bool(min_dist2 <= (jnp.asarray(threshold_distance_m, dtype=jnp.float32) ** 2))


def get_current_speed(scorer: Any, sim_state, timestep, world_idx=0, target_vehicle="ego"):
    ego_idx = get_ego_idx(sim_state, world_idx)
    target_idx = ego_idx if target_vehicle == "ego" else int(target_vehicle)

    object_vel_x = jnp.asarray(sim_state.log_trajectory.vel_x[world_idx], dtype=jnp.float32)
    object_vel_y = jnp.asarray(sim_state.log_trajectory.vel_y[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_vel_x.ndim != 2 or object_vel_y.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.vel_x and vel_y must have shape [num_objects, num_timesteps], "
            f"got {object_vel_x.shape} and {object_vel_y.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )
    max_t = int(object_vel_x.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    if not bool(object_valid[target_idx, t]):
        return None
    vel_x = object_vel_x[target_idx, t]
    vel_y = object_vel_y[target_idx, t]
    speed = jnp.sqrt(vel_x ** 2 + vel_y ** 2)
    return float(speed)

def get_relative_position(scorer: Any, sim_state, timestep, target_vehicle, world_idx=0):
    ego_idx = get_ego_idx(sim_state, world_idx)
    target_idx = int(target_vehicle)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_heading = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))
    if not bool(object_valid[ego_idx, t]) or not bool(object_valid[target_idx, t]):
        return None

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_heading[ego_idx, t]
    target_xy = object_xy[target_idx, t]
    # assume the relative position is in the ego's local frame, with x forward and y left
    cos_h = jnp.cos(ego_heading)
    sin_h = jnp.sin(ego_heading)
    rot_matrix = jnp.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=jnp.float32)
    rel_xy = target_xy - ego_xy
    rel_xy = rot_matrix @ rel_xy
    return tuple(rel_xy.tolist())

def get_relative_heading(scorer: Any, sim_state, timestep, target_vehicle, world_idx=0):
    ego_idx = get_ego_idx(sim_state, world_idx)
    target_idx = int(target_vehicle)

    object_heading = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_heading.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.yaw must have shape [num_objects, num_timesteps], "
            f"got {object_heading.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_heading.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))
    if not bool(object_valid[ego_idx, t]) or not bool(object_valid[target_idx, t]):
        return None

    ego_heading = object_heading[ego_idx, t]
    target_heading = object_heading[target_idx, t]
    rel_heading = target_heading - ego_heading
    rel_heading = (rel_heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return float(rel_heading)

def get_relative_goal_xy(scorer: Any, sim_state, timestep, world_idx=0):
    ego_idx = get_ego_idx(sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_heading = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_heading[ego_idx, t]
    goal_xy = object_xy[ego_idx, max_t]
    rel_goal_xy = goal_xy - ego_xy
    cos_h = jnp.cos(ego_heading)
    sin_h = jnp.sin(ego_heading)
    rot_matrix = jnp.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=jnp.float32)
    rel_goal_xy = rot_matrix @ rel_goal_xy

    return tuple(rel_goal_xy.tolist())

def shift_lane_right(scorer: Any, lane, shift_distance_m=3.5):
    del scorer
    lane = jnp.asarray(lane, dtype=jnp.float32)
    if lane.ndim != 2 or lane.shape[1] < 4:
        raise ValueError(
            "lane must have shape [num_points, >=4] with x,y,dir_x,dir_y columns, "
            f"got {lane.shape}"
        )
    if int(lane.shape[0]) == 0:
        return lane

    shifted_lane = lane.copy()
    dir_xy = shifted_lane[:, 2:4]
    dir_norm = jnp.maximum(jnp.linalg.norm(dir_xy, axis=-1, keepdims=True), 1e-6)
    dir_unit = dir_xy / dir_norm

    right_vec = jnp.stack([dir_unit[:, 1], -dir_unit[:, 0]], axis=-1)
    shifted_xy = shifted_lane[:, :2] + jnp.asarray(shift_distance_m, dtype=jnp.float32) * right_vec
    shifted_lane = shifted_lane.at[:, :2].set(shifted_xy)
    return shifted_lane