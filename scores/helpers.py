from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from waymax.datatypes.roadgraph import MapElementIds


CENTERLINE_TYPES = {
    int(MapElementIds.LANE_FREEWAY.value),
    int(MapElementIds.LANE_SURFACE_STREET.value),
}


def get_ego_idx(scorer: Any, sim_state, world_idx):
    ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
    if int(jnp.sum(ego_mask)) != 1:
        raise ValueError(
            f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}."
        )
    return int(jnp.argmax(ego_mask.astype(jnp.int32)))


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
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        raise ValueError("lane_points is empty. Call update_lane_points with a valid scenario first.")

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_yaw.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.yaw must have shape [num_objects, num_timesteps], "
            f"got {object_yaw.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    if target_vehicle == "ego":
        target_idx = get_ego_idx(scorer, sim_state, world_idx)
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


def extend_lane_points(scorer: Any, seed_idx, ray_distance_threshold_m=0.5):
    lane_points = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
    if lane_points.ndim != 2 or lane_points.shape[1] < 4:
        raise ValueError(
            "lane_points must have shape [num_points, >=4] with x,y,dir_x,dir_y columns, "
            f"got {lane_points.shape}"
        )
    if int(lane_points.shape[0]) == 0:
        return jnp.empty((0, 4), dtype=jnp.float32)

    seed_idx = int(seed_idx)
    if seed_idx < 0 or seed_idx >= int(lane_points.shape[0]):
        raise IndexError(f"seed_idx out of range: {seed_idx} for num_lane_points={int(lane_points.shape[0])}")

    lane_xy = lane_points[:, :2]
    current_lane_indices = [seed_idx]
    visited_indices = {seed_idx}
    max_hops = int(lane_points.shape[0])

    for _ in range(max_hops):
        cur_idx = current_lane_indices[-1]
        cur_xy = lane_xy[cur_idx]
        cur_dir = lane_points[cur_idx, 2:4]
        cur_dir_norm = jnp.maximum(jnp.linalg.norm(cur_dir), 1e-6)
        ray_dir = cur_dir / cur_dir_norm

        rel_xy = lane_xy - cur_xy[None, :]
        proj = jnp.sum(rel_xy * ray_dir[None, :], axis=-1)
        rel_norm2 = jnp.sum(rel_xy * rel_xy, axis=-1)
        perp2 = jnp.maximum(rel_norm2 - proj * proj, 0.0)
        perp = jnp.sqrt(perp2)

        visited_mask = jnp.zeros((lane_points.shape[0],), dtype=jnp.bool_)
        if visited_indices:
            visited_arr = jnp.asarray(sorted(visited_indices), dtype=jnp.int32)
            visited_mask = visited_mask.at[visited_arr].set(True)

        valid_forward = proj > 1e-5
        valid_ray = perp < ray_distance_threshold_m
        candidate_mask = valid_forward & valid_ray & (~visited_mask)

        if not bool(jnp.any(candidate_mask)):
            break

        candidate_dir = lane_points[:, 2:4]
        candidate_dir_norm = jnp.maximum(
            jnp.linalg.norm(candidate_dir, axis=-1, keepdims=True), 1e-6
        )
        candidate_dir_unit = candidate_dir / candidate_dir_norm
        cur_heading = jnp.arctan2(ray_dir[1], ray_dir[0])
        cand_heading = jnp.arctan2(candidate_dir_unit[:, 1], candidate_dir_unit[:, 0])
        heading_delta = cur_heading - cand_heading
        wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
        direction_error = jnp.abs(wrapped_delta)

        large = jnp.asarray(1e9, dtype=jnp.float32)
        tie_break = 1e-3 * proj
        selection_score = jnp.where(candidate_mask, direction_error + tie_break, large)
        next_idx = int(jnp.argmin(selection_score))

        if not bool(candidate_mask[next_idx]):
            break

        current_lane_indices.append(next_idx)
        visited_indices.add(next_idx)

    lane_idx_arr = jnp.asarray(current_lane_indices, dtype=jnp.int32)
    return lane_points[lane_idx_arr, :4].astype(jnp.float32)


def get_current_lane_from_points(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
):
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return jnp.empty((0, 4), dtype=jnp.float32)
    seed_idx = get_closest_lane_point(
        scorer,
        sim_state,
        timestep,
        world_idx=world_idx,
        target_vehicle="ego",
        seed_heading_threshold_rad=seed_heading_threshold_rad,
    )
    return extend_lane_points(
        scorer, seed_idx, ray_distance_threshold_m=ray_distance_threshold_m
    )


def get_side_lane_from_points(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    side="left",
    max_side_distance_m=5.0,
    min_side_distance_m=0.5,
    max_longitudinal_distance_m=0.5,
    max_direction_delta_rad=jnp.pi / 6.0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
):
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return None

    seed_idx = get_closest_lane_point(
        scorer,
        sim_state,
        timestep,
        world_idx=world_idx,
        target_vehicle="ego",
        seed_heading_threshold_rad=seed_heading_threshold_rad,
    )

    lane_points = jnp.asarray(scorer.lane_points, dtype=jnp.float32)
    lane_xy = lane_points[:, :2]

    seed_xy = lane_xy[seed_idx]
    seed_dir = lane_points[seed_idx, 2:4]
    seed_dir_norm = jnp.maximum(jnp.linalg.norm(seed_dir), 1e-6)
    seed_dir_unit = seed_dir / seed_dir_norm

    left_vec = jnp.asarray([-seed_dir_unit[1], seed_dir_unit[0]], dtype=jnp.float32)
    rel_xy = lane_xy - seed_xy[None, :]
    longitudinal_proj = jnp.sum(rel_xy * seed_dir_unit[None, :], axis=-1)
    lateral_signed = jnp.sum(rel_xy * left_vec[None, :], axis=-1)
    lateral_dist = lateral_signed if side == "left" else -lateral_signed
    dist2 = jnp.sum(rel_xy * rel_xy, axis=-1)

    lane_ids = lane_points[:, 4]
    seed_lane_id = lane_ids[seed_idx]

    side_mask = lateral_dist >= jnp.asarray(min_side_distance_m, dtype=jnp.float32)
    side_mask = side_mask & (lateral_dist < jnp.asarray(max_side_distance_m, dtype=jnp.float32))
    side_mask = side_mask & (
        jnp.abs(longitudinal_proj) <= jnp.asarray(max_longitudinal_distance_m, dtype=jnp.float32)
    )
    side_mask = side_mask & (lane_ids != seed_lane_id)
    side_mask = side_mask.at[seed_idx].set(False)

    if not bool(jnp.any(side_mask)):
        return None

    large = jnp.asarray(1e12, dtype=jnp.float32)
    side_dist2 = jnp.where(side_mask, dist2, large)
    candidate_idx = int(jnp.argmin(side_dist2))
    candidate_dist = jnp.sqrt(dist2[candidate_idx])
    if float(candidate_dist) > float(max_side_distance_m):
        return None

    candidate_dir = lane_points[candidate_idx, 2:4]
    candidate_dir_norm = jnp.maximum(jnp.linalg.norm(candidate_dir), 1e-6)
    candidate_dir_unit = candidate_dir / candidate_dir_norm

    seed_heading = jnp.arctan2(seed_dir_unit[1], seed_dir_unit[0])
    candidate_heading = jnp.arctan2(candidate_dir_unit[1], candidate_dir_unit[0])
    heading_delta = seed_heading - candidate_heading
    wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
    direction_error = jnp.abs(wrapped_delta)
    if float(direction_error) > float(max_direction_delta_rad):
        return None

    return extend_lane_points(
        scorer, candidate_idx, ray_distance_threshold_m=ray_distance_threshold_m
    )


def unique_lane_ids(lane_ids: List[int]) -> List[int]:
    seen = set()
    out = []
    for lane_id in lane_ids:
        if lane_id in seen:
            continue
        seen.add(lane_id)
        out.append(lane_id)
    return out


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
    lane_chain = unique_lane_ids(lane_chain)
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

    if scorer._lane_graph is None:
        return get_current_lane_from_points(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            ray_distance_threshold_m=0.5,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )

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
    min_side_distance_m=0.5,
    max_longitudinal_distance_m=0.5,
    max_direction_delta_rad=jnp.pi / 6.0,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
    n_hops: int = 2,
    successor_branch: str = "straight",
):
    if scorer.lane_points is None:
        update_lane_points(scorer, sim_state, world_idx)
    if scorer.lane_points is None or int(scorer.lane_points.shape[0]) == 0:
        return None

    if scorer._lane_graph is None:
        return get_side_lane_from_points(
            scorer=scorer,
            sim_state=sim_state,
            timestep=timestep,
            world_idx=world_idx,
            side=side,
            max_side_distance_m=max_side_distance_m,
            min_side_distance_m=min_side_distance_m,
            max_longitudinal_distance_m=max_longitudinal_distance_m,
            max_direction_delta_rad=max_direction_delta_rad,
            ray_distance_threshold_m=ray_distance_threshold_m,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )

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
        max_direction_delta_rad=max_direction_delta_rad,
        ray_distance_threshold_m=ray_distance_threshold_m,
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
        max_direction_delta_rad=max_direction_delta_rad,
        ray_distance_threshold_m=ray_distance_threshold_m,
        seed_heading_threshold_rad=seed_heading_threshold_rad,
        n_hops=n_hops,
        successor_branch=successor_branch,
    )


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


def get_vehicle_front(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    same_lane_distance_threshold_m=2.0,
    max_distance=50,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
):
    del ray_distance_threshold_m
    ego_idx = get_ego_idx(scorer, sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_yaw.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.yaw must have shape [num_objects, num_timesteps], "
            f"got {object_yaw.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    ego_left = jnp.asarray([-ego_forward[1], ego_forward[0]], dtype=jnp.float32)

    num_objects = int(object_xy.shape[0])
    best_obj_idx = None
    best_distance = float("inf")
    same_lane_threshold = float(same_lane_distance_threshold_m)

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx:
            continue
        if not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        lateral_distance = jnp.abs(jnp.dot(rel_xy, ego_left))
        if float(lateral_distance) > same_lane_threshold:
            continue

        obj_heading = object_yaw[obj_idx, t]
        heading_delta = obj_heading - ego_heading
        wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
        if float(jnp.abs(wrapped_delta)) > float(seed_heading_threshold_rad):
            continue

        longitudinal = jnp.dot(rel_xy, ego_forward)
        if float(longitudinal) <= 0.0:
            continue

        distance = jnp.linalg.norm(rel_xy)
        if distance > float(max_distance):
            continue
        if distance < best_distance:
            best_distance = distance
            best_obj_idx = obj_idx

    return best_obj_idx


def get_vehicle_behind(
    scorer: Any,
    sim_state,
    timestep,
    world_idx=0,
    same_lane_distance_threshold_m=2.0,
    max_distance=50,
    ray_distance_threshold_m=0.5,
    seed_heading_threshold_rad=jnp.pi / 6.0,
):
    del ray_distance_threshold_m
    ego_idx = get_ego_idx(scorer, sim_state, world_idx)

    object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
    object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
    object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)
    if object_xy.ndim != 3 or object_xy.shape[-1] != 2:
        raise ValueError(
            "sim_state.log_trajectory.xy must have shape [num_objects, num_timesteps, 2], "
            f"got {object_xy.shape}"
        )
    if object_yaw.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.yaw must have shape [num_objects, num_timesteps], "
            f"got {object_yaw.shape}"
        )
    if object_valid.ndim != 2:
        raise ValueError(
            "sim_state.log_trajectory.valid must have shape [num_objects, num_timesteps], "
            f"got {object_valid.shape}"
        )

    max_t = int(object_xy.shape[1]) - 1
    t = int(jnp.clip(jnp.asarray(timestep), 0, max_t))

    ego_xy = object_xy[ego_idx, t]
    ego_heading = object_yaw[ego_idx, t]
    ego_forward = jnp.asarray([jnp.cos(ego_heading), jnp.sin(ego_heading)], dtype=jnp.float32)
    ego_left = jnp.asarray([-ego_forward[1], ego_forward[0]], dtype=jnp.float32)

    num_objects = int(object_xy.shape[0])
    best_obj_idx = None
    best_distance = float("inf")
    same_lane_threshold = float(same_lane_distance_threshold_m)

    for obj_idx in range(num_objects):
        if obj_idx == ego_idx:
            continue
        if not bool(object_valid[obj_idx, t]):
            continue

        obj_xy = object_xy[obj_idx, t]
        rel_xy = obj_xy - ego_xy
        lateral_distance = jnp.abs(jnp.dot(rel_xy, ego_left))
        if float(lateral_distance) > same_lane_threshold:
            continue

        obj_heading = object_yaw[obj_idx, t]
        heading_delta = obj_heading - ego_heading
        wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
        if float(jnp.abs(wrapped_delta)) > float(seed_heading_threshold_rad):
            continue

        longitudinal = jnp.dot(rel_xy, ego_forward)
        if float(longitudinal) >= 0.0:
            continue

        distance = jnp.linalg.norm(rel_xy)
        if distance > float(max_distance):
            continue
        if distance < best_distance:
            best_distance = distance
            best_obj_idx = obj_idx

    return best_obj_idx


def is_ahead_of(
    scorer: Any,
    sim_state,
    timestep,
    target_vehicle,
    world_idx=0,
    min_longitudinal_distance_m=0.0,
):
    ego_idx = get_ego_idx(scorer, sim_state, world_idx)

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

    ego_idx = get_ego_idx(scorer, sim_state, world_idx)

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
    ego_idx = get_ego_idx(scorer, sim_state, world_idx)
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
