from __future__ import annotations

import jax
from jax import numpy as jnp

from waymax.datatypes.roadgraph import MapElementIds
from .polygon import pairwise_intersections, polygon_corners

CENTERLINE_TYPES = {
    int(MapElementIds.LANE_FREEWAY.value),
    int(MapElementIds.LANE_SURFACE_STREET.value),
}
MULTIPLICATIVE_METRICS = {"collision", "offroad", "direction"}
WEIGHTED_METRICS = {"progress", "follow_lane", "set_speed", "overtake"}

class Scorer:
    def __init__(self, metrics=["collision", "offroad"], weights=None):
        self._multiplicative_metrics = {metric for metric in metrics if metric in MULTIPLICATIVE_METRICS}
        self._weighted_metrics = {metric: 1.0 for metric in metrics if metric in WEIGHTED_METRICS}
        if weights is not None:
            for metric, weight in weights.items():
                if metric in self._weighted_metrics:
                    self._weighted_metrics[metric] = float(weight)
        self.lane_points = None
        self.target_lane = None
        self.target_speed = None
        self.target_vehicle = None
        self.goal = None
        self.base_metrics = set(metrics)
        self.base_weights = weights

    def reset(self):
        self._multiplicative_metrics = {metric for metric in self.base_metrics if metric in MULTIPLICATIVE_METRICS}
        self._weighted_metrics = {metric: 1.0 for metric in self.base_metrics if metric in WEIGHTED_METRICS}
        if self.base_weights is not None:
            for metric, weight in self.base_weights.items():
                if metric in self._weighted_metrics:
                    self._weighted_metrics[metric] = float(weight)
        self.lane_points = None
        self.target_lane = None
        self.target_speed = None
        self.target_vehicle = None
        self.goal = None
        
    def add_metric(self, name, target=None, weight=1.0):
        if name in {"collision", "offroad", "direction"}:
            self._multiplicative_metrics.add(name)
        elif name == "follow_lane" and target is not None:
            self._weighted_metrics[name] = float(weight)
            self.target_lane = target
        elif name == "set_speed" and target is not None:
            self._weighted_metrics[name] = float(weight)
            self.target_speed = float(target)
        elif name == "overtake" and target is not None:
            self._weighted_metrics[name] = float(weight)
            self.target_vehicle = target
        elif name == "goal" and target is not None:
            self.goal = target

    def remove_metric(self, name):
        if name in self._multiplicative_metrics:
            self._multiplicative_metrics.remove(name)
        elif name in self._weighted_metrics:
            del self._weighted_metrics[name]
            if name == "follow_lane":
                self.target_lane = None
            elif name == "set_speed":
                self.target_speed = None
            elif name == "overtake":
                self.target_vehicle = None
            elif name == "goal":
                self.goal = None

    def get_ego_idx(self, sim_state, world_idx):
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        return int(jnp.argmax(ego_mask.astype(jnp.int32)))

    def update_lane_points(self, sim_state, world_idx=0):
        roadgraph_points = getattr(sim_state, "roadgraph_points", None)
        if roadgraph_points is None:
            raise ValueError("sim_state must include roadgraph_points")

        x = jnp.asarray(roadgraph_points.x)
        y = jnp.asarray(roadgraph_points.y)
        dir_x = jnp.asarray(roadgraph_points.dir_x)
        dir_y = jnp.asarray(roadgraph_points.dir_y)
        ids = jnp.asarray(roadgraph_points.ids)
        types = jnp.asarray(roadgraph_points.types)
        valid = jnp.asarray(roadgraph_points.valid).astype(jnp.bool_)

        if x.ndim > 1:
            world_idx = int(world_idx)
            x = x[world_idx]
            y = y[world_idx]
            dir_x = dir_x[world_idx]
            dir_y = dir_y[world_idx]
            ids = ids[world_idx]
            types = types[world_idx]
            valid = valid[world_idx]

        centerline_mask = jnp.isin(
            types,
            jnp.asarray(sorted(CENTERLINE_TYPES), dtype=types.dtype),
        )
        mask = valid & centerline_mask

        self.lane_points = jnp.stack(
            [
                x[mask].astype(jnp.float32),
                y[mask].astype(jnp.float32),
                dir_x[mask].astype(jnp.float32),
                dir_y[mask].astype(jnp.float32),
                ids[mask].astype(jnp.float32),
            ],
            axis=-1,
        )

    def get_closest_lane_point(
        self,
        sim_state,
        timestep,
        world_idx=0,
        target_vehicle="ego",
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        if self.lane_points is None:
            self.update_lane_points(sim_state, world_idx)
        if self.lane_points is None or int(self.lane_points.shape[0]) == 0:
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
            ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
            if int(jnp.sum(ego_mask)) != 1:
                raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
            target_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))
        else:
            target_idx = int(target_vehicle)
            if target_idx < 0 or target_idx >= int(object_xy.shape[0]):
                raise IndexError(
                    f"target_vehicle index out of range: {target_idx} for num_objects={int(object_xy.shape[0])}"
                )

        target_xy = object_xy[target_idx, t]
        target_heading = object_yaw[target_idx, t]

        lane_points = jnp.asarray(self.lane_points, dtype=jnp.float32)
        lane_xy = lane_points[:, :2]
        lane_dir = lane_points[:, 2:4]
        lane_heading = jnp.arctan2(lane_dir[:, 1], lane_dir[:, 0])

        seed_heading_delta = target_heading - lane_heading
        seed_heading_delta_wrapped = jnp.arctan2(jnp.sin(seed_heading_delta), jnp.cos(seed_heading_delta))
        seed_heading_ok = jnp.abs(seed_heading_delta_wrapped) <= jnp.asarray(seed_heading_threshold_rad, dtype=jnp.float32)

        dist2 = jnp.sum((lane_xy - target_xy[None, :]) ** 2, axis=-1)
        large = jnp.asarray(1e12, dtype=jnp.float32)
        seed_dist2 = jnp.where(seed_heading_ok, dist2, large)
        if bool(jnp.any(seed_heading_ok)):
            return int(jnp.argmin(seed_dist2))
        return int(jnp.argmin(dist2))
    
    def extend_lane_points(self, seed_idx, ray_distance_threshold_m=0.5):
        lane_points = jnp.asarray(self.lane_points, dtype=jnp.float32)
        if lane_points.ndim != 2 or lane_points.shape[1] < 4:
            raise ValueError(
                "lane_points must have shape [num_points, >=4] with x,y,dir_x,dir_y columns, "
                f"got {lane_points.shape}"
            )
        if int(lane_points.shape[0]) == 0:
            return jnp.empty((0, 4), dtype=jnp.float32)

        seed_idx = int(seed_idx)
        if seed_idx < 0 or seed_idx >= int(lane_points.shape[0]):
            raise IndexError(
                f"seed_idx out of range: {seed_idx} for num_lane_points={int(lane_points.shape[0])}"
            )

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
            candidate_dir_norm = jnp.maximum(jnp.linalg.norm(candidate_dir, axis=-1, keepdims=True), 1e-6)
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

    def get_current_lane(
        self,
        sim_state,
        timestep,
        world_idx=0,
        ray_distance_threshold_m=0.5,
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        if self.lane_points is None:
            self.update_lane_points(sim_state, world_idx)
        if self.lane_points is None or int(self.lane_points.shape[0]) == 0:
            return jnp.empty((0, 4), dtype=jnp.float32)

        seed_idx = self.get_closest_lane_point(
            sim_state,
            timestep,
            world_idx=world_idx,
            target_vehicle="ego",
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )
        return self.extend_lane_points(seed_idx, ray_distance_threshold_m=ray_distance_threshold_m)

    def _get_side_lane(
        self,
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

        if self.lane_points is None:
            self.update_lane_points(sim_state, world_idx)
        if self.lane_points is None or int(self.lane_points.shape[0]) == 0:
            return None

        seed_idx = self.get_closest_lane_point(
            sim_state,
            timestep,
            world_idx=world_idx,
            target_vehicle="ego",
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )

        lane_points = jnp.asarray(self.lane_points, dtype=jnp.float32)
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
        side_mask = side_mask & (jnp.abs(longitudinal_proj) <= jnp.asarray(max_longitudinal_distance_m, dtype=jnp.float32))
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

        return self.extend_lane_points(candidate_idx, ray_distance_threshold_m=ray_distance_threshold_m)

    def get_left_lane(
        self,
        sim_state,
        timestep,
        world_idx=0,
        max_side_distance_m=6.0,
        max_direction_delta_rad=jnp.pi / 6.0,
        ray_distance_threshold_m=0.5,
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        return self._get_side_lane(
            sim_state,
            timestep,
            world_idx=world_idx,
            side="left",
            max_side_distance_m=max_side_distance_m,
            max_direction_delta_rad=max_direction_delta_rad,
            ray_distance_threshold_m=ray_distance_threshold_m,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )

    def get_right_lane(
        self,
        sim_state,
        timestep,
        world_idx=0,
        max_side_distance_m=6.0,
        max_direction_delta_rad=jnp.pi / 6.0,
        ray_distance_threshold_m=0.5,
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        return self._get_side_lane(
            sim_state,
            timestep,
            world_idx=world_idx,
            side="right",
            max_side_distance_m=max_side_distance_m,
            max_direction_delta_rad=max_direction_delta_rad,
            ray_distance_threshold_m=ray_distance_threshold_m,
            seed_heading_threshold_rad=seed_heading_threshold_rad,
        )

    def shift_lane_right(self, lane, shift_distance_m=3.5):
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
        self,
        sim_state,
        timestep,
        world_idx=0,
        same_lane_distance_threshold_m=2.0,
        max_distance=50,
        ray_distance_threshold_m=0.5,
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))

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
        ego_forward = jnp.asarray(
            [jnp.cos(ego_heading), jnp.sin(ego_heading)],
            dtype=jnp.float32,
        )
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
        self,
        sim_state,
        timestep,
        world_idx=0,
        same_lane_distance_threshold_m=2.0,
        max_distance=50,
        ray_distance_threshold_m=0.5,
        seed_heading_threshold_rad=jnp.pi / 6.0,
    ):
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))

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
        ego_forward = jnp.asarray(
            [jnp.cos(ego_heading), jnp.sin(ego_heading)],
            dtype=jnp.float32,
        )
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
        self,
        sim_state,
        timestep,
        target_vehicle,
        world_idx=0,
        min_longitudinal_distance_m=0.0,
    ):
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))

        object_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx], dtype=jnp.float32)
        object_yaw = jnp.asarray(sim_state.log_trajectory.yaw[world_idx], dtype=jnp.float32)
        object_valid = jnp.asarray(sim_state.log_trajectory.valid[world_idx]).astype(jnp.bool_)

        if target_vehicle is None:
            return False
        
        target_idx = int(target_vehicle)
        num_objects = int(object_xy.shape[0])
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

        ego_forward = jnp.asarray(
            [jnp.cos(ego_heading), jnp.sin(ego_heading)],
            dtype=jnp.float32,
        )
        target_forward = jnp.asarray(
            [jnp.cos(target_heading), jnp.sin(target_heading)],
            dtype=jnp.float32,
        )
        relative_xy = target_xy - ego_xy
        ego_to_target = jnp.dot(relative_xy, ego_forward)
        target_to_ego = jnp.dot(-relative_xy, target_forward)
        return bool(ego_to_target < jnp.asarray(min_longitudinal_distance_m, dtype=jnp.float32)) and bool(target_to_ego > jnp.asarray(min_longitudinal_distance_m, dtype=jnp.float32))
    
    def is_behind_of(
        self,
        sim_state,
        timestep,
        target_vehicle,
        world_idx=0,
        min_longitudinal_distance_m=0.0,
    ):
        return not self.is_ahead_of(
            sim_state=sim_state,
            timestep=timestep,
            target_vehicle=target_vehicle,
            world_idx=world_idx,
            min_longitudinal_distance_m=min_longitudinal_distance_m,
        )

    def on_target_lane(
        self,
        sim_state,
        timestep,
        world_idx=0,
        threshold_distance_m=1.0
    ):
        if self.target_lane is None:
            return False

        target_lane = jnp.asarray(self.target_lane, dtype=jnp.float32)
        if target_lane.ndim != 2 or target_lane.shape[0] == 0 or target_lane.shape[1] < 2:
            raise ValueError(
                "target_lane must have shape [num_points, >=2] with x,y columns, "
                f"got {target_lane.shape}"
            )

        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))

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

    def compute_overtake_score(self, ego_trajectories, object_trajectories):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
        object_trajectories: [num_objects, num_time_steps, 5 (x, y, heading, length, width)]
        target_vehicle_idx: int index of the target vehicle to check overtaking against
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        object_trajectories = jnp.asarray(object_trajectories)

        ego_xy = ego_trajectories[:, -1, :2]
        object_xy = object_trajectories[:, -1, :2]
        object_heading = object_trajectories[:, -1, 2]

        rel_xy =  ego_xy - object_xy[self.target_vehicle]
        longitudinal = rel_xy[:, 0] * jnp.cos(object_heading[self.target_vehicle]) + rel_xy[:, 1] * jnp.sin(object_heading[self.target_vehicle])
        longitudinal = jnp.clip(longitudinal, a_min=-50.0, a_max=10.0)
        score = (longitudinal + 50.0) / 60.0    # Normalize to [0, 1], where 1 means 10m ahead or more, and 0 means 50m behind or more
        return jnp.clip(score, a_min=0.0, a_max=1.0)

    def compute_collision_score(self, ego_trajectories, object_trajectories, object_masks):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
        object_trajectories: [num_objects, num_time_steps, 5 (x, y, heading, length, width)]
        object_masks: [num_objects, num_time_steps] or [num_time_steps, num_objects]
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        object_trajectories = jnp.asarray(object_trajectories)
        object_masks = jnp.asarray(object_masks)

        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        if object_trajectories.ndim != 3 or object_trajectories.shape[-1] != 5:
            raise ValueError(
                "object_trajectories must have shape [num_objects, num_time_steps, 5], "
                f"got {object_trajectories.shape}"
            )

        num_objects = object_trajectories.shape[0]
        num_time_steps = object_trajectories.shape[1]
        if ego_trajectories.shape[1] != num_time_steps:
            raise ValueError(
                "ego_trajectories and object_trajectories must share num_time_steps, "
                f"got {ego_trajectories.shape[1]} and {num_time_steps}"
            )

        if object_masks.ndim != 2:
            raise ValueError(
                "object_masks must have shape [num_objects, num_time_steps] or [num_time_steps, num_objects], "
                f"got {object_masks.shape}"
            )
        if object_masks.shape == (num_objects, num_time_steps):
            object_masks_o_t = object_masks > 0
        elif object_masks.shape == (num_time_steps, num_objects):
            object_masks_o_t = jnp.swapaxes(object_masks, 0, 1) > 0
        else:
            raise ValueError(
                "object_masks shape mismatch: "
                f"got {object_masks.shape}, expected ({num_objects}, {num_time_steps}) "
                f"or ({num_time_steps}, {num_objects})"
            )

        ego_polygons = polygon_corners(ego_trajectories)
        object_polygons = polygon_corners(object_trajectories)

        def _time_pairwise(ego_t: jax.Array, object_t: jax.Array) -> jax.Array:
            return pairwise_intersections(ego_t, object_t)

        intersections_t_e_o = jax.vmap(_time_pairwise, in_axes=(1, 1), out_axes=0)(
            ego_polygons,
            object_polygons,
        )
        valid_mask_t_o = jnp.swapaxes(object_masks_o_t, 0, 1)
        intersections_t_e_o = intersections_t_e_o & valid_mask_t_o[:, None, :]
        collision_flags = jnp.any(intersections_t_e_o, axis=(0, 2))

        return jnp.where(collision_flags, 0.0, 1.0)
    
    def compute_offroad_score(self, ego_trajectories, threshold=3.0):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        if self.lane_points is None:
            raise ValueError("lane_points is not initialized. Call update_lane_points first.")
        if int(self.lane_points.shape[0]) == 0:
            return jnp.zeros((ego_trajectories.shape[0],), dtype=jnp.float32)

        ego_xy = ego_trajectories[..., :2]
        lane_xy = jnp.asarray(self.lane_points[:, :2], dtype=jnp.float32)

        squared_distances = jnp.sum(
            (ego_xy[:, :, None, :] - lane_xy[None, None, :, :]) ** 2,
            axis=-1,
        )
        min_squared_distance_per_t = jnp.min(squared_distances, axis=-1)
        offroad_flags = jnp.any(min_squared_distance_per_t >= (threshold ** 2), axis=1)
        return jnp.where(offroad_flags, 0.0, 1.0)
    
    def compute_direction_score(self, ego_trajectories):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
        Returns 0.0 if any timestep has heading difference >= pi/2 to nearest lane direction, else 1.0.
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        if self.lane_points is None:
            raise ValueError("lane_points is not initialized. Call update_lane_points first.")
        if int(self.lane_points.shape[0]) == 0:
            return jnp.zeros((ego_trajectories.shape[0],), dtype=jnp.float32)

        ego_xy = ego_trajectories[..., :2]
        ego_heading = ego_trajectories[..., 2]

        lane_xy = jnp.asarray(self.lane_points[:, :2], dtype=jnp.float32)
        lane_dir_xy = jnp.asarray(self.lane_points[:, 2:4], dtype=jnp.float32)

        squared_distances = jnp.sum(
            (ego_xy[:, :, None, :] - lane_xy[None, None, :, :]) ** 2,
            axis=-1,
        )
        nearest_lane_idx = jnp.argmin(squared_distances, axis=-1)
        nearest_lane_dir_xy = lane_dir_xy[nearest_lane_idx]
        nearest_lane_heading = jnp.arctan2(nearest_lane_dir_xy[..., 1], nearest_lane_dir_xy[..., 0])

        heading_delta = ego_heading - nearest_lane_heading
        wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
        wrong_direction_flags = jnp.abs(wrapped_delta) >= (jnp.pi / 2.0)
        any_wrong_direction = jnp.any(wrong_direction_flags, axis=1)
        return jnp.where(any_wrong_direction, 0.0, 1.0)
    
    def compute_lane_score(self, ego_trajectories):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]

        Per timestep score:
        - distance >= 3.5 -> 0
        - distance <= 1.0 -> 1
        - otherwise linearly interpolated between 1 and 0

        Returns mean score over timesteps for each ego trajectory.
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        if self.target_lane is None:
            return jnp.ones((ego_trajectories.shape[0],), dtype=jnp.float32)

        def _estimate_heading(xy: jax.Array) -> jax.Array:
            if xy.ndim != 3 or xy.shape[-1] != 2:
                raise ValueError(f"xy must have shape [batch, timesteps, 2], got {xy.shape}")
            if int(xy.shape[1]) <= 1:
                return jnp.zeros((xy.shape[0], xy.shape[1]), dtype=jnp.float32)
            delta = xy[:, 1:, :] - xy[:, :-1, :]
            heading_tail = jnp.arctan2(delta[..., 1], delta[..., 0])
            heading = jnp.concatenate([heading_tail[:, :1], heading_tail], axis=1)
            return heading.astype(jnp.float32)

        target_lane = jnp.asarray(self.target_lane, dtype=jnp.float32)
        if target_lane.ndim != 2 or target_lane.shape[0] == 0 or target_lane.shape[1] < 2:
            raise ValueError(
                "target_lane must have shape [num_points, >=2] with x,y columns, "
                f"got {target_lane.shape}"
            )

        trajectories = ego_trajectories[:, ::5, :2]
        trajectory_heading = _estimate_heading(trajectories)

        max_lane_deviation = jnp.asarray(10.0, dtype=jnp.float32)

        centerline = target_lane[::5] if int(target_lane.shape[0]) > 50 else target_lane
        centerline_xy = centerline[:, :2]
        centerline_heading = _estimate_heading(centerline_xy[None, ...])[0]

        xy_loss = jnp.linalg.norm(
            trajectories[:, None, :, :] - centerline_xy[None, :, None, :],
            axis=-1,
        )
        heading_loss = -jnp.cos(centerline_heading[None, :, None] - trajectory_heading[:, None, :])

        closest_idx = jnp.argmin(xy_loss, axis=1, keepdims=True)
        xy_loss_min = jnp.min(xy_loss, axis=1)
        heading_loss_sel = jnp.take_along_axis(heading_loss, closest_idx, axis=1).squeeze(1)

        position_score = jnp.clip(max_lane_deviation - xy_loss_min, 0.0, max_lane_deviation) / max_lane_deviation
        heading_score = 1.0 - ((heading_loss_sel + 1.0) / 2.0)
        score_t = (position_score + heading_score) / 2.0
        final_score = jnp.mean(score_t, axis=1)
        return final_score.astype(jnp.float32)
    
    def compute_speed_score(self, ego_trajectories):
        '''
        ego_trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
        Returns mean score over timesteps for each ego trajectory.
        '''
        ego_trajectories = jnp.asarray(ego_trajectories)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        if self.target_speed is None:
            return jnp.ones((ego_trajectories.shape[0],), dtype=jnp.float32)

        speeds = jnp.linalg.norm(ego_trajectories[..., 3:5], axis=-1)

        def _speed_score(speed: jax.Array) -> jax.Array:
            delta = speed - jnp.asarray(self.target_speed, dtype=jnp.float32)
            score = 1.0 - jnp.minimum(jnp.abs(delta) / 25.0, 1.0)
            return score.astype(jnp.float32)

        score_t = _speed_score(speeds)
        final_score = jnp.mean(score_t, axis=1)
        return final_score.astype(jnp.float32)
    
    def compute_progress_score(self, ego_trajectories):
        ego_trajectories = jnp.asarray(ego_trajectories)
        progress = jnp.linalg.norm(ego_trajectories[:, -1, :2] - ego_trajectories[:, 0, :2], axis=-1)
        return progress / 100.0
    
    def compute_goal_score(self, ego_trajectories):
        ego_trajectories = jnp.asarray(ego_trajectories)
        if self.goal is None:
            return jnp.ones((ego_trajectories.shape[0],), dtype=jnp.float32)
        goal = jnp.asarray(self.goal, dtype=jnp.float32)
        if goal.shape != (2,):
            raise ValueError(f"goal must have shape (2,), got {goal.shape}")
        traj_xy = ego_trajectories[:, :, :2]
        dist_to_goal = jnp.linalg.norm(traj_xy - goal[None, None, :], axis=-1)
        dist_to_goal = jnp.mean(dist_to_goal, axis=1)
        score = jnp.clip(50.0 - dist_to_goal, 0.0, 50.0) / 50.0
        return score.astype(jnp.float32)

    
    def compute_score(self, trajectories, sim_state, timestep, world_idx=0):
        '''
        trajectories: [num_ego_trajectories, num_time_steps, 5 (x, y, heading, vx, vy)]
        '''
        trajectories = jnp.asarray(trajectories)
        if trajectories.ndim != 3 or trajectories.shape[-1] != 5:
            raise ValueError(
                "trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {trajectories.shape}"
            )
        if self.lane_points is None:
            self.update_lane_points(sim_state, world_idx)

        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        object_xy = sim_state.log_trajectory.xy[world_idx]
        object_yaw = sim_state.log_trajectory.yaw[world_idx]
        object_length = sim_state.log_trajectory.length[world_idx]
        object_width = sim_state.log_trajectory.width[world_idx]
        object_valid = sim_state.log_trajectory.valid[world_idx]

        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")

        traj_length = trajectories.shape[1]
        max_available = object_xy.shape[1] - int(timestep)
        traj_length = min(int(traj_length), int(max_available))

        object_xy = object_xy[:, timestep:timestep + traj_length, :]
        object_yaw = object_yaw[:, timestep:timestep + traj_length]
        object_length = object_length[:, timestep:timestep + traj_length]
        object_width = object_width[:, timestep:timestep + traj_length]
        object_masks = object_valid[:, timestep:timestep + traj_length]
        object_trajectories = jnp.stack(
            [object_xy[..., 0], object_xy[..., 1], object_yaw, object_length, object_width],
            axis=-1,
        )
        object_masks = (object_masks > 0) & (~ego_mask[:, None])

        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))
        ego_length_t = object_length[ego_idx]
        ego_width_t = object_width[ego_idx]
        num_ego_trajectories = trajectories.shape[0]
        ego_length = jnp.broadcast_to(ego_length_t[None, :], (num_ego_trajectories, traj_length))
        ego_width = jnp.broadcast_to(ego_width_t[None, :], (num_ego_trajectories, traj_length))
        ego_trajectories = jnp.stack(
            [
                trajectories[:, :traj_length, 0], 
                trajectories[:, :traj_length, 1], 
                trajectories[:, :traj_length, 2], 
                ego_length, ego_width
            ], axis=-1
        )
        score = jnp.ones((num_ego_trajectories,), dtype=jnp.float32)
        if len(self._weighted_metrics) > 0:
            score = jnp.zeros((num_ego_trajectories,), dtype=jnp.float32)
            total_weight = sum(self._weighted_metrics.values())
            for metric_name, weight in self._weighted_metrics.items():
                if metric_name == "follow_lane":
                    lane_score = self.compute_lane_score(ego_trajectories)
                    score = score + weight * lane_score
                elif metric_name == "progress":
                    progress_score = self.compute_progress_score(ego_trajectories)
                    score = score + weight * progress_score
                elif metric_name == "set_speed":
                    # Registered by add_metric under the name "set_speed"; the
                    # branch previously matched "speed", so the metric was dead
                    # code (never scored, yet still diluting total_weight below).
                    speed_score = self.compute_speed_score(ego_trajectories)
                    score = score + weight * speed_score
                elif metric_name == "overtake":
                    overtake_score = self.compute_overtake_score(ego_trajectories, object_trajectories)
                    score = score + weight * overtake_score
                elif metric_name == "goal":
                    goal_score = self.compute_goal_score(ego_trajectories)
                    score = score + weight * goal_score
                score = score / total_weight
        if "collision" in self._multiplicative_metrics:
            collision_score = self.compute_collision_score(ego_trajectories, object_trajectories, object_masks)
            score = score * collision_score
        if "offroad" in self._multiplicative_metrics:
            offroad_score = self.compute_offroad_score(ego_trajectories)
            score = score * offroad_score
        if "direction" in self._multiplicative_metrics:
            direction_score = self.compute_direction_score(ego_trajectories)
            score = score * direction_score

        return score

    def get_current_speed(self, sim_state, timestep, world_idx=0, target_vehicle="ego"):
        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            raise ValueError(f"Expected exactly one SDC object, got {int(jnp.sum(ego_mask))}.")
        target_idx = int(jnp.argmax(ego_mask.astype(jnp.int32))) if target_vehicle == "ego" else int(target_vehicle)

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