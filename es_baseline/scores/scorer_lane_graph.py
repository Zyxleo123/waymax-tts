from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from waymax import datatypes

from . import helpers as scorer_helpers

from .polygon import pairwise_intersections, polygon_corners
from .lane_graph_utils import LaneGraphData, LaneGraphShardStore


MULTIPLICATIVE_METRICS = {"collision", "offroad", "direction", "tl_violation"}
WEIGHTED_METRICS = {"progress", "follow_lane", "set_speed", "overtake"}


class Scorer:
    """Scorer variant that supports fast lane queries from precomputed lane graph.

    This class provides the scorer API and uses lane-id relations
    (`left_neighbor_edges` / `right_neighbor_edges`) when lane graph data is available.
    """

    def __init__(self, metrics=["collision", "offroad", "tl_violation"], weights=None):
        self._multiplicative_metrics = {
            metric for metric in metrics if metric in MULTIPLICATIVE_METRICS
        }
        self._weighted_metrics = {
            metric: 1.0 for metric in metrics if metric in WEIGHTED_METRICS
        }
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

        self._lane_graph: Optional[LaneGraphData] = None
        self._lane_graph_shard_store: Optional[LaneGraphShardStore] = None
        self._scenario_indices: Optional[List[int]] = None
        self._scenario_index: Optional[int] = None
        self._lane_first_node_to_lane_id: Dict[int, int] = {}
        self._lane_last_node_to_lane_id: Dict[int, int] = {}
        self._successor_lane_ids: Dict[int, List[int]] = {}
        self._predecessor_lane_ids: Dict[int, List[int]] = {}
        self._left_neighbor_lane_ids: Dict[int, List[int]] = {}
        self._right_neighbor_lane_ids: Dict[int, List[int]] = {}

    def reset(self):
        self._multiplicative_metrics = {
            metric for metric in self.base_metrics if metric in MULTIPLICATIVE_METRICS
        }
        self._weighted_metrics = {
            metric: 1.0 for metric in self.base_metrics if metric in WEIGHTED_METRICS
        }
        if self.base_weights is not None:
            for metric, weight in self.base_weights.items():
                if metric in self._weighted_metrics:
                    self._weighted_metrics[metric] = float(weight)
        self.lane_points = None
        self.target_lane = None
        self.target_speed = None
        self.target_vehicle = None
        self.goal = None
        self._lane_graph = None
        self._lane_first_node_to_lane_id = {}
        self._lane_last_node_to_lane_id = {}
        self._successor_lane_ids = {}
        self._predecessor_lane_ids = {}
        self._left_neighbor_lane_ids = {}
        self._right_neighbor_lane_ids = {}

    def add_metric(self, name, target=None, weight=1.0):
        if name in MULTIPLICATIVE_METRICS:
            self._multiplicative_metrics.add(name)
        elif name == "follow_lane" and target is not None:
            self._weighted_metrics[name] = float(weight)
            self.target_lane = target
        elif name == "follow_lane_by_id" and target is not None:
            if isinstance(target, int) and target >= 0:
                lane_id = int(target)
                if self._lane_graph is None and self._lane_graph_shard_store is not None and self._scenario_index is not None:
                    scorer_helpers.load_lane_graph_by_index(self, int(self._scenario_index))

                lane_polyline = scorer_helpers.lane_polyline_from_lane_id(
                    self,
                    lane_id=lane_id,
                    n_hops=3,
                    successor_branch="straight",
                )
                if lane_polyline is None or int(lane_polyline.shape[0]) == 0:
                    raise ValueError(
                        f"Failed to build lane polyline from lane_id={lane_id}. "
                        "Ensure lane graph is loaded and the lane id exists."
                    )

                # Reuse the existing follow_lane scoring path by setting a lane pose target.
                self._weighted_metrics["follow_lane"] = float(weight)
                self.target_lane = lane_polyline
                self._weighted_metrics.pop("follow_lane_by_id", None)
            
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
        elif name == "follow_lane_by_id":
            self._weighted_metrics.pop("follow_lane", None)
            self.target_lane = None
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

    def set_lane_graph_shard_store(
        self,
        lane_graph_zip_path: str,
        scenario_indices: Optional[List[int]] = None,
        scenario_index: Optional[int] = None,
    ) -> None:
        """Configure one shard zip and optional scenario index mapping.

        Args:
            lane_graph_zip_path: Path to a single '*.lanegraph.zip' file.
            scenario_indices: Optional world-index to scenario-index mapping.
            scenario_index: Optional per-scorer scenario index.
        """
        self._lane_graph_shard_store = LaneGraphShardStore(lane_graph_zip_path)
        self._scenario_indices = [int(x) for x in scenario_indices] if scenario_indices is not None else None
        self._scenario_index = int(scenario_index) if scenario_index is not None else None

    def set_scenario_index(self, scenario_index: int) -> None:
        self._scenario_index = int(scenario_index)

    def set_lane_graph(self, lane_graph: LaneGraphData) -> None:
        self._lane_graph = lane_graph

        self._lane_first_node_to_lane_id = {
            int(start): int(lane_id)
            for lane_id, (start, _end) in lane_graph.lane_id_to_node_range.items()
        }
        self._lane_last_node_to_lane_id = {
            int(end - 1): int(lane_id)
            for lane_id, (_start, end) in lane_graph.lane_id_to_node_range.items()
            if int(end) > 0
        }

        self._successor_lane_ids = {}
        self._predecessor_lane_ids = {}
        self._left_neighbor_lane_ids = {}
        self._right_neighbor_lane_ids = {}

        for src_node, dst_node in lane_graph.successor_edges:
            src_lane_id = self._lane_last_node_to_lane_id.get(int(src_node))
            dst_lane_id = self._lane_first_node_to_lane_id.get(int(dst_node))
            if src_lane_id is None or dst_lane_id is None:
                continue
            self._successor_lane_ids.setdefault(src_lane_id, []).append(dst_lane_id)

        for src_node, dst_node in lane_graph.predecessor_edges:
            src_lane_id = self._lane_last_node_to_lane_id.get(int(src_node))
            dst_lane_id = self._lane_first_node_to_lane_id.get(int(dst_node))
            if src_lane_id is None or dst_lane_id is None:
                continue
            self._predecessor_lane_ids.setdefault(dst_lane_id, []).append(src_lane_id)

        for src_node, dst_node in lane_graph.left_neighbor_edges:
            src_lane_id = self._lane_first_node_to_lane_id.get(int(src_node))
            dst_lane_id = self._lane_first_node_to_lane_id.get(int(dst_node))
            if src_lane_id is None or dst_lane_id is None:
                continue
            self._left_neighbor_lane_ids.setdefault(src_lane_id, []).append(dst_lane_id)

        for src_node, dst_node in lane_graph.right_neighbor_edges:
            src_lane_id = self._lane_first_node_to_lane_id.get(int(src_node))
            dst_lane_id = self._lane_first_node_to_lane_id.get(int(dst_node))
            if src_lane_id is None or dst_lane_id is None:
                continue
            self._right_neighbor_lane_ids.setdefault(src_lane_id, []).append(dst_lane_id)

        nodes_xyz = jnp.asarray(lane_graph.nodes_xyz, dtype=jnp.float32)
        lane_ids = jnp.asarray(lane_graph.node_lane_ids, dtype=jnp.float32)

        dir_xy = jnp.zeros((nodes_xyz.shape[0], 2), dtype=jnp.float32)
        for lane_id, (start, end) in lane_graph.lane_id_to_node_range.items():
            if end - start <= 1:
                continue
            seg = nodes_xyz[start:end, :2]
            d = seg[1:] - seg[:-1]
            norms = jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            d_unit = d / norms
            dir_xy = dir_xy.at[start : end - 1].set(d_unit)
            dir_xy = dir_xy.at[end - 1].set(d_unit[-1])

        self.lane_points = jnp.concatenate([nodes_xyz[:, :2], dir_xy, lane_ids[:, None]], axis=-1)

    def compute_overtake_score(self, ego_trajectories, object_trajectories):
        ego_trajectories = jnp.asarray(ego_trajectories)
        object_trajectories = jnp.asarray(object_trajectories)

        ego_xy = ego_trajectories[:, -1, :2]
        object_xy = object_trajectories[:, -1, :2]
        object_heading = object_trajectories[:, -1, 2]

        rel_xy = ego_xy - object_xy[self.target_vehicle]
        longitudinal = rel_xy[:, 0] * jnp.cos(
            object_heading[self.target_vehicle]
        ) + rel_xy[:, 1] * jnp.sin(object_heading[self.target_vehicle])
        longitudinal = jnp.clip(longitudinal, a_min=-50.0, a_max=10.0)
        score = (longitudinal + 50.0) / 60.0
        return jnp.clip(score, a_min=0.0, a_max=1.0)

    def compute_collision_score(self, ego_trajectories, object_trajectories, object_masks):
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

    def compute_offroad_score(self, ego_trajectories, threshold=3.5):
        # raise NotImplementedError("compute_offroad_score is deprecated. Use compute_offroad_score_v2 instead.")
        # ego_trajectories = jnp.asarray(ego_trajectories)
        # if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
        #     raise ValueError(
        #         "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
        #         f"got {ego_trajectories.shape}"
        #     )
        # if self.lane_points is None:
        #     raise ValueError("lane_points is not initialized. Call update_lane_points first.")
        # if int(self.lane_points.shape[0]) == 0:
        #     return jnp.zeros((ego_trajectories.shape[0],), dtype=jnp.float32)

        # ego_xy = ego_trajectories[..., :2]
        # lane_xy = jnp.asarray(self.lane_points[:, :2], dtype=jnp.float32)

        # squared_distances = jnp.sum(
        #     (ego_xy[:, :, None, :] - lane_xy[None, None, :, :]) ** 2,
        #     axis=-1,
        # )
        # min_squared_distance_per_t = jnp.min(squared_distances, axis=-1)
        # offroad_flags = jnp.any(min_squared_distance_per_t >= (threshold ** 2), axis=1)
        # return jnp.where(offroad_flags, 0.0, 1.0)
        return jnp.ones((ego_trajectories.shape[0],), dtype=jnp.float32)

    def compute_offroad_score_v2(self, ego_trajectories, sim_state, timestep, world_idx=0,
                                 k_chunk: int = 4):
        ego_trajectories = jnp.asarray(ego_trajectories, dtype=jnp.float32)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )

        roadgraph_points = getattr(sim_state, "roadgraph_points", None)
        if roadgraph_points is None or not hasattr(roadgraph_points, "xyz"):
            return self.compute_offroad_score(ego_trajectories)

        roadgraph_xyz = jnp.asarray(roadgraph_points.xyz, dtype=jnp.float32)
        roadgraph_dir_xyz = jnp.asarray(roadgraph_points.dir_xyz, dtype=jnp.float32)
        roadgraph_types = jnp.asarray(roadgraph_points.types)
        roadgraph_valid = jnp.asarray(roadgraph_points.valid).astype(bool)
        roadgraph_ids = jnp.asarray(roadgraph_points.ids)

        if roadgraph_xyz.ndim >= 3:
            roadgraph_xyz = roadgraph_xyz[world_idx]
        if roadgraph_dir_xyz.ndim >= 3:
            roadgraph_dir_xyz = roadgraph_dir_xyz[world_idx]
        if roadgraph_types.ndim >= 2:
            roadgraph_types = roadgraph_types[world_idx]
        if roadgraph_valid.ndim >= 2:
            roadgraph_valid = roadgraph_valid[world_idx]
        if roadgraph_ids.ndim >= 2:
            roadgraph_ids = roadgraph_ids[world_idx]

        if roadgraph_xyz.ndim != 2 or roadgraph_xyz.shape[-1] != 3:
            return self.compute_offroad_score(ego_trajectories)
        if roadgraph_dir_xyz.ndim != 2 or roadgraph_dir_xyz.shape[-1] != 3:
            return self.compute_offroad_score(ego_trajectories)
        if roadgraph_types.shape[0] != roadgraph_xyz.shape[0]:
            return self.compute_offroad_score(ego_trajectories)
        if roadgraph_valid.shape[0] != roadgraph_xyz.shape[0]:
            return self.compute_offroad_score(ego_trajectories)
        if roadgraph_ids.shape[0] != roadgraph_xyz.shape[0]:
            return self.compute_offroad_score(ego_trajectories)

        roadgraph_xyz = roadgraph_xyz[roadgraph_valid]
        roadgraph_dir_xyz = roadgraph_dir_xyz[roadgraph_valid]
        roadgraph_types = roadgraph_types[roadgraph_valid]
        roadgraph_ids = roadgraph_ids[roadgraph_valid]

        road_edge_mask = datatypes.is_road_edge(roadgraph_types)
        if not bool(jnp.any(road_edge_mask)):
            return self.compute_offroad_score(ego_trajectories)

        roadgraph_xyz = roadgraph_xyz[road_edge_mask]
        roadgraph_dir_xyz = roadgraph_dir_xyz[road_edge_mask]
        roadgraph_ids = roadgraph_ids[road_edge_mask]
        if roadgraph_xyz.shape[0] == 0:
            return self.compute_offroad_score(ego_trajectories)

        # Chunk over K: full [K, T*4, P] distances OOMs on smaller GPUs
        # (e.g. K=64, T=52 → 208 corners × P≈2k ≈ 350MiB+ for one reduce).
        K = int(ego_trajectories.shape[0])
        chunk = max(1, int(k_chunk))
        parts = []
        for i0 in range(0, K, chunk):
            parts.append(self._offroad_score_v2_chunk(
                ego_trajectories[i0:i0 + chunk],
                roadgraph_xyz, roadgraph_dir_xyz, roadgraph_ids,
            ))
        return jnp.concatenate(parts, axis=0)

    @staticmethod
    def _offroad_score_v2_chunk(ego_trajectories, roadgraph_xyz, roadgraph_dir_xyz, roadgraph_ids):
        """Binary offroad score for a K-chunk. Returns [Ck] in {0,1}."""
        bbox_corners_xy = polygon_corners(ego_trajectories)
        z = jnp.zeros_like(bbox_corners_xy[..., :1])
        bbox_corners = jnp.concatenate([bbox_corners_xy, z], axis=-1)
        shape_prefix = bbox_corners.shape[:-3]
        num_agents, num_points, dim = bbox_corners.shape[-3:]
        bbox_corners = jnp.reshape(bbox_corners, [*shape_prefix, num_agents * num_points, dim])

        sampled_points = roadgraph_xyz
        nearest_vector_xyz = roadgraph_dir_xyz

        differences = sampled_points - jnp.expand_dims(bbox_corners, axis=-2)
        z_stretched_differences = differences * jnp.asarray([1.0, 1.0, 2.0], dtype=jnp.float32)
        square_distances = jnp.sum(z_stretched_differences**2, axis=-1)

        nearest_indices = jnp.argmin(square_distances, axis=-1)
        prior_indices = jnp.maximum(jnp.zeros_like(nearest_indices), nearest_indices - 1)

        nearest_xys = sampled_points[nearest_indices, :2]
        nearest_vector_xys = nearest_vector_xyz[nearest_indices, :2]
        prior_vector_xys = nearest_vector_xyz[prior_indices, :2]
        points_to_edge = bbox_corners[..., :2] - nearest_xys

        cross_product = points_to_edge[..., 0] * nearest_vector_xys[..., 1] - points_to_edge[..., 1] * nearest_vector_xys[..., 0]
        cross_product_prior = points_to_edge[..., 0] * prior_vector_xys[..., 1] - points_to_edge[..., 1] * prior_vector_xys[..., 0]
        prior_point_in_same_curve = roadgraph_ids[nearest_indices] == roadgraph_ids[prior_indices]
        offroad_sign = jnp.sign(
            jnp.where(
                jnp.logical_and(prior_point_in_same_curve, cross_product_prior < cross_product),
                cross_product_prior,
                cross_product,
            )
        )
        distances = jnp.linalg.norm(nearest_xys - bbox_corners[..., :2], axis=-1) * offroad_sign
        distances = jnp.reshape(distances, [*shape_prefix, num_agents, num_points])
        offroad = jnp.any(distances > 0.0, axis=(-1, -2))
        return jnp.where(offroad, 0.0, 1.0).astype(jnp.float32)

    def compute_direction_score(self, ego_trajectories):
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
        nearest_lane_heading = jnp.arctan2(
            nearest_lane_dir_xy[..., 1], nearest_lane_dir_xy[..., 0]
        )

        heading_delta = ego_heading - nearest_lane_heading
        wrapped_delta = jnp.arctan2(jnp.sin(heading_delta), jnp.cos(heading_delta))
        wrong_direction_flags = jnp.abs(wrapped_delta) >= (jnp.pi / 2.0)
        any_wrong_direction = jnp.any(wrong_direction_flags, axis=1)
        return jnp.where(any_wrong_direction, 0.0, 1.0)

    def compute_lane_score(self, ego_trajectories):
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

        position_score = (
            jnp.clip(max_lane_deviation - xy_loss_min, 0.0, max_lane_deviation)
            / max_lane_deviation
        )
        heading_score = 1.0 - ((heading_loss_sel + 1.0) / 2.0)
        score_t = (position_score + heading_score) / 2.0
        final_score = jnp.mean(score_t, axis=1)
        return final_score.astype(jnp.float32)

    def compute_speed_score(self, ego_trajectories):
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
        progress = jnp.linalg.norm(
            ego_trajectories[:, -1, :2] - ego_trajectories[:, 0, :2], axis=-1
        )
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

    def compute_tl_violation_score(
        self,
        ego_trajectories,
        sim_state,
        timestep,
        world_idx=0,
        dist_threshold=2.0,
        stop_state=4,
    ):
        """Returns 0 if planned trajectory violates red-light stopline, else 1."""
        ego_trajectories = jnp.asarray(ego_trajectories, dtype=jnp.float32)
        if ego_trajectories.ndim != 3 or ego_trajectories.shape[-1] != 5:
            raise ValueError(
                "ego_trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {ego_trajectories.shape}"
            )
        num_ego_trajectories = int(ego_trajectories.shape[0])

        if self.lane_points is None:
            scorer_helpers.update_lane_points(self, sim_state, world_idx)
        if self.lane_points is None or int(self.lane_points.shape[0]) == 0:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)
        if self._lane_graph is None:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        ego_mask = jnp.asarray(sim_state.object_metadata.is_sdc[world_idx]).astype(bool)
        if int(jnp.sum(ego_mask)) != 1:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)
        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))

        tls = sim_state.log_traffic_light
        tl_state = jnp.asarray(tls.state)
        tl_lane_ids = jnp.asarray(tls.lane_ids)
        tl_valid = jnp.asarray(tls.valid).astype(bool)

        if hasattr(tls, "xy"):
            tl_xy = jnp.asarray(tls.xy, dtype=jnp.float32)
        elif hasattr(tls, "x") and hasattr(tls, "y"):
            tl_x = jnp.asarray(tls.x, dtype=jnp.float32)
            tl_y = jnp.asarray(tls.y, dtype=jnp.float32)
            tl_xy = jnp.stack([tl_x, tl_y], axis=-1)
        else:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        if tl_state.ndim == 3:
            b = int(world_idx)
            tl_state = tl_state[b]
            tl_lane_ids = tl_lane_ids[b]
            tl_valid = tl_valid[b]
            tl_xy = tl_xy[b]
        elif tl_state.ndim != 2:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        if tl_xy.ndim != 3 or tl_xy.shape[-1] != 2:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)
        if not bool(jnp.any(tl_valid)):
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        ego_log_xy = jnp.asarray(sim_state.log_trajectory.xy[world_idx, ego_idx], dtype=jnp.float32)
        tl_total_steps = int(tl_state.shape[1])
        ego_total_steps = int(ego_log_xy.shape[0])
        traj_horizon = int(ego_trajectories.shape[1])
        start_t = int(max(0, timestep))
        horizon = min(traj_horizon, tl_total_steps - start_t, ego_total_steps - start_t)
        if horizon < 2:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        step_idx = jnp.arange(horizon, dtype=jnp.int32) + int(start_t)
        ego_ref_xy_t2 = ego_log_xy[step_idx]
        tl_xy_lt2 = tl_xy[:, step_idx, :]
        tl_valid_lt = tl_valid[:, step_idx]

        distance_ego2tl_lt = jnp.linalg.norm(ego_ref_xy_t2[None, :, :] - tl_xy_lt2, axis=-1)
        large = jnp.asarray(1e9, dtype=jnp.float32)
        distance_ego2tl_masked_lt = jnp.where(tl_valid_lt, distance_ego2tl_lt, large)
        nearest_tl_idx_t = jnp.argmin(distance_ego2tl_masked_lt, axis=0)

        state_t = tl_state[nearest_tl_idx_t, step_idx]
        lane_ids_t = tl_lane_ids[nearest_tl_idx_t, step_idx]
        valid_t = tl_valid[nearest_tl_idx_t, step_idx]

        lane_ids_t_np = np.asarray(lane_ids_t)
        valid_t_np = np.asarray(valid_t)
        lane_pick_t = np.flatnonzero(valid_t_np & (lane_ids_t_np > 0))
        if lane_pick_t.size == 0:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        lane_id0 = int(lane_ids_t_np[int(lane_pick_t[0])])
        if lane_id0 <= 0:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        node_range = self._lane_graph.lane_id_to_node_range.get(lane_id0)
        if node_range is None:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)
        start, end = int(node_range[0]), int(node_range[1])
        if end <= start:
            return jnp.ones((num_ego_trajectories,), dtype=jnp.float32)

        lane_points = jnp.asarray(self.lane_points, dtype=jnp.float32)
        stopline_xy = lane_points[start, :2]
        stopline_dir = lane_points[start, 2:4]
        if float(jnp.linalg.norm(stopline_dir)) <= 1e-6 and end - start > 1:
            stopline_dir = lane_points[start + 1, :2] - lane_points[start, :2]
        stopline_dir = stopline_dir / jnp.maximum(jnp.linalg.norm(stopline_dir), 1e-6)

        distance_ego2lane_t = jnp.linalg.norm(ego_ref_xy_t2 - stopline_xy[None, :], axis=-1)
        red_mask_t = (
            valid_t
            & (state_t == int(stop_state))
            & (distance_ego2lane_t <= float(dist_threshold))
        )

        traj_xy = ego_trajectories[:, :horizon, :2]
        rel = traj_xy - stopline_xy[None, None, :]
        signed_dist = rel[..., 0] * stopline_dir[0] + rel[..., 1] * stopline_dir[1]

        crosses = (
            red_mask_t[None, :-1]
            & (signed_dist[:, :-1] < 0.0)
            & (signed_dist[:, 1:] >= 0.0)
        )
        violation = jnp.any(crosses, axis=1)
        return jnp.where(violation, 0.0, 1.0).astype(jnp.float32)

    def compute_score(self, trajectories, sim_state, timestep, world_idx=0):
        trajectories = jnp.asarray(trajectories)
        if trajectories.ndim != 3 or trajectories.shape[-1] != 5:
            raise ValueError(
                "trajectories must have shape [num_ego_trajectories, num_time_steps, 5], "
                f"got {trajectories.shape}"
            )
        if self.lane_points is None:
            scorer_helpers.update_lane_points(self, sim_state, world_idx)

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

        object_xy = object_xy[:, timestep : timestep + traj_length, :]
        object_yaw = object_yaw[:, timestep : timestep + traj_length]
        object_length = object_length[:, timestep : timestep + traj_length]
        object_width = object_width[:, timestep : timestep + traj_length]
        object_masks = object_valid[:, timestep : timestep + traj_length]
        object_trajectories = jnp.stack(
            [object_xy[..., 0], object_xy[..., 1], object_yaw, object_length, object_width],
            axis=-1,
        )
        object_masks = (object_masks > 0) & (~ego_mask[:, None])

        ego_idx = int(jnp.argmax(ego_mask.astype(jnp.int32)))
        ego_length_t = object_length[ego_idx]
        ego_width_t = object_width[ego_idx]
        num_ego_trajectories = trajectories.shape[0]
        ego_length = jnp.broadcast_to(
            ego_length_t[None, :], (num_ego_trajectories, traj_length)
        )
        ego_width = jnp.broadcast_to(
            ego_width_t[None, :], (num_ego_trajectories, traj_length)
        )
        ego_trajectories = jnp.stack(
            [
                trajectories[:, :traj_length, 0],
                trajectories[:, :traj_length, 1],
                trajectories[:, :traj_length, 2],
                ego_length,
                ego_width,
            ],
            axis=-1,
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
                elif metric_name == "speed":
                    speed_score = self.compute_speed_score(ego_trajectories)
                    score = score + weight * speed_score
                elif metric_name == "overtake":
                    overtake_score = self.compute_overtake_score(
                        ego_trajectories, object_trajectories
                    )
                    score = score + weight * overtake_score
                elif metric_name == "goal":
                    goal_score = self.compute_goal_score(ego_trajectories)
                    score = score + weight * goal_score
                score = score / total_weight
        if "collision" in self._multiplicative_metrics:
            collision_score = self.compute_collision_score(
                ego_trajectories, object_trajectories, object_masks
            )
            score = score * collision_score
        if "offroad" in self._multiplicative_metrics:
            offroad_score = self.compute_offroad_score_v2(
                ego_trajectories, sim_state=sim_state, timestep=timestep, world_idx=world_idx
            )
            score = score * offroad_score
        if "direction" in self._multiplicative_metrics:
            direction_score = self.compute_direction_score(ego_trajectories)
            score = score * direction_score
        if "tl_violation" in self._multiplicative_metrics:
            tl_score = self.compute_tl_violation_score(
                ego_trajectories, sim_state=sim_state, timestep=timestep, world_idx=world_idx
            )
            score = score * tl_score

        return score
