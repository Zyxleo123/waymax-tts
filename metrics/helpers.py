from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from waymax import datatypes
from waymax.datatypes.roadgraph import MapElementIds


ArrayLike = Union[jax.Array, int, float]


def _scalar_int(x: ArrayLike) -> int:
	arr = jnp.asarray(x)
	return int(arr.item())


def _select_batch(
		arr: jax.Array,
		trailing_dims: int,
		batch_index: Optional[Union[int, Sequence[int]]] = None,
) -> jax.Array:
	jarr = jnp.asarray(arr)
	if jarr.ndim <= trailing_dims:
		return jarr

	lead_dims = jarr.ndim - trailing_dims
	if batch_index is None:
		idx: Tuple[int, ...] = tuple(0 for _ in range(lead_dims))
	elif isinstance(batch_index, int):
		if lead_dims != 1:
			raise ValueError(f"batch_index must provide {lead_dims} indices, got scalar")
		idx = (batch_index,)
	else:
		idx = tuple(int(v) for v in batch_index)
		if len(idx) != lead_dims:
			raise ValueError(f"batch_index must have length {lead_dims}, got {len(idx)}")

	return jarr[idx + (slice(None),) * trailing_dims]


@dataclass
class LaneGraph:
	point_ids: jax.Array
	point_types: jax.Array
	points_xy: jax.Array
	point_dirs_xy: jax.Array
	lane_ids: jax.Array
	points_by_lane: Dict[int, jax.Array]
	point_dirs_by_lane: Dict[int, jax.Array]

	@classmethod
	def from_roadgraph_points(
			cls,
			roadgraph_points: datatypes.RoadgraphPoints,
			batch_index: Optional[Union[int, Sequence[int]]] = None,
	) -> "LaneGraph":
		ids = _select_batch(roadgraph_points.ids, trailing_dims=1, batch_index=batch_index)
		types = _select_batch(roadgraph_points.types, trailing_dims=1, batch_index=batch_index)
		x = _select_batch(roadgraph_points.x, trailing_dims=1, batch_index=batch_index)
		y = _select_batch(roadgraph_points.y, trailing_dims=1, batch_index=batch_index)
		dir_x = _select_batch(roadgraph_points.dir_x, trailing_dims=1, batch_index=batch_index)
		dir_y = _select_batch(roadgraph_points.dir_y, trailing_dims=1, batch_index=batch_index)
		valid = _select_batch(roadgraph_points.valid, trailing_dims=1, batch_index=batch_index).astype(jnp.bool_)

		centerline_types = {
				int(MapElementIds.LANE_FREEWAY.value),
				int(MapElementIds.LANE_SURFACE_STREET.value),
				int(MapElementIds.LANE_BIKE_LANE.value),
		}
		type_mask = jnp.isin(types, jnp.asarray(list(centerline_types), dtype=types.dtype))
		mask = valid & type_mask

		point_ids = ids[mask].astype(jnp.int32)
		point_types = types[mask].astype(jnp.int32)
		points_xy = jnp.stack([x[mask], y[mask]], axis=-1).astype(jnp.float32)
		point_dirs_xy = jnp.stack([dir_x[mask], dir_y[mask]], axis=-1).astype(jnp.float32)

		lane_ids = jnp.unique(point_ids).astype(jnp.int32)

		points_by_lane: Dict[int, jax.Array] = {}
		point_dirs_by_lane: Dict[int, jax.Array] = {}
		for lane_id in lane_ids.tolist():
			lane_mask = point_ids == lane_id
			points_by_lane[int(lane_id)] = points_xy[lane_mask]
			point_dirs_by_lane[int(lane_id)] = point_dirs_xy[lane_mask]

		return cls(
				point_ids=point_ids,
				point_types=point_types,
				points_xy=points_xy,
				point_dirs_xy=point_dirs_xy,
				lane_ids=lane_ids,
				points_by_lane=points_by_lane,
				point_dirs_by_lane=point_dirs_by_lane,
		)


@dataclass
class LaneNode:
	index: int
	lane_id: int
	position: jax.Array
	direction: jax.Array
	outgoing: List[int] = field(default_factory=list)
	incoming: List[int] = field(default_factory=list)


@dataclass
class EgoArea:
	lane_node_index: Optional[int]
	lane_id: Optional[int]
	center_xy: jax.Array
	collision_corners_xy: jax.Array
	map_corners_xy: jax.Array
	collision_polygon_xy: jax.Array
	map_polygon_xy: jax.Array
	corner_lane_node_indices: jax.Array
	corner_lane_ids: jax.Array
	corner_min_distances_m: jax.Array
	center_min_distance_m: jax.Array
	heading_error_rad: jax.Array
	multiple_lanes: jax.Array
	non_drivable_area: jax.Array

	@classmethod
	def empty(cls) -> "EgoArea":
		empty_xy = jnp.empty((0, 2), dtype=jnp.float32)
		empty_i32 = jnp.empty((0,), dtype=jnp.int32)
		empty_f32 = jnp.empty((0,), dtype=jnp.float32)
		false_scalar = jnp.asarray(False, dtype=jnp.bool_)
		return cls(
				lane_node_index=None,
				lane_id=None,
				center_xy=empty_f32,
				collision_corners_xy=empty_xy,
				map_corners_xy=empty_xy,
				collision_polygon_xy=empty_xy,
				map_polygon_xy=empty_xy,
				corner_lane_node_indices=empty_i32,
				corner_lane_ids=empty_i32,
				corner_min_distances_m=empty_f32,
				center_min_distance_m=jnp.asarray(jnp.inf, dtype=jnp.float32),
				heading_error_rad=jnp.asarray(jnp.pi, dtype=jnp.float32),
				multiple_lanes=false_scalar,
				non_drivable_area=false_scalar,
		)


@dataclass
class VehicleGraph:
	object_id_to_abs_index: Dict[int, int]
	abs_index_to_object_id: Dict[int, int]
	ego_object_id: Optional[int]
	timestep: int
	object_ids: jax.Array
	is_ego: jax.Array
	is_valid: jax.Array
	x: jax.Array
	y: jax.Array
	z: jax.Array
	yaw: jax.Array
	vel_x: jax.Array
	vel_y: jax.Array
	length: jax.Array
	width: jax.Array
	height: jax.Array

	@property
	def num_tracked(self) -> int:
		return int(self.object_ids.shape[0])

	@classmethod
	def empty(cls) -> "VehicleGraph":
		empty_i32 = jnp.empty((0,), dtype=jnp.int32)
		empty_bool = jnp.empty((0,), dtype=jnp.bool_)
		empty_f32 = jnp.empty((0,), dtype=jnp.float32)
		return cls(
				object_id_to_abs_index={},
				abs_index_to_object_id={},
				ego_object_id=None,
				timestep=-1,
				object_ids=empty_i32,
				is_ego=empty_bool,
				is_valid=empty_bool,
				x=empty_f32,
				y=empty_f32,
				z=empty_f32,
				yaw=empty_f32,
				vel_x=empty_f32,
				vel_y=empty_f32,
				length=empty_f32,
				width=empty_f32,
				height=empty_f32,
		)


class World:
	"""Waymax simulator state 기반 world helper.

	- lanegraph: centerline point/dir 저장
	- vehiclegraph: timestep별 vehicle 상태 저장 (stable absolute index)
	"""

	def __init__(
			self,
			simulator_state: datatypes.SimulatorState,
			batch_index: Optional[Union[int, Sequence[int]]] = None,
			prediction_horizon: int = 10,
			prediction_dt_s: float = 0.1,
			centerline_radius_m: float = 150.0,
			centerline_lateral_tolerance_m: float = 0.5,
			drivable_area_tolerance_m: float = 1.5,
			drivable_area_reduction_m: float = 0.0,
			collision_scale: float = 1.0,
	):
		roadgraph_points = getattr(simulator_state, "road_graph", None)
		if roadgraph_points is None:
			roadgraph_points = getattr(simulator_state, "roadgraph_points", None)

		if roadgraph_points is None:
			raise ValueError("simulator_state must include road_graph or roadgraph_points")

		self.simulator_state = simulator_state
		self.batch_index = batch_index
		self.prediction_horizon = int(prediction_horizon)
		self.prediction_dt_s = float(prediction_dt_s)
		self.roadgraph_points = roadgraph_points
		self.centerline_radius_m = float(centerline_radius_m)
		self.centerline_lateral_tolerance_m = float(centerline_lateral_tolerance_m)
		self.drivable_area_tolerance_m = float(drivable_area_tolerance_m)
		self.drivable_area_reduction_m = float(drivable_area_reduction_m)
		self.collision_scale = float(collision_scale)
		self.lanegraph = LaneGraph.from_roadgraph_points(
				roadgraph_points=roadgraph_points,
				batch_index=batch_index,
		)
		self.drivable_area_point_ids = self.lanegraph.point_ids
		self.drivable_area_point_types = self.lanegraph.point_types
		self.drivable_area_points_xy = self.lanegraph.points_xy
		self.drivable_area_point_dirs_xy = self.lanegraph.point_dirs_xy
		self.drivable_area_lane_ids = self.lanegraph.lane_ids
		self.lane_nodes: List[LaneNode] = []
		self.lane_nodes_by_index: Dict[int, LaneNode] = {}
		self.lane_nodes_by_lane: Dict[int, List[LaneNode]] = {}
		self.vehiclegraph = VehicleGraph.empty()
		self.vehicle_centers_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.vehicle_corners_xy = jnp.empty((0, 4, 2), dtype=jnp.float32)
		self.vehicle_polygons_xy = jnp.empty((0, 5, 2), dtype=jnp.float32)
		self.vehicle_valid_mask = jnp.empty((0,), dtype=jnp.bool_)
		self.vehicle_velocity_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.vehicle_speed = jnp.empty((0,), dtype=jnp.float32)
		self.vehicle_heading_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.vehicle_center_distance_matrix = jnp.empty((0, 0), dtype=jnp.float32)
		self.ego_to_vehicle_center_distances = jnp.empty((0,), dtype=jnp.float32)
		self.ego_center_xy = jnp.empty((0,), dtype=jnp.float32)
		self.ego_velocity_xy = jnp.empty((0,), dtype=jnp.float32)
		self.ego_speed = jnp.asarray(0.0, dtype=jnp.float32)
		self.ego_heading_xy = jnp.empty((0,), dtype=jnp.float32)
		self.ego_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.ego_collision_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.ego_map_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.ego_collision_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.ego_map_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
		self.ego_area = EgoArea.empty()
		self.other_vehicle_object_ids = jnp.empty((0,), dtype=jnp.int32)
		self.other_vehicle_future_centers_xy = jnp.empty((0, self.prediction_horizon + 1, 2), dtype=jnp.float32)
		self.other_vehicle_future_yaw = jnp.empty((0, self.prediction_horizon + 1), dtype=jnp.float32)
		self.other_vehicle_future_polygons_xy = jnp.empty((0, self.prediction_horizon + 1, 5, 2), dtype=jnp.float32)
		self.other_vehicle_future_valid = jnp.empty((0, self.prediction_horizon + 1), dtype=jnp.bool_)
		self.other_vehicle_future_speed = jnp.empty((0, self.prediction_horizon + 1), dtype=jnp.float32)
		self._next_non_ego_abs_index = 1
		self.update(simulator_state)

	def _ensure_vehicle_capacity(self, required_size: int) -> None:
		current_size = int(self.vehiclegraph.object_ids.shape[0])
		if required_size <= current_size:
			return

		pad = required_size - current_size

		self.vehiclegraph.object_ids = jnp.concatenate(
				[self.vehiclegraph.object_ids, jnp.full((pad,), -1, dtype=jnp.int32)], axis=0
		)
		self.vehiclegraph.is_ego = jnp.concatenate(
				[self.vehiclegraph.is_ego, jnp.zeros((pad,), dtype=jnp.bool_)], axis=0
		)
		self.vehiclegraph.is_valid = jnp.concatenate(
				[self.vehiclegraph.is_valid, jnp.zeros((pad,), dtype=jnp.bool_)], axis=0
		)

		for attr in ["x", "y", "z", "yaw", "vel_x", "vel_y", "length", "width", "height"]:
			value = getattr(self.vehiclegraph, attr)
			padded = jnp.concatenate([value, jnp.zeros((pad,), dtype=jnp.float32)], axis=0)
			setattr(self.vehiclegraph, attr, padded)

	def _assign_abs_index(self, object_id: int, is_ego: bool) -> int:
		mapping = self.vehiclegraph.object_id_to_abs_index

		if is_ego:
			self.vehiclegraph.ego_object_id = int(object_id)
			existing = mapping.get(int(object_id))
			if existing == 0:
				return 0

			occupant_id = self.vehiclegraph.abs_index_to_object_id.get(0)
			if occupant_id is not None and occupant_id != int(object_id):
				new_index = self._next_non_ego_abs_index
				self._next_non_ego_abs_index += 1
				self._ensure_vehicle_capacity(new_index + 1)
				mapping[int(occupant_id)] = new_index
				self.vehiclegraph.abs_index_to_object_id[new_index] = int(occupant_id)
				del self.vehiclegraph.abs_index_to_object_id[0]
				for attr in ["object_ids", "is_ego", "is_valid", "x", "y", "z", "yaw", "vel_x", "vel_y", "length", "width", "height"]:
					arr = getattr(self.vehiclegraph, attr)
					arr = arr.at[new_index].set(arr[0])
					setattr(self.vehiclegraph, attr, arr)

			if existing is not None and existing != 0:
				del self.vehiclegraph.abs_index_to_object_id[existing]

			mapping[int(object_id)] = 0
			self.vehiclegraph.abs_index_to_object_id[0] = int(object_id)
			self._ensure_vehicle_capacity(1)
			return 0

		if int(object_id) in mapping:
			return int(mapping[int(object_id)])

		index = self._next_non_ego_abs_index
		self._next_non_ego_abs_index += 1
		mapping[int(object_id)] = index
		self.vehiclegraph.abs_index_to_object_id[index] = int(object_id)
		self._ensure_vehicle_capacity(index + 1)
		return index

	def update(self, simulator_state: datatypes.SimulatorState) -> None:
		self.simulator_state = simulator_state
		self.vehiclegraph.timestep = _scalar_int(simulator_state.timestep)

		traj = simulator_state.sim_trajectory
		metadata = simulator_state.object_metadata

		object_ids = _select_batch(metadata.ids, trailing_dims=1, batch_index=self.batch_index).astype(jnp.int32)
		is_sdc = _select_batch(metadata.is_sdc, trailing_dims=1, batch_index=self.batch_index).astype(jnp.bool_)
		meta_valid = _select_batch(metadata.is_valid, trailing_dims=1, batch_index=self.batch_index).astype(jnp.bool_)

		timestep = self.vehiclegraph.timestep

		x_all = _select_batch(traj.x, trailing_dims=2, batch_index=self.batch_index)
		y_all = _select_batch(traj.y, trailing_dims=2, batch_index=self.batch_index)
		z_all = _select_batch(traj.z, trailing_dims=2, batch_index=self.batch_index)
		yaw_all = _select_batch(traj.yaw, trailing_dims=2, batch_index=self.batch_index)
		vel_x_all = _select_batch(traj.vel_x, trailing_dims=2, batch_index=self.batch_index)
		vel_y_all = _select_batch(traj.vel_y, trailing_dims=2, batch_index=self.batch_index)
		length_all = _select_batch(traj.length, trailing_dims=2, batch_index=self.batch_index)
		width_all = _select_batch(traj.width, trailing_dims=2, batch_index=self.batch_index)
		height_all = _select_batch(traj.height, trailing_dims=2, batch_index=self.batch_index)
		traj_valid_all = _select_batch(traj.valid, trailing_dims=2, batch_index=self.batch_index).astype(jnp.bool_)

		num_steps = int(x_all.shape[-1])
		if timestep < 0 or timestep >= num_steps:
			raise IndexError(f"timestep {timestep} out of range [0, {num_steps - 1}]")

		x_now = x_all[:, timestep].astype(jnp.float32)
		y_now = y_all[:, timestep].astype(jnp.float32)
		z_now = z_all[:, timestep].astype(jnp.float32)
		yaw_now = yaw_all[:, timestep].astype(jnp.float32)
		vel_x_now = vel_x_all[:, timestep].astype(jnp.float32)
		vel_y_now = vel_y_all[:, timestep].astype(jnp.float32)
		length_now = length_all[:, timestep].astype(jnp.float32)
		width_now = width_all[:, timestep].astype(jnp.float32)
		height_now = height_all[:, timestep].astype(jnp.float32)
		valid_now = (traj_valid_all[:, timestep] & meta_valid).astype(jnp.bool_)

		self.vehiclegraph.is_valid = self.vehiclegraph.is_valid.at[:].set(False)

		ego_indices = jnp.where(is_sdc)[0]
		ego_id: Optional[int] = None
		if int(ego_indices.shape[0]) > 0:
			ego_id = int(object_ids[int(ego_indices[0])])

		for local_index in range(int(object_ids.shape[0])):
			object_id = int(object_ids[local_index])
			if object_id < 0:
				continue

			is_ego_vehicle = (ego_id is not None and object_id == ego_id)
			abs_index = self._assign_abs_index(object_id=object_id, is_ego=is_ego_vehicle)

			self.vehiclegraph.object_ids = self.vehiclegraph.object_ids.at[abs_index].set(jnp.int32(object_id))
			self.vehiclegraph.is_ego = self.vehiclegraph.is_ego.at[abs_index].set(bool(is_ego_vehicle))
			self.vehiclegraph.is_valid = self.vehiclegraph.is_valid.at[abs_index].set(bool(valid_now[local_index]))
			self.vehiclegraph.x = self.vehiclegraph.x.at[abs_index].set(x_now[local_index])
			self.vehiclegraph.y = self.vehiclegraph.y.at[abs_index].set(y_now[local_index])
			self.vehiclegraph.z = self.vehiclegraph.z.at[abs_index].set(z_now[local_index])
			self.vehiclegraph.yaw = self.vehiclegraph.yaw.at[abs_index].set(yaw_now[local_index])
			self.vehiclegraph.vel_x = self.vehiclegraph.vel_x.at[abs_index].set(vel_x_now[local_index])
			self.vehiclegraph.vel_y = self.vehiclegraph.vel_y.at[abs_index].set(vel_y_now[local_index])
			self.vehiclegraph.length = self.vehiclegraph.length.at[abs_index].set(length_now[local_index])
			self.vehiclegraph.width = self.vehiclegraph.width.at[abs_index].set(width_now[local_index])
			self.vehiclegraph.height = self.vehiclegraph.height.at[abs_index].set(height_now[local_index])

		if ego_id is not None:
			self.vehiclegraph.is_ego = self.vehiclegraph.is_ego.at[:].set(False)
			self.vehiclegraph.is_ego = self.vehiclegraph.is_ego.at[0].set(True)

		self._rebuild_vehicle_geometry()
		self._build_lane_nodes()
		self._rebuild_metric_cache()
		self._rebuild_other_vehicle_predictions()

	def _ego_position_xy(self) -> Optional[jax.Array]:
		if self.vehiclegraph.ego_object_id is None:
			return None

		if int(self.vehiclegraph.x.shape[0]) == 0:
			return None

		if int(self.vehiclegraph.is_valid.shape[0]) > 0 and not bool(self.vehiclegraph.is_valid[0]):
			return None

		return jnp.stack([self.vehiclegraph.x[0], self.vehiclegraph.y[0]], axis=0)

	def _rebuild_vehicle_geometry(self) -> None:
		if int(self.vehiclegraph.x.shape[0]) == 0:
			self.vehicle_centers_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.vehicle_corners_xy = jnp.empty((0, 4, 2), dtype=jnp.float32)
			self.vehicle_polygons_xy = jnp.empty((0, 5, 2), dtype=jnp.float32)
			self.vehicle_valid_mask = jnp.empty((0,), dtype=jnp.bool_)
			self.vehicle_velocity_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.vehicle_speed = jnp.empty((0,), dtype=jnp.float32)
			self.vehicle_heading_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.vehicle_center_distance_matrix = jnp.empty((0, 0), dtype=jnp.float32)
			self.ego_to_vehicle_center_distances = jnp.empty((0,), dtype=jnp.float32)
			self.ego_center_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_velocity_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_speed = jnp.asarray(0.0, dtype=jnp.float32)
			self.ego_heading_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_collision_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_map_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_collision_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_map_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			return

		centers = jnp.stack([self.vehiclegraph.x, self.vehiclegraph.y], axis=-1).astype(jnp.float32)
		velocity_xy = jnp.stack([self.vehiclegraph.vel_x, self.vehiclegraph.vel_y], axis=-1).astype(jnp.float32)
		speed = jnp.linalg.norm(velocity_xy, axis=-1).astype(jnp.float32)
		yaw = self.vehiclegraph.yaw.astype(jnp.float32)
		length = self.vehiclegraph.length.astype(jnp.float32)
		width = self.vehiclegraph.width.astype(jnp.float32)
		valid_mask = self.vehiclegraph.is_valid.astype(jnp.bool_)
		heading_xy = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1).astype(jnp.float32)

		half_l = length * 0.5
		half_w = width * 0.5
		cos_yaw = jnp.cos(yaw)
		sin_yaw = jnp.sin(yaw)

		local_corners = jnp.stack(
				[
					jnp.stack([half_l, half_w], axis=-1),
					jnp.stack([half_l, -half_w], axis=-1),
					jnp.stack([-half_l, -half_w], axis=-1),
					jnp.stack([-half_l, half_w], axis=-1),
				],
				axis=1,
		)

		rot_x = local_corners[..., 0] * cos_yaw[:, None] - local_corners[..., 1] * sin_yaw[:, None]
		rot_y = local_corners[..., 0] * sin_yaw[:, None] + local_corners[..., 1] * cos_yaw[:, None]
		corners = jnp.stack([rot_x, rot_y], axis=-1) + centers[:, None, :]
		polygons = jnp.concatenate([corners, corners[:, :1, :]], axis=1)

		self.vehicle_centers_xy = centers
		self.vehicle_corners_xy = corners
		self.vehicle_polygons_xy = polygons
		self.vehicle_valid_mask = valid_mask
		self.vehicle_velocity_xy = velocity_xy
		self.vehicle_speed = speed
		self.vehicle_heading_xy = heading_xy
		center_diffs = centers[:, None, :] - centers[None, :, :]
		self.vehicle_center_distance_matrix = jnp.linalg.norm(center_diffs, axis=-1).astype(jnp.float32)

		if int(valid_mask.shape[0]) > 0 and bool(valid_mask[0]):
			self.ego_center_xy = centers[0]
			self.ego_velocity_xy = velocity_xy[0]
			self.ego_speed = speed[0]
			self.ego_heading_xy = heading_xy[0]
			self.ego_polygon_xy = polygons[0]
			self.ego_to_vehicle_center_distances = self.vehicle_center_distance_matrix[0]
		else:
			self.ego_center_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_velocity_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_speed = jnp.asarray(0.0, dtype=jnp.float32)
			self.ego_heading_xy = jnp.empty((0,), dtype=jnp.float32)
			self.ego_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_to_vehicle_center_distances = jnp.empty((0,), dtype=jnp.float32)

	def _polygon_from_corners(self, corners_xy: jax.Array) -> jax.Array:
		if int(corners_xy.shape[0]) == 0:
			return jnp.empty((0, 2), dtype=jnp.float32)
		return jnp.concatenate([corners_xy, corners_xy[:1]], axis=0).astype(jnp.float32)

	def _scaled_ego_corners(self, scale_longitudinal: float, scale_lateral: float) -> jax.Array:
		if int(self.vehicle_corners_xy.shape[0]) == 0 or not bool(self.vehicle_valid_mask[0]):
			return jnp.empty((0, 2), dtype=jnp.float32)

		center = self.vehicle_centers_xy[0]
		corners = self.vehicle_corners_xy[0]
		local = corners - center
		scaled = local * jnp.asarray([scale_longitudinal, scale_lateral], dtype=jnp.float32)
		return (center + scaled).astype(jnp.float32)

	def _rebuild_metric_cache(self) -> None:
		if int(self.vehicle_corners_xy.shape[0]) == 0 or not bool(self.vehicle_valid_mask[0]):
			self.ego_collision_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_map_corners_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_collision_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_map_polygon_xy = jnp.empty((0, 2), dtype=jnp.float32)
			self.ego_area = EgoArea.empty()
			return

		length = float(self.vehiclegraph.length[0])
		width = float(self.vehiclegraph.width[0])
		collision_scale = max(self.collision_scale, 0.0)
		if length > 1e-6:
			map_longitudinal_scale = max(length - 2.0 * self.drivable_area_reduction_m, 0.0) / length
		else:
			map_longitudinal_scale = 1.0
		if width > 1e-6:
			map_lateral_scale = max(width - 2.0 * self.drivable_area_reduction_m, 0.0) / width
		else:
			map_lateral_scale = 1.0

		self.ego_collision_corners_xy = self._scaled_ego_corners(
				scale_longitudinal=collision_scale,
				scale_lateral=collision_scale,
		)
		self.ego_map_corners_xy = self._scaled_ego_corners(
				scale_longitudinal=map_longitudinal_scale,
				scale_lateral=map_lateral_scale,
		)
		self.ego_collision_polygon_xy = self._polygon_from_corners(self.ego_collision_corners_xy)
		self.ego_map_polygon_xy = self._polygon_from_corners(self.ego_map_corners_xy)
		self.ego_area = self._compute_ego_area()

	def _rebuild_other_vehicle_predictions(self) -> None:
		horizon = self.prediction_horizon + 1
		if int(self.vehicle_centers_xy.shape[0]) <= 1:
			self.other_vehicle_object_ids = jnp.empty((0,), dtype=jnp.int32)
			self.other_vehicle_future_centers_xy = jnp.empty((0, horizon, 2), dtype=jnp.float32)
			self.other_vehicle_future_yaw = jnp.empty((0, horizon), dtype=jnp.float32)
			self.other_vehicle_future_polygons_xy = jnp.empty((0, horizon, 5, 2), dtype=jnp.float32)
			self.other_vehicle_future_valid = jnp.empty((0, horizon), dtype=jnp.bool_)
			self.other_vehicle_future_speed = jnp.empty((0, horizon), dtype=jnp.float32)
			return

		times = (jnp.arange(horizon, dtype=jnp.float32) * self.prediction_dt_s)[None, :, None]
		other_centers_now = self.vehicle_centers_xy[1:]
		other_velocity_xy = self.vehicle_velocity_xy[1:]
		other_yaw_now = self.vehiclegraph.yaw[1:].astype(jnp.float32)
		other_valid_now = self.vehicle_valid_mask[1:]
		other_speed_now = self.vehicle_speed[1:]
		other_corners_local = self.vehicle_corners_xy[1:] - other_centers_now[:, None, :]

		future_centers = other_centers_now[:, None, :] + other_velocity_xy[:, None, :] * times
		future_corners = other_corners_local[:, None, :, :] + future_centers[:, :, None, :]
		future_polygons = jnp.concatenate([future_corners, future_corners[:, :, :1, :]], axis=2)

		self.other_vehicle_object_ids = self.vehiclegraph.object_ids[1:].astype(jnp.int32)
		self.other_vehicle_future_centers_xy = future_centers.astype(jnp.float32)
		self.other_vehicle_future_yaw = jnp.repeat(other_yaw_now[:, None], horizon, axis=1).astype(jnp.float32)
		self.other_vehicle_future_polygons_xy = future_polygons.astype(jnp.float32)
		self.other_vehicle_future_valid = jnp.repeat(other_valid_now[:, None], horizon, axis=1).astype(jnp.bool_)
		self.other_vehicle_future_speed = jnp.repeat(other_speed_now[:, None], horizon, axis=1).astype(jnp.float32)

	def _compute_ego_area(self) -> EgoArea:
		if not self.lane_nodes or int(self.ego_center_xy.shape[0]) == 0:
			return EgoArea.empty()

		node_positions = jnp.stack([node.position for node in self.lane_nodes], axis=0).astype(jnp.float32)
		node_dirs = jnp.stack([node.direction for node in self.lane_nodes], axis=0).astype(jnp.float32)
		node_lane_ids = jnp.asarray([node.lane_id for node in self.lane_nodes], dtype=jnp.int32)

		center_diffs = node_positions - self.ego_center_xy[None, :]
		center_dists = jnp.linalg.norm(center_diffs, axis=1)
		center_lane_idx = int(jnp.argmin(center_dists))
		center_min_dist = center_dists[center_lane_idx].astype(jnp.float32)
		center_lane_id = int(node_lane_ids[center_lane_idx])

		node_dir = node_dirs[center_lane_idx]
		node_dir_norm = jnp.linalg.norm(node_dir)
		ego_heading = self.ego_heading_xy
		ego_heading_norm = jnp.linalg.norm(ego_heading)
		cos_heading = jnp.where(
				(node_dir_norm > 1e-6) & (ego_heading_norm > 1e-6),
				jnp.clip(jnp.dot(node_dir, ego_heading) / (node_dir_norm * ego_heading_norm), -1.0, 1.0),
				jnp.asarray(1.0, dtype=jnp.float32),
		)
		heading_error = jnp.arccos(cos_heading).astype(jnp.float32)

		if int(self.ego_map_corners_xy.shape[0]) == 0:
			corner_lane_indices = jnp.empty((0,), dtype=jnp.int32)
			corner_lane_ids = jnp.empty((0,), dtype=jnp.int32)
			corner_min_distances = jnp.empty((0,), dtype=jnp.float32)
			multiple_lanes = jnp.asarray(False, dtype=jnp.bool_)
			non_drivable_area = jnp.asarray(False, dtype=jnp.bool_)
		else:
			corner_diffs = self.ego_map_corners_xy[:, None, :] - node_positions[None, :, :]
			corner_dists = jnp.linalg.norm(corner_diffs, axis=-1)
			corner_lane_indices = jnp.argmin(corner_dists, axis=1).astype(jnp.int32)
			corner_min_distances = jnp.min(corner_dists, axis=1).astype(jnp.float32)
			corner_lane_ids = node_lane_ids[corner_lane_indices]
			nearby_lane_ids = node_lane_ids[center_dists <= self.drivable_area_tolerance_m]
			if int(nearby_lane_ids.shape[0]) == 0:
				unique_lane_ids = jnp.empty((0,), dtype=jnp.int32)
			else:
				unique_lane_ids = jnp.unique(nearby_lane_ids)
			multiple_lanes = jnp.asarray(int(unique_lane_ids.shape[0]) > 1, dtype=jnp.bool_)
			non_drivable_area = jnp.asarray(
					bool(jnp.any(corner_min_distances > self.drivable_area_tolerance_m)),
					dtype=jnp.bool_,
			)

		return EgoArea(
				lane_node_index=center_lane_idx,
				lane_id=center_lane_id,
				center_xy=self.ego_center_xy.astype(jnp.float32),
				collision_corners_xy=self.ego_collision_corners_xy,
				map_corners_xy=self.ego_map_corners_xy,
				collision_polygon_xy=self.ego_collision_polygon_xy,
				map_polygon_xy=self.ego_map_polygon_xy,
				corner_lane_node_indices=corner_lane_indices,
				corner_lane_ids=corner_lane_ids,
				corner_min_distances_m=corner_min_distances,
				center_min_distance_m=center_min_dist,
				heading_error_rad=heading_error,
				multiple_lanes=multiple_lanes,
				non_drivable_area=non_drivable_area,
		)

	def _build_lane_nodes(self) -> None:
		ego_xy = self._ego_position_xy()
		if ego_xy is None:
			self.lane_nodes = []
			self.lane_nodes_by_index = {}
			self.lane_nodes_by_lane = {}
			return

		if int(self.lanegraph.points_xy.shape[0]) == 0:
			self.lane_nodes = []
			self.lane_nodes_by_index = {}
			self.lane_nodes_by_lane = {}
			return

		distances = jnp.linalg.norm(self.lanegraph.points_xy - ego_xy, axis=1)
		mask = distances <= self.centerline_radius_m
		positions = self.lanegraph.points_xy[mask]
		directions = self.lanegraph.point_dirs_xy[mask]
		lane_ids = self.lanegraph.point_ids[mask]

		num_nodes = int(positions.shape[0])
		if num_nodes == 0:
			self.lane_nodes = []
			self.lane_nodes_by_index = {}
			self.lane_nodes_by_lane = {}
			return

		nodes: List[LaneNode] = []
		for idx in range(num_nodes):
			nodes.append(
					LaneNode(
							index=idx,
							lane_id=int(lane_ids[idx]),
							position=positions[idx],
							direction=directions[idx],
					),
			)

		positions_arr = jnp.asarray(positions)
		directions_arr = jnp.asarray(directions)
		outgoing_lists: List[List[int]] = [[] for _ in range(num_nodes)]
		incoming_lists: List[List[int]] = [[] for _ in range(num_nodes)]

		for idx in range(num_nodes):
			pos_i = positions_arr[idx]
			dir_i = directions_arr[idx]
			norm_dir = float(jnp.linalg.norm(dir_i))
			if norm_dir < 1e-6:
				continue

			dir_unit = dir_i / norm_dir
			diffs = positions_arr - pos_i
			perp = jnp.abs(diffs[:, 0] * dir_unit[1] - diffs[:, 1] * dir_unit[0])
			proj = diffs[:, 0] * dir_unit[0] + diffs[:, 1] * dir_unit[1]
			euclid = jnp.sqrt(jnp.sum(diffs * diffs, axis=1))
			candidate_mask = (perp <= self.centerline_lateral_tolerance_m) & (proj > 0.0) & (euclid > 0.0)
			if not bool(jnp.any(candidate_mask)):
				continue

			candidate_dists = euclid[candidate_mask]
			min_dist = float(jnp.min(candidate_dists))
			near_mask = candidate_mask & (euclid <= min_dist + 1e-3)
			outgoing_idx = jnp.where(near_mask)[0].tolist()

			for out_idx in outgoing_idx:
				outgoing_lists[idx].append(int(out_idx))
				incoming_lists[int(out_idx)].append(idx)

		for idx in range(num_nodes):
			nodes[idx].outgoing = outgoing_lists[idx]
			nodes[idx].incoming = incoming_lists[idx]

		self.lane_nodes = nodes
		self.lane_nodes_by_index = {node.index: node for node in nodes}
		lane_map: Dict[int, List[LaneNode]] = {}
		for node in nodes:
			lane_map.setdefault(node.lane_id, []).append(node)
		self.lane_nodes_by_lane = lane_map

	@property
	def centerline_point_ids(self) -> jax.Array:
		return self.lanegraph.point_ids

	@property
	def centerline_point_types(self) -> jax.Array:
		return self.lanegraph.point_types

	@property
	def centerline_points_xy(self) -> jax.Array:
		return self.lanegraph.points_xy

	@property
	def centerline_point_dirs_xy(self) -> jax.Array:
		return self.lanegraph.point_dirs_xy

	@property
	def centerline_lane_ids(self) -> jax.Array:
		return self.lanegraph.lane_ids

	@property
	def ego_lane_node_index(self) -> Optional[int]:
		return self.ego_area.lane_node_index

	@property
	def ego_lane_id(self) -> Optional[int]:
		return self.ego_area.lane_id

	@property
	def ego_in_multiple_lanes(self) -> bool:
		return bool(self.ego_area.multiple_lanes)

	@property
	def ego_in_non_drivable_area(self) -> bool:
		return bool(self.ego_area.non_drivable_area)

	@property
	def num_centerline_points(self) -> int:
		return int(self.centerline_points_xy.shape[0])

	@property
	def num_centerline_lanes(self) -> int:
		return int(self.lanegraph.lane_ids.shape[0])

	@property
	def ego_absolute_index(self) -> int:
		return 0

	def absolute_index(self, object_id: int) -> Optional[int]:
		return self.vehiclegraph.object_id_to_abs_index.get(int(object_id))

	def get_centerline_points(self, lane_id: int) -> jax.Array:
		return self.lanegraph.points_by_lane.get(int(lane_id), jnp.empty((0, 2), dtype=jnp.float32))

	def get_centerline_dirs(self, lane_id: int) -> jax.Array:
		return self.lanegraph.point_dirs_by_lane.get(int(lane_id), jnp.empty((0, 2), dtype=jnp.float32))

	def get_current_lane(self, vehicle_idx: int) -> Optional[int]:
		if vehicle_idx < 0 or vehicle_idx >= int(self.vehiclegraph.x.shape[0]):
			return None

		if int(self.vehiclegraph.is_valid.shape[0]) > vehicle_idx and not bool(self.vehiclegraph.is_valid[vehicle_idx]):
			return None

		if not self.lane_nodes:
			return None

		veh_xy = jnp.stack([self.vehiclegraph.x[vehicle_idx], self.vehiclegraph.y[vehicle_idx]], axis=0)
		positions = jnp.stack([node.position for node in self.lane_nodes], axis=0)
		dists = jnp.linalg.norm(positions - veh_xy, axis=1)
		nearest = int(jnp.argmin(dists))
		return int(self.lane_nodes[nearest].index)

	def _choose_next_node(self, current_idx: int, candidates: List[int]) -> Optional[int]:
		if not candidates:
			return None

		current_node = self.lane_nodes[current_idx]
		curr_dir = jnp.asarray(current_node.direction)
		curr_norm = float(jnp.linalg.norm(curr_dir))
		if curr_norm < 1e-6:
			return None

		best_idx: Optional[int] = None
		best_cos = -1.0
		for cand_idx in candidates:
			cand_node = self.lane_nodes[cand_idx]
			if cand_node.lane_id != current_node.lane_id:
				continue

			cand_dir = jnp.asarray(cand_node.direction)
			cand_norm = float(jnp.linalg.norm(cand_dir))
			if cand_norm < 1e-6:
				continue

			cos_sim = float(jnp.dot(curr_dir, cand_dir) / (curr_norm * cand_norm))
			if cos_sim > best_cos:
				best_cos = cos_sim
				best_idx = cand_idx

		return best_idx

	def extend_centerline(self, lane_idx: int) -> Tuple[List[int], jax.Array, jax.Array]:
		if lane_idx < 0 or lane_idx >= len(self.lane_nodes):
			return [], jnp.empty((0, 2), dtype=jnp.float32), jnp.empty((0, 2), dtype=jnp.float32)

		start_node = self.lane_nodes[lane_idx]
		target_lane_id = start_node.lane_id
		if target_lane_id is None:
			return [], jnp.empty((0, 2), dtype=jnp.float32), jnp.empty((0, 2), dtype=jnp.float32)

		visited = set([lane_idx])
		forward: List[int] = []
		current = lane_idx
		while True:
			next_idx = self._choose_next_node(current, self.lane_nodes[current].outgoing)
			if next_idx is None or next_idx in visited:
				break
			if self.lane_nodes[next_idx].lane_id != target_lane_id:
				break
			forward.append(next_idx)
			visited.add(next_idx)
			current = next_idx

		backward: List[int] = []
		current = lane_idx
		while True:
			next_idx = self._choose_next_node(current, self.lane_nodes[current].incoming)
			if next_idx is None or next_idx in visited:
				break
			if self.lane_nodes[next_idx].lane_id != target_lane_id:
				break
			backward.append(next_idx)
			visited.add(next_idx)
			current = next_idx

		ordered_indices = list(reversed(backward)) + [lane_idx] + forward
		positions = jnp.stack([self.lane_nodes[idx].position for idx in ordered_indices], axis=0).astype(jnp.float32)
		directions = jnp.stack([self.lane_nodes[idx].direction for idx in ordered_indices], axis=0).astype(jnp.float32)
		return ordered_indices, positions, directions

	def _get_lateral_lane_node(self, lane_idx: int, side: str) -> Optional[int]:
		if lane_idx < 0 or lane_idx >= len(self.lane_nodes):
			return None

		if side not in {"left", "right"}:
			return None

		if not self.lane_nodes:
			return None

		current = self.lane_nodes[lane_idx]
		curr_dir = jnp.asarray(current.direction)
		curr_pos = jnp.asarray(current.position)
		curr_norm = float(jnp.linalg.norm(curr_dir))
		if curr_norm < 1e-6:
			return None
		curr_dir_unit = curr_dir / curr_norm

		dist_threshold = 5.0
		dir_cos_threshold = 0.7

		best_idx: Optional[int] = None
		best_dist = float("inf")
		for cand_idx, cand_node in enumerate(self.lane_nodes):
			if cand_idx == lane_idx:
				continue
			if cand_node.lane_id == current.lane_id:
				continue

			cand_dir = jnp.asarray(cand_node.direction)
			cand_norm = float(jnp.linalg.norm(cand_dir))
			if cand_norm < 1e-6:
				continue
			cand_dir_unit = cand_dir / cand_norm

			dir_cos = float(jnp.dot(curr_dir_unit, cand_dir_unit))
			if dir_cos < dir_cos_threshold:
				continue

			vec = jnp.asarray(cand_node.position) - curr_pos
			signed_lateral = float(curr_dir_unit[0] * vec[1] - curr_dir_unit[1] * vec[0])
			if side == "left" and signed_lateral <= 0.0:
				continue
			if side == "right" and signed_lateral >= 0.0:
				continue

			dist = float(jnp.linalg.norm(vec))
			if dist > dist_threshold:
				continue

			if dist < best_dist:
				best_dist = dist
				best_idx = cand_idx

		return best_idx

	def get_left_lane(self, lane_idx: int) -> Optional[int]:
		return self._get_lateral_lane_node(lane_idx=lane_idx, side="left")

	def get_right_lane(self, lane_idx: int) -> Optional[int]:
		return self._get_lateral_lane_node(lane_idx=lane_idx, side="right")

	@classmethod
	def from_simulator_state(
			cls,
			simulator_state: datatypes.SimulatorState,
			batch_index: Optional[Union[int, Sequence[int]]] = None,
			prediction_horizon: int = 10,
			prediction_dt_s: float = 0.1,
			centerline_radius_m: float = 150.0,
			centerline_lateral_tolerance_m: float = 0.5,
			drivable_area_tolerance_m: float = 1.5,
			drivable_area_reduction_m: float = 0.0,
			collision_scale: float = 1.0,
	) -> "World":
		return cls(
				simulator_state=simulator_state,
				batch_index=batch_index,
				prediction_horizon=prediction_horizon,
				prediction_dt_s=prediction_dt_s,
				centerline_radius_m=centerline_radius_m,
				centerline_lateral_tolerance_m=centerline_lateral_tolerance_m,
				drivable_area_tolerance_m=drivable_area_tolerance_m,
				drivable_area_reduction_m=drivable_area_reduction_m,
				collision_scale=collision_scale,
		)


def build_world(
		simulator_state: datatypes.SimulatorState,
		batch_index: Optional[Union[int, Sequence[int]]] = None,
		prediction_horizon: int = 10,
		prediction_dt_s: float = 0.1,
		centerline_radius_m: float = 150.0,
		centerline_lateral_tolerance_m: float = 0.5,
		drivable_area_tolerance_m: float = 1.5,
		drivable_area_reduction_m: float = 0.0,
		collision_scale: float = 1.0,
) -> World:
	return World(
			simulator_state=simulator_state,
			batch_index=batch_index,
			prediction_horizon=prediction_horizon,
			prediction_dt_s=prediction_dt_s,
			centerline_radius_m=centerline_radius_m,
			centerline_lateral_tolerance_m=centerline_lateral_tolerance_m,
			drivable_area_tolerance_m=drivable_area_tolerance_m,
			drivable_area_reduction_m=drivable_area_reduction_m,
			collision_scale=collision_scale,
	)
