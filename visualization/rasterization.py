from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np


BACKGROUND_RGB = (255, 255, 255)
BASELINE_PATHS_RGB = (150, 75, 0)
AGENTS_RGB = (113, 100, 222)
EGO_RGB = (82, 86, 92)
TARGET_TRAJECTORY_RGB = (61, 160, 179)
PREDICTED_TRAJECTORY_RGB = (158, 63, 120)

_OBJECT_TYPE_PALETTE = {
    0: (128, 128, 128),  # unset
    1: (0, 0, 255),      # vehicle
    2: (30, 180, 30),    # pedestrian
    3: (255, 140, 0),    # cyclist
    4: (100, 100, 100),  # other
}


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _select_example(state: Any) -> Any:
    shape = tuple(getattr(state, "shape", ()))
    if len(shape) == 0:
        return state

    def _select_leaf(v: Any) -> Any:
        if hasattr(v, "shape"):
            arr = _to_numpy(v)
            if arr.ndim >= 1:
                return v[0]
        return v

    def _select_dataclass(dc: Any) -> Any:
        if dc is None:
            return None
        if dataclasses.is_dataclass(dc):
            values = {f.name: _select_leaf(getattr(dc, f.name)) for f in dataclasses.fields(dc)}
            return dataclasses.replace(dc, **values)
        if hasattr(dc, "replace") and hasattr(dc, "__dataclass_fields__"):
            values = {name: _select_leaf(getattr(dc, name)) for name in dc.__dataclass_fields__}
            return dc.replace(**values)
        return dc

    values = {}
    for field_name in ("sim_trajectory", "log_trajectory", "log_traffic_light", "object_metadata", "sdc_paths", "roadgraph_points"):
        if hasattr(state, field_name):
            values[field_name] = _select_dataclass(getattr(state, field_name))
    if hasattr(state, "timestep"):
        values["timestep"] = _select_leaf(getattr(state, "timestep"))

    if dataclasses.is_dataclass(state):
        return dataclasses.replace(state, **values)
    if hasattr(state, "replace"):
        return state.replace(**values)
    for key, value in values.items():
        setattr(state, key, value)
    return state


def _get_current_timestep(state: Any) -> int:
    timestep = int(_to_numpy(state.timestep))
    return timestep


def _world_to_local_xy(xy_world: np.ndarray, origin_xy: np.ndarray, origin_yaw: float) -> np.ndarray:
    xy = _to_numpy(xy_world).astype(np.float32) - _to_numpy(origin_xy).astype(np.float32)[None, :]
    c = float(np.cos(-origin_yaw))
    s = float(np.sin(-origin_yaw))
    x = xy[..., 0]
    y = xy[..., 1]
    xr = x * c - y * s
    yr = x * s + y * c
    return np.stack([xr, yr], axis=-1)


def _metric_to_pixel(
    xy_local: np.ndarray,
    radius: float,
    pixel_size: float,
    size: int,
    flip_y: bool = False,
) -> np.ndarray:
    px = (radius + xy_local[..., 0]) / pixel_size
    py = (radius + xy_local[..., 1]) / pixel_size
    if flip_y:
        py = (size - 1) - py
    return np.stack([px, py], axis=-1).astype(np.float32)


def _draw_trajectory(
    image: np.ndarray,
    trajectory_world_xy: np.ndarray,
    origin_xy: np.ndarray,
    origin_yaw: float,
    color_rgb: tuple[int, int, int],
    radius_m: float,
    pixel_size: float,
    point_radius: int = 3,
    thickness: int = 2,
) -> None:
    size = image.shape[0]
    local = _world_to_local_xy(trajectory_world_xy, origin_xy, origin_yaw)
    pix = _metric_to_pixel(local, radius=radius_m, pixel_size=pixel_size, size=size, flip_y=True)
    pts = np.round(pix).astype(np.int32)
    in_view = (
        (pts[:, 0] >= 0) & (pts[:, 0] < size) &
        (pts[:, 1] >= 0) & (pts[:, 1] < size)
    )
    pts = pts[in_view]
    if pts.shape[0] == 0:
        return
    for p in pts:
        cv2.circle(image, (int(p[0]), int(p[1])), point_radius, color_rgb, -1)
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_rgb, thickness)


def _box_corners_xy(length: float, width: float, yaw: float, center_xy: np.ndarray) -> np.ndarray:
    hl = float(length) * 0.5
    hw = float(width) * 0.5
    corners = np.array(
        [[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]],
        dtype=np.float32,
    )
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + center_xy[None, :]


def _create_map_raster_from_roadgraph(
    roadgraph_points: Any,
    origin_xy: np.ndarray,
    origin_yaw: float,
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    thickness: int = 2,
) -> np.ndarray:
    layer = np.zeros((size, size), dtype=np.uint8)
    if roadgraph_points is None:
        return layer

    x = _to_numpy(roadgraph_points.x)
    y = _to_numpy(roadgraph_points.y)
    ids = _to_numpy(roadgraph_points.ids).astype(np.int64)
    valid = _to_numpy(roadgraph_points.valid).astype(bool)

    xy_world = np.stack([x, y], axis=-1).astype(np.float32)
    xy_local = _world_to_local_xy(xy_world, origin_xy, origin_yaw)
    in_range = (
        (xy_local[:, 0] >= -radius) & (xy_local[:, 0] <= radius) &
        (xy_local[:, 1] >= -radius) & (xy_local[:, 1] <= radius)
    )
    keep = valid & in_range
    if not np.any(keep):
        return layer

    xy_local = xy_local[keep]
    ids = ids[keep]
    xy_pix = _metric_to_pixel(xy_local, radius=radius, pixel_size=pixel_size, size=size, flip_y=False)
    xy_shifted = np.round(xy_pix * (2**bit_shift)).astype(np.int64)

    grouped: dict[int, list[np.ndarray]] = {}
    for pt, lane_id in zip(xy_shifted, ids):
        grouped.setdefault(int(lane_id), []).append(pt)

    for points in grouped.values():
        arr = np.asarray(points, dtype=np.int64)
        if arr.shape[0] < 2:
            continue
        cv2.polylines(
            layer,
            [arr],
            isClosed=False,
            color=1,
            thickness=thickness,
            shift=bit_shift,
            lineType=cv2.LINE_AA,
        )
    return np.ascontiguousarray(np.flipud(layer), dtype=np.uint8)


def _create_agents_raster_from_trajectory(
    trajectory: Any,
    object_metadata: Any,
    timestep: int,
    ego_idx: int,
    origin_xy: np.ndarray,
    origin_yaw: float,
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    agent_color_by_type: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    mask = np.zeros((size, size), dtype=np.uint8)
    color_layer = np.zeros((size, size, 3), dtype=np.uint8) if agent_color_by_type else None

    x = _to_numpy(trajectory.x[:, timestep]).astype(np.float32)
    y = _to_numpy(trajectory.y[:, timestep]).astype(np.float32)
    yaw = _to_numpy(trajectory.yaw[:, timestep]).astype(np.float32)
    length = _to_numpy(trajectory.length[:, timestep]).astype(np.float32)
    width = _to_numpy(trajectory.width[:, timestep]).astype(np.float32)
    valid = _to_numpy(trajectory.valid[:, timestep]).astype(bool)
    obj_types = _to_numpy(object_metadata.object_types).astype(np.int32)

    xy_world = np.stack([x, y], axis=-1)
    xy_local = _world_to_local_xy(xy_world, origin_xy, origin_yaw)

    for i in range(xy_local.shape[0]):
        if i == ego_idx or not valid[i]:
            continue
        cx, cy = xy_local[i]
        if abs(cx) > radius or abs(cy) > radius:
            continue

        local_yaw = float(yaw[i] - origin_yaw)
        corners = _box_corners_xy(length[i], width[i], local_yaw, np.array([cx, cy], dtype=np.float32))
        corners_px = _metric_to_pixel(corners, radius=radius, pixel_size=pixel_size, size=size, flip_y=False)
        corners_shifted = np.round(corners_px * (2**bit_shift)).astype(np.int64)
        corners_shifted[:, 0] = np.clip(corners_shifted[:, 0], 0, ((size - 1) * (2**bit_shift)))
        corners_shifted[:, 1] = np.clip(corners_shifted[:, 1], 0, ((size - 1) * (2**bit_shift)))

        cv2.fillPoly(mask, [corners_shifted], color=1, shift=bit_shift, lineType=cv2.LINE_AA)
        if color_layer is not None:
            rgb = _OBJECT_TYPE_PALETTE.get(int(obj_types[i]), _OBJECT_TYPE_PALETTE[0])
            cv2.fillPoly(color_layer, [corners_shifted], color=rgb, shift=bit_shift, lineType=cv2.LINE_AA)

    mask = np.ascontiguousarray(np.flipud(mask), dtype=np.uint8)
    if color_layer is not None:
        color_layer = np.ascontiguousarray(np.flipud(color_layer), dtype=np.uint8)
    return mask, color_layer


def _create_ego_raster_from_state(
    trajectory: Any,
    timestep: int,
    ego_idx: int,
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
) -> np.ndarray:
    layer = np.zeros((size, size), dtype=np.uint8)
    length = float(_to_numpy(trajectory.length[ego_idx, timestep]))
    width = float(_to_numpy(trajectory.width[ego_idx, timestep]))
    valid = bool(_to_numpy(trajectory.valid[ego_idx, timestep]))
    if not valid:
        return layer

    corners = _box_corners_xy(length, width, 0.0, np.array([0.0, 0.0], dtype=np.float32))
    corners_px = _metric_to_pixel(corners, radius=radius, pixel_size=pixel_size, size=size, flip_y=False)
    corners_shifted = np.round(corners_px * (2**bit_shift)).astype(np.int64)
    corners_shifted[:, 0] = np.clip(corners_shifted[:, 0], 0, ((size - 1) * (2**bit_shift)))
    corners_shifted[:, 1] = np.clip(corners_shifted[:, 1], 0, ((size - 1) * (2**bit_shift)))
    cv2.fillPoly(layer, [corners_shifted], color=1, shift=bit_shift, lineType=cv2.LINE_AA)
    return np.ascontiguousarray(np.flipud(layer), dtype=np.uint8)


def _normalize_trajectory_input(trajectory: Any) -> np.ndarray:
    arr = _to_numpy(trajectory).astype(np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] < 2:
            raise ValueError(f"Trajectory must have last dim >= 2, got shape {arr.shape}")
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr
        if arr.shape[-1] < 2:
            raise ValueError(f"Trajectory must have last dim >= 2, got shape {arr.shape}")
        return arr
    raise ValueError(f"Unsupported trajectory shape {arr.shape}; expected (T,D) or (K,T,D).")


def _probability_to_rgb(prob: float) -> tuple[int, int, int]:
    p = float(np.clip(prob, 0.0, 1.0))
    # Blue (low) to green (high).
    r = 20
    g = int(80 + 175 * p)
    b = int(255 - 200 * p)
    return (r, g, b)


def get_raster_from_simulator_state_with_agents(
    state: Any,
    target_trajectory: Any = None,
    predicted_trajectory: Any = None,
    probabilities: Any = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    use_log_trajectory: bool = False,
    map_thickness: int = 2,
    agent_color_by_type: bool = False,
) -> np.ndarray:
    state = _select_example(state)
    traj = state.log_trajectory if use_log_trajectory else state.sim_trajectory
    t = _get_current_timestep(state)

    is_sdc = _to_numpy(state.object_metadata.is_sdc).astype(bool)
    if is_sdc.ndim != 1:
        raise ValueError(f"Expected unbatched object_metadata.is_sdc with shape (N,), got {is_sdc.shape}")
    if not np.any(is_sdc):
        raise ValueError("No valid SDC index found in state.object_metadata.is_sdc.")
    ego_idx = int(np.argmax(is_sdc.astype(np.int32)))

    valid_at_t = bool(_to_numpy(traj.valid[ego_idx, t]))
    if not valid_at_t:
        raise ValueError(f"SDC is invalid at timestep {t}.")

    origin_xy = np.array(
        [
            float(_to_numpy(traj.x[ego_idx, t])),
            float(_to_numpy(traj.y[ego_idx, t])),
        ],
        dtype=np.float32,
    )
    origin_yaw = float(_to_numpy(traj.yaw[ego_idx, t]))

    size = int(2 * radius / pixel_size)
    image = np.full((size, size, 3), BACKGROUND_RGB, dtype=np.uint8)

    map_raster = _create_map_raster_from_roadgraph(
        roadgraph_points=getattr(state, "roadgraph_points", None),
        origin_xy=origin_xy,
        origin_yaw=origin_yaw,
        radius=radius,
        size=size,
        bit_shift=bit_shift,
        pixel_size=pixel_size,
        thickness=map_thickness,
    )
    image[map_raster > 0] = BASELINE_PATHS_RGB

    agents_mask, agents_layer = _create_agents_raster_from_trajectory(
        trajectory=traj,
        object_metadata=state.object_metadata,
        timestep=t,
        ego_idx=ego_idx,
        origin_xy=origin_xy,
        origin_yaw=origin_yaw,
        radius=radius,
        size=size,
        bit_shift=bit_shift,
        pixel_size=pixel_size,
        agent_color_by_type=agent_color_by_type,
    )
    if agents_layer is None:
        image[agents_mask > 0] = AGENTS_RGB
    else:
        mask = np.any(agents_layer > 0, axis=-1)
        image[mask] = agents_layer[mask]

    ego_mask = _create_ego_raster_from_state(
        trajectory=traj,
        timestep=t,
        ego_idx=ego_idx,
        radius=radius,
        size=size,
        bit_shift=bit_shift,
        pixel_size=pixel_size,
    )
    image[ego_mask > 0] = EGO_RGB

    if predicted_trajectory is not None:
        pred = _normalize_trajectory_input(predicted_trajectory)
        if pred.shape[0] == 1:
            _draw_trajectory(
                image=image,
                trajectory_world_xy=pred[0, :, :2],
                origin_xy=origin_xy,
                origin_yaw=origin_yaw,
                color_rgb=PREDICTED_TRAJECTORY_RGB,
                radius_m=radius,
                pixel_size=pixel_size,
            )
        else:
            if probabilities is not None:
                probs = _to_numpy(probabilities).astype(np.float32).reshape(-1)
                if probs.shape[0] != pred.shape[0]:
                    raise ValueError(
                        f"probabilities length ({probs.shape[0]}) must match number of trajectories ({pred.shape[0]})."
                    )
                order = np.argsort(probs)
                for idx in order:
                    _draw_trajectory(
                        image=image,
                        trajectory_world_xy=pred[idx, :, :2],
                        origin_xy=origin_xy,
                        origin_yaw=origin_yaw,
                        color_rgb=_probability_to_rgb(float(probs[idx])),
                        radius_m=radius,
                        pixel_size=pixel_size,
                        point_radius=2,
                        thickness=1,
                    )
            else:
                for k in range(pred.shape[0]):
                    _draw_trajectory(
                        image=image,
                        trajectory_world_xy=pred[k, :, :2],
                        origin_xy=origin_xy,
                        origin_yaw=origin_yaw,
                        color_rgb=PREDICTED_TRAJECTORY_RGB,
                        radius_m=radius,
                        pixel_size=pixel_size,
                        point_radius=2,
                        thickness=1,
                    )

    if target_trajectory is not None:
        target = _normalize_trajectory_input(target_trajectory)
        _draw_trajectory(
            image=image,
            trajectory_world_xy=target[0, :, :2],
            origin_xy=origin_xy,
            origin_yaw=origin_yaw,
            color_rgb=TARGET_TRAJECTORY_RGB,
            radius_m=radius,
            pixel_size=pixel_size,
        )

    return image

