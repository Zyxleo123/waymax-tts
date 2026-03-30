from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
from waymax import config as waymax_config
from waymax import dataloader
from waymax.dataloader import womd_factories

from . import viz as viz_module


def _normalize_override_lists(
    num_requests: int,
    ego_start_times: list[int | None] | None,
    ego_trajectories: list[np.ndarray | None] | None,
) -> list[tuple[int | None, np.ndarray | None]]:
    """Normalizes optional ego override lists to per-request tuples."""
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative.")

    # Backward-compatible behavior: if either top-level list is missing, do not
    # apply overrides to any request.
    if ego_start_times is None or ego_trajectories is None:
        return [(None, None) for _ in range(num_requests)]

    if len(ego_start_times) != num_requests:
        raise ValueError(
            f"ego_start_times length mismatch: expected {num_requests}, got {len(ego_start_times)}."
        )
    if len(ego_trajectories) != num_requests:
        raise ValueError(
            f"ego_trajectories length mismatch: expected {num_requests}, got {len(ego_trajectories)}."
        )

    return list(zip(ego_start_times, ego_trajectories))


def _get_sdc_index(state_batch, batch_idx: int) -> int:
    """Returns unique SDC object index for a single batched scenario."""
    is_sdc = np.asarray(state_batch.object_metadata.is_sdc[batch_idx]).astype(bool)
    sdc_indices = np.flatnonzero(is_sdc)
    if sdc_indices.size != 1:
        raise ValueError(
            f"Expected exactly one SDC object for batch_idx={batch_idx}, got {sdc_indices.size}."
        )
    return int(sdc_indices[0])


def _apply_ego_override_to_state_batch(
    state_batch,
    *,
    batch_idx: int,
    start_t: int,
    ego_traj_l5: np.ndarray,
):
    """
    Returns a copy of state_batch with ego log trajectory replaced on [start_t, start_t+L).

    ego_traj_l5 format is [L, 5] = (x, y, yaw, vel_x, vel_y).
    """
    if start_t < 0:
        raise ValueError(f"start_t must be >= 0, got {start_t}.")

    ego_traj_l5 = np.asarray(ego_traj_l5)
    if ego_traj_l5.ndim != 2 or ego_traj_l5.shape[1] != 5:
        raise ValueError(
            f"ego trajectory must have shape [L,5], got {tuple(ego_traj_l5.shape)}."
        )

    traj = state_batch.log_trajectory
    num_steps = int(traj.x.shape[-1])
    length = int(ego_traj_l5.shape[0])
    end_t = start_t + length
    if end_t > num_steps:
        raise ValueError(
            f"Ego override out of bounds: start_t={start_t}, len={length}, num_steps={num_steps}."
        )
    if length == 0:
        return state_batch

    sdc_idx = _get_sdc_index(state_batch, batch_idx)
    sl = slice(start_t, end_t)

    x = np.array(traj.x, copy=True)
    y = np.array(traj.y, copy=True)
    yaw = np.array(traj.yaw, copy=True)
    vel_x = np.array(traj.vel_x, copy=True)
    vel_y = np.array(traj.vel_y, copy=True)
    valid = np.array(traj.valid, copy=True)

    x[batch_idx, sdc_idx, sl] = ego_traj_l5[:, 0]
    y[batch_idx, sdc_idx, sl] = ego_traj_l5[:, 1]
    yaw[batch_idx, sdc_idx, sl] = ego_traj_l5[:, 2]
    vel_x[batch_idx, sdc_idx, sl] = ego_traj_l5[:, 3]
    vel_y[batch_idx, sdc_idx, sl] = ego_traj_l5[:, 4]
    valid[batch_idx, sdc_idx, sl] = True

    updated_log = traj.replace(
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        valid=valid,
    )
    return dataclasses.replace(state_batch, log_trajectory=updated_log)


def _get_reference_ego_heading_rad(state_batch, batch_idx: int) -> float:
    """Returns ego heading at first frame (fallback: first valid frame)."""
    sdc_idx = _get_sdc_index(state_batch, batch_idx)
    valid = np.asarray(state_batch.log_trajectory.valid[batch_idx, sdc_idx]).astype(bool)
    if valid.size == 0:
        return 0.0
    if valid[0]:
        t_ref = 0
    elif valid.any():
        t_ref = int(np.flatnonzero(valid)[0])
    else:
        t_ref = 0
    return float(np.asarray(state_batch.log_trajectory.yaw[batch_idx, sdc_idx, t_ref]))


def _load_scenario_state_fast(cfg: waymax_config.DatasetConfig, scenario_index: int):
    """Loads one scenario directly from TFRecord by skipping raw records first."""
    raw_ds = tf.data.TFRecordDataset([cfg.path]).skip(int(scenario_index)).take(1)
    iterator = iter(raw_ds)
    try:
        serialized = next(iterator)
    except StopIteration as exc:
        raise ValueError(
            f"scenario_index {scenario_index} is out of range for tfrecord '{cfg.path}'."
        ) from exc

    serialized = tf.expand_dims(serialized, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized, cfg)
    return womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )


def _load_scenario_state_batch_fast(
    cfg: waymax_config.DatasetConfig, scenario_indices: Iterable[int]
):
    """Loads a batch of scenarios from one TFRecord in a single dataset scan."""
    indices = [int(i) for i in scenario_indices]
    if not indices:
        raise ValueError("scenario_indices must be non-empty.")
    if any(i < 0 for i in indices):
        raise ValueError(f"scenario_indices must be >= 0, got {indices}.")

    unique_indices = sorted(set(indices))
    needed = set(unique_indices)
    selected: list[tf.Tensor] = []
    selected_indices: list[int] = []

    raw_ds = tf.data.TFRecordDataset([cfg.path])
    for raw_idx, serialized in enumerate(raw_ds):
        if raw_idx in needed:
            selected.append(serialized)
            selected_indices.append(raw_idx)
            if len(selected) == len(needed):
                break

    missing = sorted(needed.difference(selected_indices))
    if missing:
        raise ValueError(
            f"scenario_index values {missing} are out of range for tfrecord '{cfg.path}'."
        )

    serialized_batch = tf.stack(selected, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized_batch, cfg)
    state_batch = womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )
    scenario_to_batch_idx = {scenario_idx: i for i, scenario_idx in enumerate(selected_indices)}
    return state_batch, scenario_to_batch_idx


def render_video_from_tfrecord(
    *,
    tfrecord: str,
    output: str,
    scenario_index: int = 0,
    fps: int = 10,
    num_frames: int = 91,
    max_num_objects: int = 32,
    width: int = 384,
    height: int = 384,
    px_per_meter: float = 4.0,
    show_agent_id: bool = True,
    use_log_traj: bool = True,
    front_x: float = 30.0,
    back_x: float = 30.0,
    front_y: float = 30.0,
    back_y: float = 30.0,
) -> Path:
    """Render a WOMD scenario video using the local viz/ renderer."""
    if int(scenario_index) < 0:
        raise ValueError(f"scenario_index must be >= 0, got {scenario_index}.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=tfrecord,
        max_num_objects=int(max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )
    state = _load_scenario_state_fast(ds_cfg, int(scenario_index))
    num_frames = min(int(num_frames), int(state.log_trajectory.x.shape[-1]))

    viz_config = {
        "show_agent_id": bool(show_agent_id),
        "center_agent_idx": -1,
        "front_x": float(front_x),
        "back_x": float(back_x),
        "front_y": float(front_y),
        "back_y": float(back_y),
        "px_per_meter": float(px_per_meter),
    }

    frames: list[np.ndarray] = []
    for t in range(num_frames):
        state_t = dataclasses.replace(state, timestep=np.array([t], dtype=np.int32))
        frame = viz_module.plot_simulator_state(
            state_t,
            use_log_traj=bool(use_log_traj),
            viz_config=viz_config,
            batch_idx=0,
        )
        frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    imageio.mimsave(output_path.as_posix(), frames, fps=int(fps), codec="libx264")
    return output_path


def render_videos_batched(
    *,
    tfrecord_scenarios: list[tuple[str, int]],
    target_vehicles: list[int] | None = None,
    output_dir: str,
    fps: int = 10,
    num_frames: int = 91,
    max_num_objects: int = 32,
    width: int = 384,
    height: int = 384,
    px_per_meter: float = 4.0,
    show_agent_id: bool = True,
    use_log_traj: bool = True,
    ego_start_times: list[int | None] | None = None,
    ego_trajectories: list[np.ndarray | None] | None = None,
    front_x: float = 30.0,
    back_x: float = 30.0,
    front_y: float = 30.0,
    back_y: float = 30.0,
    align_ego_heading_up: bool = False,
) -> list[Path]:
    """
    Render multiple videos for arbitrary (tfrecord, scenario_index) pairs efficiently.

    Scenarios are grouped by TFRecord so each TFRecord is scanned only once, and
    requested scenarios are preprocessed in a single batch per TFRecord.
    """
    if not tfrecord_scenarios:
        return []
    if target_vehicles[0] is None:
        target_vehicles = None

    overrides = _normalize_override_lists(
        len(tfrecord_scenarios), ego_start_times, ego_trajectories
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for req_idx, (tfrecord, scenario_index) in enumerate(tfrecord_scenarios):
        if int(scenario_index) < 0:
            raise ValueError(f"scenario_index must be >= 0, got {scenario_index}.")
        grouped[str(tfrecord)].append((req_idx, int(scenario_index)))

    outputs: list[Path | None] = [None] * len(tfrecord_scenarios)
    viz_config = {
        "show_agent_id": bool(show_agent_id),
        "center_agent_idx": -1,
        "front_x": float(front_x),
        "back_x": float(back_x),
        "front_y": float(front_y),
        "back_y": float(back_y),
        "px_per_meter": float(px_per_meter),
    }

    for tfrecord, requests in grouped.items():
        unique_scenarios = sorted({scenario_index for _, scenario_index in requests})
        ds_cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=tfrecord,
            max_num_objects=int(max_num_objects),
            batch_dims=(len(unique_scenarios),),
            shuffle_seed=0,
        )
        state_batch, scenario_to_batch_idx = _load_scenario_state_batch_fast(
            ds_cfg, unique_scenarios
        )
        total_steps = int(state_batch.log_trajectory.x.shape[-1])
        steps = min(int(num_frames), total_steps)
        batch_size = len(unique_scenarios)

        for req_idx, scenario_index in requests:
            batch_idx = scenario_to_batch_idx[scenario_index]
            start_t, ego_traj = overrides[req_idx]

            state_for_render = state_batch
            request_use_log_traj = bool(use_log_traj)
            if start_t is not None and ego_traj is not None:
                state_for_render = _apply_ego_override_to_state_batch(
                    state_batch,
                    batch_idx=batch_idx,
                    start_t=int(start_t),
                    ego_traj_l5=ego_traj,
                )
                request_use_log_traj = True

            output_path = output_root / (
                f"{Path(tfrecord).name}.scenario_{scenario_index:03d}.mp4"
            )
            frames: list[np.ndarray] = []
            world_rotation_rad = 0.0
            if bool(align_ego_heading_up):
                heading0 = _get_reference_ego_heading_rad(state_for_render, batch_idx)
                world_rotation_rad = float(np.pi / 2.0 - heading0)
            target_vehicle = target_vehicles[req_idx] if target_vehicles is not None else None
            for t in range(steps):
                state_t = dataclasses.replace(
                    state_for_render, timestep=np.full((batch_size,), t, dtype=np.int32)
                )
                frame = viz_module.plot_simulator_state(
                    state_t,
                    use_log_traj=request_use_log_traj,
                    viz_config=viz_config,
                    batch_idx=batch_idx,
                    target_vehicle=target_vehicle,
                    world_rotation_rad=world_rotation_rad,
                )
                frame = cv2.resize(
                    frame, (int(width), int(height)), interpolation=cv2.INTER_AREA
                )
                frames.append(frame)
            imageio.mimsave(output_path.as_posix(), frames, fps=int(fps), codec="libx264")
            outputs[req_idx] = output_path

    return [p for p in outputs if p is not None]
