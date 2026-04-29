from __future__ import annotations

import argparse
import dataclasses
import math
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib.patches import Polygon
from glob import glob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diffusion_planner import DiffusionPlanner, PlannerResult
from model.diffusion_es_planner import DiffusionESPlanner, ESPlannerResult
from scores import helpers as scorer_helpers
from scores.scorer_lane_graph import LaneGraphShardStore, Scorer
from viz import viz as viz_module
from viz.render import _load_scenario_state_fast
from waymax import config as waymax_config
from simulation_utils.utils import (
    _lane_graph_zip_path_for_tfrecord,
    _save_json
)

def extract_scene_qa(
    sim_state,
    scorer,
    *,
    world_idx: int,
    timestep: int = 0,
) -> dict[str, Any]:
    left_lane = scorer_helpers.get_left_lane(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    right_lane = scorer_helpers.get_right_lane(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    vehicle_current_lane_ids = scorer_helpers.get_vehicle_current_lane_ids(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    front_vehicles = scorer_helpers.get_vehicle_front(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
        vehicle_current_lane_ids=vehicle_current_lane_ids,
    )
    behind_vehicles = scorer_helpers.get_vehicle_behind(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
        vehicle_current_lane_ids=vehicle_current_lane_ids,
    )
    left_vehicles = scorer_helpers.get_vehicle_left(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    right_vehicles = scorer_helpers.get_vehicle_right(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    front_pedestrians = scorer_helpers.get_pedestrian_front(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )
    traffic_light_state = scorer_helpers.get_traffic_light_state_ahead(
        scorer,
        sim_state,
        timestep=timestep,
        world_idx=world_idx,
    )

    answers = {
        "has_left_lane": bool(left_lane is not None),
        "has_right_lane": bool(right_lane is not None),
        "num_vehicle_front_same_lane": len(front_vehicles),
        "num_vehicle_behind_same_lane": len(behind_vehicles),
        "num_vehicle_left": len(left_vehicles),
        "num_vehicle_right": len(right_vehicles),
        "num_pedestrian_front": len(front_pedestrians),
        "traffic_light_state": int(traffic_light_state),
    }

    return {
        "timestep": int(timestep),
        "answers": answers,
    }


def _transform_world_to_ego(xy_t2: np.ndarray, ego_xy_2: np.ndarray, ego_yaw: float, align_heading: bool) -> np.ndarray:
    rel_xy = np.asarray(xy_t2, dtype=np.float32) - np.asarray(ego_xy_2, dtype=np.float32)[None, :]
    if not align_heading:
        return rel_xy
    rot_rad = (0.5 * math.pi) - float(ego_yaw)
    c = math.cos(rot_rad)
    s = math.sin(rot_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return rel_xy @ rot.T


def _replace_xy_fields_safe(obj, xy_local: np.ndarray):
    kwargs = {}
    if hasattr(obj, "x"):
        kwargs["x"] = np.asarray(xy_local)[..., 0]
    if hasattr(obj, "y"):
        kwargs["y"] = np.asarray(xy_local)[..., 1]
    if kwargs:
        return obj.replace(**kwargs)
    if hasattr(obj, "xy"):
        return obj.replace(xy=np.asarray(xy_local))
    return obj


def _vehicle_box_polygon(center_xy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    half_length = 0.5 * float(length)
    half_width = 0.5 * float(width)
    corners = np.array(
        [[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]],
        dtype=np.float32,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.asarray(center_xy, dtype=np.float32)[None, :]


def _plot_lane_centerlines_by_tl_control(
    ax,
    roadgraph_local,
    traffic_lights_local,
    *,
    timestep: int,
    red_states: tuple[int, ...] = (4,),
) -> None:
    if roadgraph_local is None:
        return

    rg_valid = np.asarray(roadgraph_local.valid).astype(bool)
    rg_ids = np.asarray(roadgraph_local.ids)
    rg_types = np.asarray(roadgraph_local.types)
    rg_xy = np.asarray(roadgraph_local.xy)

    lane_center_mask = np.isin(rg_types, np.array([1, 2, 3], dtype=rg_types.dtype))
    lane_mask = rg_valid & lane_center_mask
    if not np.any(lane_mask):
        return

    red_lane_ids = set()
    if traffic_lights_local is not None:
        tl_valid = np.asarray(traffic_lights_local.valid[:, timestep]).astype(bool)
        if np.any(tl_valid):
            tl_state = np.asarray(traffic_lights_local.state[:, timestep])
            tl_lane_ids = np.asarray(traffic_lights_local.lane_ids[:, timestep])
            red_mask = tl_valid & np.isin(tl_state, np.asarray(red_states, dtype=tl_state.dtype)) & (tl_lane_ids > 0)
            red_lane_ids = set(np.asarray(tl_lane_ids[red_mask]).tolist())

    for lane_id in np.unique(rg_ids[lane_mask]):
        lane_points = rg_xy[lane_mask & (rg_ids == lane_id)]
        if lane_points.shape[0] < 2:
            continue
        if int(lane_id) in red_lane_ids:
            color = "#ff9aa2"
            linewidth = 2.2
            zorder = 4.2
        else:
            color = "#b7e4c7"
            linewidth = 2.2
            zorder = 4.0
        ax.plot(
            lane_points[:, 0],
            lane_points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            solid_capstyle="round",
            zorder=zorder,
        )


def render_bev_t0_ego_centered(
    state_batch,
    *,
    out_path: Path,
    batch_idx: int = 0,
    timestep: int = 0,
    front_x: float = 30.0,
    back_x: float = 30.0,
    left_y: float = 30.0,
    right_y: float = 30.0,
    align_heading: bool = True,
    draw_stop_controlled_lanes: bool = True,
    dpi: int = 200,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> None:
    state_b = viz_module._index_pytree(state_batch, batch_idx)
    ego_idx = scorer_helpers.get_ego_idx(state_batch, batch_idx)
    ego_valid = np.asarray(state_batch.log_trajectory.valid[batch_idx, ego_idx]).astype(bool)
    ego_ref_t = int(timestep)
    if ego_ref_t >= ego_valid.shape[0] or not bool(ego_valid[ego_ref_t]):
        valid_steps = np.flatnonzero(ego_valid)
        ego_ref_t = int(valid_steps[0]) if valid_steps.size > 0 else 0

    ego_xy = np.asarray(state_batch.log_trajectory.xy[batch_idx, ego_idx, ego_ref_t], dtype=np.float32)
    ego_yaw = float(np.asarray(state_batch.log_trajectory.yaw[batch_idx, ego_idx, ego_ref_t]))

    obj_xy = np.asarray(state_batch.log_trajectory.xy[batch_idx, :, timestep, :], dtype=np.float32)
    obj_yaw = np.asarray(state_batch.log_trajectory.yaw[batch_idx, :, timestep], dtype=np.float32)
    obj_length = np.asarray(state_batch.log_trajectory.length[batch_idx, :, timestep], dtype=np.float32)
    obj_width = np.asarray(state_batch.log_trajectory.width[batch_idx, :, timestep], dtype=np.float32)
    obj_valid = np.asarray(state_batch.log_trajectory.valid[batch_idx, :, timestep]).astype(bool)
    obj_is_ego = np.asarray(state_batch.object_metadata.is_sdc[batch_idx]).astype(bool)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    roadgraph_local = None
    if hasattr(state_b, "roadgraph_points") and state_b.roadgraph_points is not None:
        rg_world_xy = np.stack(
            [np.asarray(state_b.roadgraph_points.x), np.asarray(state_b.roadgraph_points.y)],
            axis=-1,
        )
        rg_xy_local = _transform_world_to_ego(rg_world_xy, ego_xy, ego_yaw, align_heading)
        roadgraph_local = _replace_xy_fields_safe(state_b.roadgraph_points, rg_xy_local)
        viz_module.plot_roadgraph_points(ax, roadgraph_local, verbose=False)

    traffic_lights_local = None
    if hasattr(state_b, "log_traffic_light") and state_b.log_traffic_light is not None:
        tl_world_xy = np.stack(
            [np.asarray(state_b.log_traffic_light.x), np.asarray(state_b.log_traffic_light.y)],
            axis=-1,
        )
        tl_xy_local = _transform_world_to_ego(tl_world_xy, ego_xy, ego_yaw, align_heading)
        traffic_lights_local = _replace_xy_fields_safe(state_b.log_traffic_light, tl_xy_local)
        viz_module.plot_traffic_light_signals_as_points(ax, traffic_lights_local, timestep=timestep, verbose=False)

    _plot_lane_centerlines_by_tl_control(
        ax,
        roadgraph_local,
        traffic_lights_local,
        timestep=timestep,
        red_states=(4,),
    )

    obj_xy_local = _transform_world_to_ego(obj_xy, ego_xy, ego_yaw, align_heading)
    for obj_idx in range(obj_xy.shape[0]):
        if not bool(obj_valid[obj_idx]):
            continue
        if align_heading:
            yaw_local = float(obj_yaw[obj_idx] - ego_yaw + 0.5 * math.pi)
        else:
            yaw_local = float(obj_yaw[obj_idx])
        polygon = _vehicle_box_polygon(obj_xy_local[obj_idx], yaw_local, float(obj_length[obj_idx]), float(obj_width[obj_idx]))
        face_color = "tab:red" if bool(obj_is_ego[obj_idx]) else "tab:blue"
        alpha = 0.95 if bool(obj_is_ego[obj_idx]) else 0.72
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=face_color,
                edgecolor="black",
                linewidth=0.55,
                alpha=alpha,
                zorder=6,
            )
        )

    ax.scatter([0.0], [0.0], c="yellow", edgecolors="black", s=80, zorder=8)
    ax.set_xlim(-float(back_x), float(front_x))
    ax.set_ylim(-float(right_y), float(left_y))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", alpha=0.35, linestyle="-")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def run(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tfrecord_paths = sorted(glob(str(Path(args.tfrecord_dir) / "*")))
    lane_graph_dir = getattr(args, "lane_graph_dir", None)
    if lane_graph_dir is None:
        raise ValueError("lane_graph_dir is required to extract lane-based QA data.")
    lane_graph_store: LaneGraphShardStore | None = None
    lane_graph_zip_paths = {
        str(tfrecord_path): _lane_graph_zip_path_for_tfrecord(str(tfrecord_path), lane_graph_dir)
        for tfrecord_path in tfrecord_paths
    }
    for tfrecord_path, lane_graph_zip_path in lane_graph_zip_paths.items():
        if lane_graph_zip_path is not None and not lane_graph_zip_path.exists():
            raise FileNotFoundError(
                f"Lane graph zip file {lane_graph_zip_path} not found for tfrecord {tfrecord_path}."
            )

    written_json_paths: list[Path] = []
    for tfrecord_path in tfrecord_paths:
        ds_cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=str(tfrecord_path),
            max_num_objects=int(args.max_num_objects),
            batch_dims=(int(args.num_worlds),),
            shuffle_seed=0,
        )

        lane_graph_zip_path = _lane_graph_zip_path_for_tfrecord(str(tfrecord_path), lane_graph_dir)
        lane_graph_store = LaneGraphShardStore(lane_graph_zip_path.as_posix())
        if (output_dir / f"{Path(tfrecord_path).name}.scenario_00000.t0.png").exists():
            continue

        try:
            first_state_batch = _load_scenario_state_fast(ds_cfg, 0)
            render_bev_t0_ego_centered(
                first_state_batch,
                out_path=output_dir / f"{Path(tfrecord_path).name}.scenario_00000.t0.png",
                timestep=0,
                front_x=float(getattr(args, "front_x", 30.0)),
                back_x=float(getattr(args, "back_x", 30.0)),
                left_y=float(getattr(args, "left_y", 30.0)),
                right_y=float(getattr(args, "right_y", 30.0)),
                align_heading=True,
                draw_stop_controlled_lanes=True,
            )
        except Exception as exc:
            print(f"Failed to render first scenario preview for {Path(tfrecord_path).name}: {exc}")

        scenario_qas: dict[int, dict[str, Any]] = {}
        curr_scenario_index = 0
        result_path = output_dir / f"{Path(tfrecord_path).name}.json"
        pbar = tqdm(desc=f"Extracting QA for {Path(tfrecord_path).name}", unit="scenario")
        while True:
            try:
                sim_state = _load_scenario_state_fast(ds_cfg, curr_scenario_index)
            except Exception as exc:
                # print(f"Failed to load scenario state for {Path(tfrecord_path).name}: {exc}")
                break
            scorer = Scorer()
            if lane_graph_store is not None:
                graph = lane_graph_store.get_by_record_index(int(curr_scenario_index))
                if graph is not None:
                    scorer.set_lane_graph(graph)
            scorer_helpers.update_lane_points(scorer, sim_state, world_idx=0)
            scenario_qas[int(curr_scenario_index)] = extract_scene_qa(
                sim_state,
                scorer,
                world_idx=0,
                timestep=0,
            )
            _save_json(result_path, scenario_qas)

            curr_scenario_index += 1
            pbar.update(1)

        pbar.close()
        written_json_paths.append(result_path)

    return written_json_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a question-answer dataset for the WOD scenarios.")
    parser.add_argument("--tfrecord_dir", type=str, required=True, help="Directory containing WOD tfrecord files.")
    parser.add_argument("--lane_graph_dir", type=str, required=True, help="Directory containing lane graph zip files corresponding to the tfrecords.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the output QA JSON files and scenario preview images.")
    parser.add_argument("--max_num_objects", type=int, default=64, help="Maximum number of objects to load from each scenario for QA extraction.")
    parser.add_argument("--num_worlds", type=int, default=1, help="Number of worlds (scenarios) to load in each batch. Set to 1 for simplicity in QA extraction.")
    parser.add_argument("--front_x", type=float, default=30.0, help="Distance in front of the ego vehicle to include in the BEV renderings.")
    parser.add_argument("--back_x", type=float, default=30.0, help="Distance behind the ego vehicle to include in the BEV renderings.")
    parser.add_argument("--left_y", type=float, default=30.0, help="Distance to the left of the ego vehicle to include in the BEV renderings.")
    parser.add_argument("--right_y", type=float, default=30.0, help="Distance to the right of the ego vehicle to include in the BEV renderings.")
    args = parser.parse_args()

    written_json_paths = run(args)
    print(f"Written QA JSON files: {[str(p) for p in written_json_paths]}")