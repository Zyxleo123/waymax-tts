from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.diffusion_planner_v3 import (
    DiffusionPlanner,
    PlannerResult,
)
from train.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from train.preprocess import preprocess_simulator_state
from viz.render import _load_scenario_state_fast, render_videos_batched  # noqa: E402
from viz import viz as viz_module  # noqa: E402
import time



def _parse_int_csv(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DiffusionPlanner on Waymax scenarios, evaluate metrics, and render videos."
    )
    parser.add_argument("--checkpoint_path", type=str, default="/data/user_data/mineuih/checkpoints/diffusion_lr-0p0001_20260301_215223/epoch_0420")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--tfrecord_path", type=str, default="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example/training/training_tfexample.tfrecord-00100-of-01000")
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_es_output")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--resample_timesteps", type=int, default=5)
    parser.add_argument("--replan_interval_steps", type=int, default=5)
    parser.add_argument("--speed_limit_mps", type=float, default=None)
    parser.add_argument("--render_sample_indices", type=str, default="0")
    parser.add_argument("--rollout_sample_indices", type=str, default=None)
    parser.add_argument("--rollout_num_steps", type=int, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="overlap,offroad,sdc_progression",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=91)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--px_per_meter", type=float, default=4.0)
    parser.add_argument("--show_agent_id", action="store_true", default=True)
    parser.add_argument("--front_x", type=float, default=30.0)
    parser.add_argument("--back_x", type=float, default=30.0)
    parser.add_argument("--front_y", type=float, default=30.0)
    parser.add_argument("--back_y", type=float, default=30.0)
    parser.add_argument("--render_original", action="store_true")
    parser.add_argument("--save_predictions_npz", action="store_true")
    return parser.parse_args()


def _summarize_metric(value_bkt: np.ndarray, valid_bkt: np.ndarray) -> dict[str, Any]:
    batch_size, num_samples, _ = value_bkt.shape
    per_entry: list[list[dict[str, Any]]] = []
    for b in range(batch_size):
        scenario_entries: list[dict[str, Any]] = []
        for k in range(num_samples):
            valid = valid_bkt[b, k].astype(bool)
            if np.any(valid):
                vals = value_bkt[b, k][valid]
                final = float(vals[-1])
                mean = float(vals.mean())
                max_value = float(vals.max())
                valid_steps = int(valid.sum())
            else:
                final = float("nan")
                mean = float("nan")
                max_value = float("nan")
                valid_steps = 0
            scenario_entries.append(
                {
                    "final": final,
                    "mean": mean,
                    "max": max_value,
                    "valid_steps": valid_steps,
                }
            )
        per_entry.append(scenario_entries)
    return {"per_scenario_sample": per_entry}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def _save_population_xy_score_image(
    *,
    sim_state,
    timestep: int,
    current_world_bkt5: np.ndarray,
    current_scores_bk: np.ndarray,
    output_path: Path,
    title: str,
    front_x: float,
    back_x: float,
    front_y: float,
    back_y: float,
    current_lanes: list[np.ndarray] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    world_bkt5 = np.asarray(current_world_bkt5, dtype=np.float32)
    scores_bk = np.asarray(current_scores_bk, dtype=np.float32)
    if world_bkt5.ndim != 4:
        raise ValueError(f"current_world_bkt5 must be rank-4 [B, K, T, 5], got {world_bkt5.shape}")
    if scores_bk.ndim != 2:
        raise ValueError(f"current_scores_bk must be rank-2 [B, K], got {scores_bk.shape}")

    batch_size = int(world_bkt5.shape[0])
    if int(scores_bk.shape[0]) != batch_size or int(scores_bk.shape[1]) != int(world_bkt5.shape[1]):
        raise ValueError(
            "current_world_bkt5 and current_scores_bk shapes are inconsistent: "
            f"{world_bkt5.shape} vs {scores_bk.shape}"
        )

    ncols = batch_size
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 6.0), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for b in range(batch_size):
        ax = axes[0, b]
        state_b = viz_module._index_pytree(sim_state, b)
        t = int(np.clip(timestep, 0, int(np.asarray(state_b.log_trajectory.x).shape[-1]) - 1))

        viz_module.plot_roadgraph_points(ax, state_b.roadgraph_points, verbose=False)

        if current_lanes is not None and b < len(current_lanes):
            lane_points = np.asarray(current_lanes[b], dtype=np.float32)
            if lane_points.ndim == 2 and lane_points.shape[0] >= 2 and lane_points.shape[1] >= 2:
                ax.plot(
                    lane_points[:, 0],
                    lane_points[:, 1],
                    color="deepskyblue",
                    linewidth=3.0,
                    alpha=0.95,
                    zorder=5,
                )

        is_ego = np.asarray(state_b.object_metadata.is_sdc).astype(bool)
        num_obj = int(state_b.log_trajectory.num_objects)
        is_controlled = np.zeros((num_obj,), dtype=bool)
        viz_module.plot_trajectory(
            ax,
            state_b.log_trajectory,
            is_controlled=is_controlled,
            time_idx=t,
            indices=None,
            past_traj_length=0,
            is_ego=is_ego,
        )

        traj_kt2 = world_bkt5[b, :, :, :2]
        score_k = scores_bk[b]
        for k in range(int(traj_kt2.shape[0])):
            xy_t2 = traj_kt2[k]
            ax.plot(
                xy_t2[:, 0],
                xy_t2[:, 1],
                color=cmap(float(score_k[k])),
                linewidth=1.2,
                alpha=0.95,
            )
            ax.scatter(
                xy_t2[0, 0],
                xy_t2[0, 1],
                color=cmap(float(score_k[k])),
                s=8,
                alpha=0.9,
            )

        current_xy = np.asarray(state_b.log_trajectory.xy[:, t, :])
        if np.any(is_ego):
            center_xy = current_xy[is_ego][0]
        else:
            center_xy = np.nanmean(current_xy, axis=0)
        ax.axis(
            (
                float(center_xy[0]) - float(back_x),
                float(center_xy[0]) + float(front_x),
                float(center_xy[1]) - float(back_y),
                float(center_xy[1]) + float(front_y),
            )
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"batch={b}, t={t}, K={traj_kt2.shape[0]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="score")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)