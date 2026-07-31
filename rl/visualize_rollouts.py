"""Visualize RL policy rollouts saved by ``eval_ppo`` / ``eval_sac``.

Each per-scenario JSON contains the policy trajectory (``sim_*``), the expert log
(``log_*``), the goal, and episode outcome flags. This script renders them as
static overhead plots and/or MP4 videos with the map and other agents.

Examples
--------
Plot every rollout in an eval directory:

    python -m rl.visualize_rollouts --rollout-dir runs/ppo_failures/eval_rollouts

Render videos for failed rollouts only:

    python -m rl.visualize_rollouts --rollout-dir runs/sac_failures/eval_rollouts \\
        --mode video --filter failure

Visualize one file:

    python -m rl.visualize_rollouts \\
        --rollout runs/ppo_failures/eval_rollouts/shard.scenario_00042.json
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from waymax import config as waymax_config

from viz import viz as viz_module
from viz.render import _load_scenario_state_fast, render_videos_batched


def discover_rollouts(path: Path) -> list[Path]:
    """Return per-scenario rollout JSON paths under ``path``."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Rollout path not found: {path}")

    files = sorted(
        p for p in path.glob("*.json")
        if p.name != "summary.json"
    )
    if not files:
        raise FileNotFoundError(f"No rollout JSON files found in {path}")
    return files


def load_rollout(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_xy(record: dict[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(record[f"{prefix}_x"], dtype=np.float64)
    y = np.asarray(record[f"{prefix}_y"], dtype=np.float64)
    return x, y


def build_ego_traj_l5(record: dict[str, Any]) -> np.ndarray:
    """Build ``[L, 5]`` ego trajectory for the video renderer."""
    x, y = _as_xy(record, "sim")
    yaw = np.asarray(record["sim_yaw"], dtype=np.float64)
    vel_x = np.asarray(record.get("sim_vel_x", np.zeros_like(x)), dtype=np.float64)
    vel_y = np.asarray(record.get("sim_vel_y", np.zeros_like(x)), dtype=np.float64)
    if not (len(x) == len(y) == len(yaw) == len(vel_x) == len(vel_y)):
        raise ValueError("sim trajectory fields have mismatched lengths.")
    return np.stack([x, y, yaw, vel_x, vel_y], axis=-1).astype(np.float32)


def _outcome_label(record: dict[str, Any]) -> str:
    parts = []
    if record.get("clean_success"):
        parts.append("CLEAN")
    elif record.get("reached") or record.get("success"):
        parts.append("reached")
    else:
        parts.append("missed")
    if record.get("collision"):
        parts.append("collision")
    if record.get("offroad"):
        parts.append("offroad")
    return " | ".join(parts)


def _passes_filter(record: dict[str, Any], filt: str) -> bool:
    if filt == "all":
        return True
    if filt == "clean":
        return bool(record.get("clean_success"))
    if filt == "success":
        return bool(record.get("reached") or record.get("success"))
    if filt == "failure":
        return not bool(record.get("clean_success"))
    if filt == "collision":
        return bool(record.get("collision"))
    if filt == "offroad":
        return bool(record.get("offroad"))
    raise ValueError(f"Unknown filter: {filt}")


def _load_scenario(record: dict[str, Any], max_num_objects: int | None):
    tfrecord = record.get("tfrecord")
    scenario_idx = record.get("scenario_idx")
    if not tfrecord or scenario_idx is None:
        return None
    cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfrecord),
        max_num_objects=max_num_objects,
        batch_dims=(1,),
        shuffle_seed=0,
    )
    return _load_scenario_state_fast(cfg, int(scenario_idx))


def _plot_bounds(
    *,
    sim_xy: np.ndarray,
    log_xy: np.ndarray,
    goal_xy: np.ndarray,
    margin: float,
) -> tuple[float, float, float, float]:
    pts = [sim_xy, log_xy, goal_xy[None, :]]
    stacked = np.concatenate([p for p in pts if p.size], axis=0)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def plot_static_rollout(
    record: dict[str, Any],
    *,
    out_path: Path,
    max_num_objects: int | None = None,
    margin_m: float = 15.0,
    show_map: bool = True,
    show_agents: bool = True,
) -> Path:
    """Save a static overhead plot comparing policy vs expert trajectories."""
    sim_x, sim_y = _as_xy(record, "sim")
    log_x, log_y = _as_xy(record, "log")
    log_valid = np.asarray(record.get("log_valid", np.ones_like(log_x, dtype=bool)), dtype=bool)
    goal = np.asarray(record.get("goal_xy", [np.nan, np.nan]), dtype=np.float64)

    sim_xy = np.stack([sim_x, sim_y], axis=-1)
    log_xy = np.stack([log_x[log_valid], log_y[log_valid]], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))

    state = _load_scenario(record, max_num_objects) if show_map else None
    if state is not None:
        viz_module.plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
        if show_agents:
            is_ego = np.asarray(state.object_metadata.is_sdc).astype(bool)
            is_controlled = np.zeros((state.log_trajectory.num_objects,), dtype=bool)
            viz_module.plot_trajectory(
                ax,
                state.log_trajectory,
                is_controlled=is_controlled,
                time_idx=0,
                indices=None,
                past_traj_length=0,
                is_ego=is_ego,
                is_adv=np.zeros_like(is_ego),
            )

    ax.plot(log_xy[:, 0], log_xy[:, 1], color="#4C72B0", linewidth=2.0, alpha=0.85, label="expert log")
    ax.plot(sim_xy[:, 0], sim_xy[:, 1], color="#C44E52", linewidth=2.5, alpha=0.95, label="policy rollout")
    ax.scatter(sim_xy[0, 0], sim_xy[0, 1], color="#C44E52", s=36, zorder=8, label="policy start")
    ax.scatter(sim_xy[-1, 0], sim_xy[-1, 1], color="#C44E52", s=56, marker="x", zorder=8, label="policy end")

    if np.all(np.isfinite(goal)):
        ax.scatter(goal[0], goal[1], color="gold", edgecolors="black", s=120, zorder=9, label="goal")
        goal_circle = plt.Circle(
            (float(goal[0]), float(goal[1])),
            radius=3.0,
            fill=False,
            edgecolor="gold",
            linewidth=1.5,
            alpha=0.8,
            zorder=7,
        )
        ax.add_patch(goal_circle)

    xmin, xmax, ymin, ymax = _plot_bounds(
        sim_xy=sim_xy,
        log_xy=log_xy,
        goal_xy=goal,
        margin=margin_m,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    scenario_idx = record.get("scenario_idx", "?")
    title = f"scenario {scenario_idx} — {_outcome_label(record)}"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_rollout_video(
    record: dict[str, Any],
    *,
    out_path: Path,
    max_num_objects: int = 128,
    num_frames: int = 91,
    fps: int = 10,
    width: int = 512,
    height: int = 512,
) -> Path:
    """Render an MP4 with the policy ego trajectory spliced into the scene."""
    tfrecord = record.get("tfrecord")
    scenario_idx = record.get("scenario_idx")
    if not tfrecord or scenario_idx is None:
        raise ValueError("Rollout JSON is missing tfrecord/scenario_idx; cannot render video.")

    ego_traj = build_ego_traj_l5(record)
    goal = np.asarray(record.get("goal_xy"), dtype=np.float32)
    if goal.shape != (2,):
        goal = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = render_videos_batched(
        tfrecord_scenarios=[(str(tfrecord), int(scenario_idx))],
        output_dir=str(out_path.parent),
        ego_start_times=[0],
        ego_trajectories=[ego_traj],
        goal_xy=goal,
        max_num_objects=max_num_objects,
        num_frames=min(num_frames, len(ego_traj)),
        fps=fps,
        width=width,
        height=height,
        use_log_traj=True,
        show_agent_id=False,
    )
    rendered = paths[0]
    if rendered.resolve() != out_path.resolve():
        out_path.unlink(missing_ok=True)
        rendered.replace(out_path)
    return out_path


def _default_out_path(rollout_path: Path, out_dir: Path | None, suffix: str) -> Path:
    stem = rollout_path.stem
    if out_dir is None:
        return rollout_path.with_suffix(suffix)
    return out_dir / f"{stem}{suffix}"


def visualize_rollouts(
    rollout_paths: Iterable[Path],
    *,
    mode: str,
    out_dir: Path | None,
    filt: str,
    limit: int | None,
    max_num_objects: int | None,
    margin_m: float,
    no_map: bool,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
) -> list[Path]:
    outputs: list[Path] = []
    count = 0
    for rollout_path in rollout_paths:
        record = load_rollout(rollout_path)
        if not _passes_filter(record, filt):
            continue
        if limit is not None and count >= limit:
            break

        if mode in ("static", "both"):
            png_path = _default_out_path(rollout_path, out_dir, ".png")
            plot_static_rollout(
                record,
                out_path=png_path,
                max_num_objects=max_num_objects,
                margin_m=margin_m,
                show_map=not no_map,
            )
            outputs.append(png_path)
            print(f"[static] {png_path}")

        if mode in ("video", "both"):
            mp4_path = _default_out_path(rollout_path, out_dir, ".mp4")
            render_rollout_video(
                record,
                out_path=mp4_path,
                max_num_objects=max_num_objects or 128,
                num_frames=num_frames,
                fps=fps,
                width=width,
                height=height,
            )
            outputs.append(mp4_path)
            print(f"[video]  {mp4_path}")

        count += 1
    return outputs


def _parse_args():
    p = argparse.ArgumentParser(description="Visualize RL eval rollouts (eval_ppo / eval_sac JSON).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollout-dir", type=str, help="Directory of per-scenario rollout JSON files.")
    src.add_argument("--rollout", type=str, help="Single rollout JSON file.")

    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Default: alongside each rollout JSON.")
    p.add_argument("--mode", type=str, default="static", choices=["static", "video", "both"],
                   help="Render static PNG plots, MP4 videos, or both.")
    p.add_argument("--filter", type=str, default="all",
                   choices=["all", "clean", "success", "failure", "collision", "offroad"],
                   help="Only visualize rollouts matching this outcome.")
    p.add_argument("--limit", type=int, default=None, help="Cap number of rollouts processed.")
    p.add_argument("--max-num-objects", type=int, default=None,
                   help="Truncate scene objects when loading the map (default: keep all).")
    p.add_argument("--margin-m", type=float, default=15.0,
                   help="Plot margin around trajectories for static mode.")
    p.add_argument("--no-map", action="store_true",
                   help="Skip roadgraph/agents; plot trajectories only.")
    p.add_argument("--num-frames", type=int, default=91, help="Max frames for video mode.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    return p.parse_args()


def main():
    args = _parse_args()
    rollout_root = Path(args.rollout) if args.rollout else Path(args.rollout_dir)
    rollout_paths = discover_rollouts(rollout_root)
    out_dir = Path(args.out_dir) if args.out_dir else None

    outputs = visualize_rollouts(
        rollout_paths,
        mode=args.mode,
        out_dir=out_dir,
        filt=args.filter,
        limit=args.limit,
        max_num_objects=args.max_num_objects,
        margin_m=args.margin_m,
        no_map=args.no_map,
        num_frames=args.num_frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    if not outputs:
        print("No rollouts matched the filter; nothing was written.")
    else:
        print(f"\nWrote {len(outputs)} file(s).")


if __name__ == "__main__":
    main()
