#!/usr/bin/env python3
"""Visualize a Waymax scenario with an optional ego trajectory overlay.

Supports two dataset layouts:

* **WOMD tfexample** — the format used by failure-case JSONs / ``*.trajectory.npz``
  (official WOMD 1.3.1 tf_example shards: 45×800 SDC paths, up to 128 objects).
* **ScenarioMax tfexample** — V-Max training data (10×300 paths, 64 objects).

Inputs (pick one source group):

* ``--failure-json`` — failure-sample result JSON (+ companion ``.trajectory.npz``).
* ``--trajectory-npz`` with ``--tfrecord`` + ``--scenario-idx``.
* ``--failure-json`` + ``--trace-json`` + ``--scenariomax`` — render on the mapped
  ScenarioMax record (for pseudo-GT rollouts).

Examples::

    # Failure case (WOMD tfexample) + logged/predicted trajectory
    python -m rl.visualize_scenario \\
        --failure-json /zfsauton/scratch/mineuih/waymax_rs/failure_samples/\\
            training_tfexample.tfrecord-00000-of-01000.scenario_052.json \\
        --mode both

    # Pseudo-GT on ScenarioMax scene (from trace mapping)
    python -m rl.visualize_scenario \\
        --failure-json .../scenario_052.json \\
        --trace-json /zfsauton/scratch/yixiz/failure_womd_trace.json \\
        --trajectory-npz /zfsauton/scratch/yixiz/pseudo_gt/repro_sac_v2/\\
            training_tfexample.tfrecord-00000-of-01000.scenario_052.trajectory.npz \\
        --scenariomax --mode video

    # Explicit ScenarioMax tfrecord + index
    python -m rl.visualize_scenario \\
        --tfrecord /zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord \\
        --scenario-idx 12345 --scenariomax --mode static
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

# ScenarioMax / V-Max training layout uses 64 objects.
SCENARIOMAX_MAX_NUM_OBJECTS = 64

from rl.tfrecord_fast import default_offset_index_path, offset_index_exists
from viz import viz as viz_module
from viz.render import _load_scenario_state_fast, build_dataset_config, render_videos_batched


@dataclasses.dataclass
class ScenarioViz:
    """Everything needed to render one scenario + trajectory."""

    tfrecord: str
    scenario_idx: int
    scenariomax: bool
    trajectory_bt5: np.ndarray | None = None
    start_timestep: int = 0
    goal_xy: np.ndarray | None = None
    ego_idx: int | None = None
    title: str = ""


def _load_trajectory_npz(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _traj_bt5_from_npz(npz: dict[str, Any]) -> np.ndarray:
    traj = np.asarray(npz["predicted_trajectory"], dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] != 5:
        raise ValueError(f"predicted_trajectory must be [T, 5], got {traj.shape}")
    return traj


def _sdc_log_bt5(state, ego_idx: int | None = None) -> np.ndarray:
    if ego_idx is None:
        is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
        ego_idx = int(np.argmax(is_sdc))
    traj = state.log_trajectory
    x = np.asarray(traj.x[ego_idx])
    y = np.asarray(traj.y[ego_idx])
    yaw = np.asarray(traj.yaw[ego_idx])
    vel_x = np.asarray(traj.vel_x[ego_idx])
    vel_y = np.asarray(traj.vel_y[ego_idx])
    valid = np.asarray(traj.valid[ego_idx]).astype(bool)
    out = np.stack([x, y, yaw, vel_x, vel_y], axis=-1).astype(np.float32)
    out[~valid] = np.nan
    return out


def _lookup_trace_record(trace_json: str, failure_json: str) -> dict[str, Any] | None:
    trace = json.load(open(trace_json))
    failure_json = str(Path(failure_json).resolve())
    for rec in trace.get("records", []):
        if str(Path(rec.get("json_path", "")).resolve()) == failure_json:
            return rec
    return None


def resolve_scenario_viz(
    *,
    failure_json: str | None = None,
    trajectory_npz: str | None = None,
    tfrecord: str | None = None,
    scenario_idx: int | None = None,
    trace_json: str | None = None,
    scenariomax: bool = False,
    dataset: str = "auto",
) -> ScenarioViz:
    """Resolve tfrecord, index, trajectory, and dataset layout from CLI inputs."""
    meta: dict[str, Any] = {}
    traj_npz_data: dict[str, Any] | None = None

    if failure_json:
        failure_path = Path(failure_json)
        meta = json.load(open(failure_path))
        stem = failure_path.stem
        npz_path = Path(trajectory_npz) if trajectory_npz else failure_path.with_suffix(".trajectory.npz")
        if npz_path.is_file():
            traj_npz_data = _load_trajectory_npz(npz_path)

    trace_rec = None
    if trace_json and failure_json:
        trace_rec = _lookup_trace_record(trace_json, failure_json)

    if dataset == "auto":
        use_smx = scenariomax
        if not use_smx and tfrecord and "ScenarioMax" in tfrecord:
            use_smx = True
        if trace_rec and scenariomax:
            use_smx = True
    elif dataset == "scenariomax":
        use_smx = True
    elif dataset == "womd":
        use_smx = False
    else:
        raise ValueError(f"Unknown dataset mode: {dataset}")

    if use_smx and trace_rec and trace_rec.get("scenariomax_hit"):
        rec_tf = trace_rec["scenariomax_tfrecord_path"]
        rec_idx = int(trace_rec["scenariomax_record_index"])
    elif tfrecord is not None and scenario_idx is not None:
        rec_tf = str(tfrecord)
        rec_idx = int(scenario_idx)
    elif meta:
        rec_tf = str(meta["tfrecord"])
        rec_idx = int(meta["scenario_idx"])
    else:
        raise ValueError("Provide --failure-json or (--tfrecord and --scenario-idx).")

    if dataset == "auto" and not use_smx and "ScenarioMax" in rec_tf:
        use_smx = True

    traj = _traj_bt5_from_npz(traj_npz_data) if traj_npz_data else None
    start_t = int(traj_npz_data["start_timestep"]) if traj_npz_data else 0
    ego_idx = int(traj_npz_data["ego_idx"]) if traj_npz_data and "ego_idx" in traj_npz_data else None
    goal_src = meta.get("goal_xy")
    if goal_src is None and traj_npz_data is not None:
        goal_src = traj_npz_data.get("goal_xy")
    goal = np.asarray(goal_src, dtype=np.float64) if goal_src is not None else None
    if goal is None or goal.shape != (2,) or not np.all(np.isfinite(goal)):
        goal = None

    title = f"scenario {rec_idx}"
    if failure_json:
        title = f"{Path(failure_json).stem} ({'ScenarioMax' if use_smx else 'WOMD'})"

    return ScenarioViz(
        tfrecord=rec_tf,
        scenario_idx=rec_idx,
        scenariomax=use_smx,
        trajectory_bt5=traj,
        start_timestep=start_t,
        goal_xy=goal,
        ego_idx=ego_idx,
        title=title,
    )


def plot_static(
    spec: ScenarioViz,
    *,
    out_path: Path,
    max_num_objects: int = 128,
    margin_m: float = 15.0,
    show_map: bool = True,
) -> Path:
    """Overhead PNG: map, expert log, optional trajectory overlay, goal."""
    log_bt5 = None
    log_xy = np.empty((0, 2), dtype=np.float32)
    state = None

    if show_map:
        cfg = build_dataset_config(
            spec.tfrecord,
            batch_dims=(1,),
            max_num_objects=max_num_objects,
            scenariomax=spec.scenariomax,
        )
        state = _load_scenario_state_fast(cfg, spec.scenario_idx)
        state = viz_module._index_pytree(state, 0)
        log_bt5 = _sdc_log_bt5(state, spec.ego_idx)
        log_xy = log_bt5[:, :2]
        log_valid = np.isfinite(log_xy).all(axis=1)
        log_xy = log_xy[log_valid]
    elif spec.trajectory_bt5 is not None:
        log_xy = spec.trajectory_bt5[:, :2]

    fig, ax = plt.subplots(figsize=(8, 8))
    if show_map and state is not None:
        viz_module.plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
        is_ego = np.asarray(state.object_metadata.is_sdc).astype(bool)
        viz_module.plot_trajectory(
            ax,
            state.log_trajectory,
            is_controlled=np.zeros((state.log_trajectory.num_objects,), dtype=bool),
            time_idx=0,
            is_ego=is_ego,
            is_adv=np.zeros_like(is_ego),
        )

    ax.plot(log_xy[:, 0], log_xy[:, 1], color="#4C72B0", linewidth=2.0, alpha=0.85, label="expert log" if show_map else "reference")

    if spec.trajectory_bt5 is not None:
        pred_xy = spec.trajectory_bt5[:, :2]
        ax.plot(
            pred_xy[:, 0], pred_xy[:, 1],
            color="#C44E52", linewidth=2.5, alpha=0.95, label="trajectory overlay",
        )
        if spec.start_timestep > 0:
            ax.axvline(
                pred_xy[spec.start_timestep, 0],
                color="#C44E52", linestyle=":", alpha=0.4,
            )

    if spec.goal_xy is not None:
        ax.scatter(
            spec.goal_xy[0], spec.goal_xy[1],
            color="gold", edgecolors="black", s=120, zorder=9, label="goal",
        )
        ax.add_patch(plt.Circle(
            (float(spec.goal_xy[0]), float(spec.goal_xy[1])),
            radius=3.0, fill=False, edgecolor="gold", linewidth=1.5, zorder=8,
        ))

    pts = []
    if log_xy.size:
        pts.append(log_xy)
    if spec.trajectory_bt5 is not None:
        pts.append(spec.trajectory_bt5[:, :2])
    if spec.goal_xy is not None:
        pts.append(spec.goal_xy[None, :])
    stacked = np.concatenate(pts, axis=0)
    xmin, ymin = stacked.min(axis=0) - margin_m
    xmax, ymax = stacked.max(axis=0) + margin_m
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(spec.title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_video(
    spec: ScenarioViz,
    *,
    out_path: Path,
    max_num_objects: int = 128,
    num_frames: int = 91,
    fps: int = 10,
    width: int = 512,
    height: int = 512,
) -> Path:
    """MP4 with ego trajectory spliced into the log-replayed scene."""
    ego_traj = spec.trajectory_bt5
    start_t = spec.start_timestep
    if ego_traj is None:
        cfg = build_dataset_config(
            spec.tfrecord, batch_dims=(1,), max_num_objects=max_num_objects,
            scenariomax=spec.scenariomax,
        )
        state = _load_scenario_state_fast(cfg, spec.scenario_idx)
        ego_traj = _sdc_log_bt5(state, spec.ego_idx)
        start_t = 0

    segment = np.asarray(ego_traj[start_t:], dtype=np.float32)
    goal = np.asarray(spec.goal_xy, dtype=np.float32) if spec.goal_xy is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = render_videos_batched(
        tfrecord_scenarios=[(spec.tfrecord, spec.scenario_idx)],
        output_dir=str(out_path.parent),
        ego_start_times=[start_t],
        ego_trajectories=[segment],
        goal_xy=goal,
        max_num_objects=max_num_objects,
        num_frames=min(num_frames, len(ego_traj)),
        fps=fps,
        width=width,
        height=height,
        scenariomax=spec.scenariomax,
        show_agent_id=False,
    )
    rendered = paths[0]
    if rendered.resolve() != out_path.resolve():
        out_path.unlink(missing_ok=True)
        rendered.replace(out_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failure-json", type=str, default=None,
                   help="Failure-sample result JSON (WOMD tfexample pointer).")
    p.add_argument("--trajectory-npz", type=str, default=None,
                   help="Trajectory file [T,5] = x,y,yaw,vel_x,vel_y (default: sibling of failure JSON).")
    p.add_argument("--tfrecord", type=str, default=None)
    p.add_argument("--scenario-idx", type=int, default=None)
    p.add_argument("--trace-json", type=str, default=None,
                   help="failure_womd_trace.json — required with --scenariomax + --failure-json.")
    p.add_argument("--scenariomax", action="store_true",
                   help="Render on ScenarioMax tfrecord (use with --trace-json).")
    p.add_argument("--dataset", type=str, default="auto", choices=["auto", "womd", "scenariomax"],
                   help="Tfrecord layout (default: auto-detect).")
    p.add_argument("--out", type=str, default=None, help="Output .png or .mp4 path.")
    p.add_argument("--mode", type=str, default="static", choices=["static", "video", "both"])
    p.add_argument("--max-num-objects", type=int, default=128)
    p.add_argument("--margin-m", type=float, default=15.0)
    p.add_argument("--no-map", action="store_true")
    p.add_argument("--num-frames", type=int, default=91)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    return p.parse_args()


def _default_out(args: argparse.Namespace, suffix: str) -> Path:
    if args.out:
        base = Path(args.out)
        if base.suffix in (".png", ".mp4"):
            return base
        return base.with_suffix(suffix)
    if args.failure_json:
        stem = Path(args.failure_json).stem
    elif args.scenario_idx is not None:
        stem = f"scenario_{args.scenario_idx:05d}"
    else:
        stem = "scenario_viz"
    # Default to cwd — failure_samples often has read-only media from other users.
    return Path.cwd() / f"{stem}{suffix}"


def _writable_out(path: Path) -> Path:
    """Use ``path`` when writable; otherwise fall back to cwd with the same basename."""
    if path.exists():
        if os.access(path, os.W_OK):
            return path
        alt = Path.cwd() / path.name
        print(f"Warning: cannot overwrite {path}; writing to {alt}")
        return alt
    parent = path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        alt = Path.cwd() / path.name
        print(f"Warning: cannot write to {parent}; writing to {alt}")
        return alt
    return path


def _effective_max_num_objects(args: argparse.Namespace, spec: ScenarioViz) -> int:
    if spec.scenariomax and args.max_num_objects == 128:
        return SCENARIOMAX_MAX_NUM_OBJECTS
    return int(args.max_num_objects)


def _check_scenariomax_load(spec: ScenarioViz, *, show_map: bool) -> None:
    if not spec.scenariomax or not show_map:
        return
    if spec.scenario_idx < 1000 or offset_index_exists(spec.tfrecord):
        return
    index_path = default_offset_index_path(spec.tfrecord)
    raise RuntimeError(
        f"ScenarioMax record {spec.scenario_idx} is in a large TFRecord and needs a "
        f"byte-offset index before map rendering.\n"
        f"Build once (~1h for the monolithic file):\n"
        f"  python -m rl.tfrecord_fast build {spec.tfrecord}\n"
        f"Quick trajectory-only preview (no map):\n"
        f"  python -m rl.visualize_scenario ... --scenariomax --no-map --mode static"
    )


def main() -> int:
    args = _parse_args()
    spec = resolve_scenario_viz(
        failure_json=args.failure_json,
        trajectory_npz=args.trajectory_npz,
        tfrecord=args.tfrecord,
        scenario_idx=args.scenario_idx,
        trace_json=args.trace_json,
        scenariomax=args.scenariomax,
        dataset=args.dataset,
    )
    print(f"tfrecord      : {spec.tfrecord}")
    print(f"scenario_idx  : {spec.scenario_idx}")
    print(f"dataset       : {'scenariomax' if spec.scenariomax else 'womd'}")
    print(f"start_timestep: {spec.start_timestep}")
    if spec.trajectory_bt5 is not None:
        print(f"trajectory    : {spec.trajectory_bt5.shape}")

    max_num_objects = _effective_max_num_objects(args, spec)
    show_map = not args.no_map
    _check_scenariomax_load(spec, show_map=show_map)

    outputs: list[Path] = []
    if args.mode in ("static", "both"):
        png = _writable_out(_default_out(args, ".png"))
        plot_static(
            spec, out_path=png, max_num_objects=max_num_objects,
            margin_m=args.margin_m, show_map=show_map,
        )
        outputs.append(png)
        print(f"[static] {png}")

    if args.mode in ("video", "both"):
        mp4 = _writable_out(_default_out(args, ".mp4"))
        render_video(
            spec, out_path=mp4, max_num_objects=max_num_objects,
            num_frames=args.num_frames, fps=args.fps,
            width=args.width, height=args.height,
        )
        outputs.append(mp4)
        print(f"[video]  {mp4}")

    print(f"Wrote {len(outputs)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
