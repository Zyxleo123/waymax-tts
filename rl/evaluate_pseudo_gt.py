#!/usr/bin/env python3
"""Evaluate pseudo-GT rollouts under V-Max and failure-sample success criteria.

**V-Max accuracy** (training eval standard): per episode 1.0 iff none of
``termination_keys`` fire (default: offroad, overlap, run_red_light). Metrics
are computed on the **ScenarioMax** scene with the pseudo-GT trajectory spliced
into the ego log from ``start_timestep``.

**Failure success** (VLA failure-sample standard): ``goal_reached`` (ever within
2 m of ``goal_xy`` after ``start_timestep``) AND no overlap/offroad before goal
AND no traffic-light violation before goal. Evaluated on the **WOMD failure**
tfexample scene (same map as the original failure JSON).

Example::

    python -m rl.evaluate_pseudo_gt \\
        --pseudo-gt-dir /zfsauton/scratch/yixiz/pseudo_gt/repro_sac_v2 \\
        --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2 \\
        --failure-dir /zfsauton/scratch/mineuih/waymax_rs/failure_samples
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lane_graph.lane_graph_utils import LaneGraphLoader  # noqa: E402
from rl.womd_compare import (  # noqa: E402
    LOCAL_NUM_PATHS,
    LOCAL_NUM_POINTS_PER_PATH,
    SCENARIOMAX_NUM_PATHS,
    SCENARIOMAX_NUM_POINTS_PER_PATH,
    load_waymax_state,
)
from simulation.evaluation_utils import (  # noqa: E402
    check_goal_reaching,
    check_traffic_light_violation,
    rollout_predicted_trajectories_with_metrics,
)
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from vmax.simulator.metrics.collector import check_episode_success  # noqa: E402
from vmax.simulator.wrappers.interfaces.brax import BraxWrapper  # noqa: E402
from viz import viz as viz_module  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pseudo-gt-dir", required=True)
    p.add_argument("--run-dir", required=True, help="V-Max run dir for termination_keys.")
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument(
        "--lane-graph-dir",
        default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs",
    )
    p.add_argument("--goal-threshold-m", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", default=None, help="Per-scenario CSV (default: pseudo_gt_dir/eval_summary.csv).")
    return p.parse_args()


def _load_termination_keys(run_dir: str) -> tuple[str, ...]:
    cfg = yaml.safe_load(open(Path(run_dir) / ".hydra/config.yaml"))
    return tuple(cfg["termination_keys"])


def _find_brax_wrapper(env) -> BraxWrapper:
    cur = env
    while cur is not None:
        if isinstance(cur, BraxWrapper):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("BraxWrapper not found in env chain.")


def _splice_trajectory(state, traj_bt5: np.ndarray, start_t: int):
    """Splice policy trajectory into log; mirror log → sim for metric replay.

    Waymax ``OffroadMetric`` / ``OverlapMetric`` read ``sim_trajectory``, which is
    zero-initialized except t=0 on load. Replay eval must copy the spliced log.
    """
    segment = np.asarray(traj_bt5[start_t:], dtype=np.float32)
    spliced = apply_ego_replacements_to_expanded_state(
        state,
        start_t_b=int(start_t),
        traj_bt5=segment[None, ...],
    )
    return dataclasses.replace(spliced, sim_trajectory=spliced.log_trajectory)


def _vmax_accuracy(
    env,
    state,
    *,
    start_t: int,
    termination_keys: tuple[str, ...],
) -> dict[str, float]:
    wrapper = _find_brax_wrapper(env)
    state = viz_module._index_pytree(state, 0)
    batch_metrics: dict[str, float] = {k: 0.0 for k in termination_keys}
    horizon = int(np.asarray(state.log_trajectory.x).shape[-1])
    for t in range(int(start_t), horizon):
        state_t = dataclasses.replace(state, timestep=np.int32(t))
        metrics = wrapper.metrics(state_t)
        for key in termination_keys:
            if key in metrics:
                batch_metrics[key] = max(batch_metrics[key], float(metrics[key]))
    accuracy = check_episode_success(batch_metrics, termination_keys)
    return {"vmax_accuracy": accuracy, **{f"vmax_{k}": batch_metrics[k] for k in termination_keys}}


def _failure_success(
    state,
    *,
    goal_xy: np.ndarray,
    start_t: int,
    goal_threshold_m: float,
    lane_graphs,
) -> dict[str, float | bool | int]:
    goal_b2 = jnp.asarray(goal_xy, dtype=jnp.float32)[None, :]
    rollout = rollout_predicted_trajectories_with_metrics(state, metric_names=("overlap", "offroad"))
    tl_eval = check_traffic_light_violation(state, lane_graphs)
    goal_eval = check_goal_reaching(
        state, goal_b2, start_timestep=int(start_t), goal_threshold_m=goal_threshold_m
    )

    goal_reached = bool(goal_eval["reached"][0])
    goal_step = int(goal_eval["reached_step"][0])
    overlap_ts = np.asarray(rollout["metric_timeseries"]["overlap"][start_t:, 0])
    offroad_ts = np.asarray(rollout["metric_timeseries"]["offroad"][start_t:, 0])
    ep_len = (goal_step + 1 - start_t) if goal_reached else overlap_ts.shape[0]
    overlap = bool(overlap_ts[:ep_len].sum() > 0)
    offroad = bool(offroad_ts[:ep_len].sum() > 0)
    tl_violation = bool(tl_eval["violation"][0])
    tl_step = int(tl_eval["violation_step"][0])
    if goal_reached:
        tl_violation = tl_violation and (tl_step < goal_step)

    success = goal_reached and (not overlap) and (not offroad) and (not tl_violation)
    return {
        "failure_goal_reached": goal_reached,
        "failure_goal_step": goal_step,
        "failure_min_goal_distance_m": float(goal_eval["min_goal_distance_m"][0]),
        "failure_final_goal_distance_m": float(goal_eval["final_goal_distance_m"][0]),
        "failure_overlap": overlap,
        "failure_offroad": offroad,
        "failure_tl_violation": tl_violation,
        "failure_success": success,
    }


def _meta_path_for_npz(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.name.replace(".trajectory.npz", ".pseudo_gt.json"))


def _load_meta(npz_path: Path, failure_dir: Path) -> dict:
    meta_path = _meta_path_for_npz(npz_path)
    if meta_path.is_file():
        return json.load(open(meta_path))
    failure_json = failure_dir / npz_path.name.replace(".trajectory.npz", ".json")
    if not failure_json.is_file():
        raise FileNotFoundError(f"No metadata for {npz_path}")
    failure = json.load(open(failure_json))
    return {
        "failure_json": str(failure_json),
        "scenariomax_tfrecord": None,
        "scenariomax_record_index": None,
        "scenario_idx": failure["scenario_idx"],
        "tfrecord": failure["tfrecord"],
    }


def main() -> int:
    args = _parse_args()
    pseudo_dir = Path(args.pseudo_gt_dir)
    npz_paths = sorted(pseudo_dir.glob("*.trajectory.npz"))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise RuntimeError(f"No *.trajectory.npz under {pseudo_dir}")

    termination_keys = _load_termination_keys(args.run_dir)
    from waymax import dynamics
    from vmax.simulator import make_env_for_evaluation

    eval_config = yaml.safe_load(open(Path(args.run_dir) / ".hydra/config.yaml"))
    env = make_env_for_evaluation(
        max_num_objects=int(eval_config["max_num_objects"]),
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=eval_config["observation_type"],
        observation_config=eval_config["observation_config"],
        termination_keys=list(termination_keys),
        noisy_init=False,
    )
    lane_loader = LaneGraphLoader(args.lane_graph_dir)
    max_num_objects = int(eval_config["max_num_objects"])

    rows: list[dict] = []
    for npz_path in npz_paths:
        npz = np.load(npz_path)
        traj = np.asarray(npz["predicted_trajectory"], dtype=np.float32)
        start_t = int(npz["start_timestep"])
        goal_xy = np.asarray(npz["goal_xy"], dtype=np.float32)
        meta = _load_meta(npz_path, Path(args.failure_dir))
        failure_json = meta.get("failure_json") or str(
            Path(args.failure_dir) / npz_path.name.replace(".trajectory.npz", ".json")
        )
        orig = json.load(open(failure_json))

        smx_tf = meta.get("scenariomax_tfrecord") or meta.get("scenariomax_tfrecord_path")
        smx_idx = meta.get("scenariomax_record_index")
        if smx_tf is None or smx_idx is None:
            raise RuntimeError(f"Missing ScenarioMax mapping in {_meta_path_for_npz(npz_path)}")

        smx_state = load_waymax_state(
            str(smx_tf),
            int(smx_idx),
            num_paths=SCENARIOMAX_NUM_PATHS,
            num_points=SCENARIOMAX_NUM_POINTS_PER_PATH,
            max_num_objects=max_num_objects,
        )
        smx_spliced = _splice_trajectory(smx_state, traj, start_t)
        vmax = _vmax_accuracy(env, smx_spliced, start_t=start_t, termination_keys=termination_keys)

        womd_state = load_waymax_state(
            str(orig["tfrecord"]),
            int(orig["scenario_idx"]),
            num_paths=LOCAL_NUM_PATHS,
            num_points=LOCAL_NUM_POINTS_PER_PATH,
            max_num_objects=128,
        )
        womd_spliced = _splice_trajectory(womd_state, traj, start_t)
        lane_graphs = lane_loader.get_lane_graph_for_scenarios(
            str(orig["tfrecord"]), [int(orig["scenario_idx"])]
        )
        failure = _failure_success(
            womd_spliced,
            goal_xy=goal_xy,
            start_t=start_t,
            goal_threshold_m=args.goal_threshold_m,
            lane_graphs=lane_graphs,
        )

        row = {
            "stem": npz_path.stem,
            "orig_success": bool(orig.get("success", False)),
            "orig_goal_reached": bool(orig.get("goal_reached", False)),
            "orig_overlap": bool(orig.get("overlap", False)),
            "orig_offroad": bool(orig.get("offroad", False)),
            **vmax,
            **failure,
        }
        rows.append(row)
        print(
            f"{npz_path.stem}: vmax_acc={row['vmax_accuracy']:.0f}  "
            f"failure_success={row['failure_success']}  "
            f"(orig_success={row['orig_success']})"
        )

    n = len(rows)
    summary = {
        "n": n,
        "vmax_accuracy_rate": float(np.mean([r["vmax_accuracy"] for r in rows])),
        "failure_success_rate": float(np.mean([r["failure_success"] for r in rows])),
        "failure_goal_reached_rate": float(np.mean([r["failure_goal_reached"] for r in rows])),
        "orig_success_rate": float(np.mean([r["orig_success"] for r in rows])),
        "orig_failure_rate": float(np.mean([not r["orig_success"] for r in rows])),
        "still_failure_rate": float(np.mean([not r["failure_success"] for r in rows])),
    }

    out_csv = Path(args.out_csv) if args.out_csv else pseudo_dir / "eval_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary_path = pseudo_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"  scenarios                     : {summary['n']}")
    print(f"  V-Max accuracy rate           : {summary['vmax_accuracy_rate']:.3f}")
    print(f"  failure success rate (pseudo) : {summary['failure_success_rate']:.3f}")
    print(f"  failure goal reached rate     : {summary['failure_goal_reached_rate']:.3f}")
    print(f"  original VLA success rate     : {summary['orig_success_rate']:.3f}")
    print(f"  still failure by VLA standard : {summary['still_failure_rate']:.3f}")
    print(f"  wrote {out_csv}")
    print(f"  wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
