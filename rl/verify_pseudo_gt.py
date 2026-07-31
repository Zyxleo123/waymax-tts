#!/usr/bin/env python3
"""Verification helpers for pseudo-GT rollouts and eval.

Checks (all optional via ``--checks``):

* **indices** — byte-offset index for ScenarioMax monolith + SQLite id indexes.
* **mapping** — ``scenario/id`` from each ``.pseudo_gt.json`` matches the
  ScenarioMax SQLite index (no slow TFRecord scans).
* **metrics** — per rollout, compare broken vs fixed V-Max replay accuracy and
  ScenarioMax / WOMD rollout offroad flags (diagnoses the ``sim_trajectory``
  replay bug in older ``evaluate_pseudo_gt`` runs).

Example::

    python -m rl.verify_pseudo_gt \\
        --pseudo-gt-dir /zfsauton/scratch/yixiz/pseudo_gt/repro_sac_v2 \\
        --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2

    python -m rl.verify_pseudo_gt --checks indices mapping --limit 20
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl.tfrecord_fast import default_offset_index_path, offset_index_exists  # noqa: E402
from rl.womd_compare import (  # noqa: E402
    LOCAL_NUM_PATHS,
    LOCAL_NUM_POINTS_PER_PATH,
    SCENARIOMAX_NUM_PATHS,
    SCENARIOMAX_NUM_POINTS_PER_PATH,
    load_waymax_state,
)
from simulation.evaluation_utils import rollout_predicted_trajectories_with_metrics  # noqa: E402
from simulation.planning_utils import apply_ego_replacements_to_expanded_state  # noqa: E402
from vmax.simulator import make_env_for_evaluation  # noqa: E402
from vmax.simulator.metrics.collector import check_episode_success  # noqa: E402
from vmax.simulator.wrappers.interfaces.brax import BraxWrapper  # noqa: E402
from viz import viz as viz_module  # noqa: E402

DEFAULT_SCENARIOMAX_TFRECORD = "/zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord"
DEFAULT_SCENARIOMAX_INDEX = "/zfsauton/scratch/yixiz/scenariomax_scenario_id_index.sqlite3"
DEFAULT_WOMD_INDEX = "/zfsauton/scratch/yixiz/womd_scenario_id_index.sqlite3"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pseudo-gt-dir", required=True)
    p.add_argument("--run-dir", default=None, help="V-Max run dir (required for metrics check).")
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument(
        "--checks",
        nargs="+",
        default=["indices", "mapping", "metrics"],
        choices=["indices", "mapping", "metrics"],
    )
    p.add_argument("--scenariomax-tfrecord", default=DEFAULT_SCENARIOMAX_TFRECORD)
    p.add_argument("--scenariomax-index", default=DEFAULT_SCENARIOMAX_INDEX)
    p.add_argument("--womd-index", default=DEFAULT_WOMD_INDEX)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="Default: pseudo-gt-dir.")
    return p.parse_args()


def _sqlite_count(path: str, table: str) -> int:
    if not os.path.isfile(path):
        return -1
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return -1
    finally:
        conn.close()


def check_indices(
    *,
    scenariomax_tfrecord: str,
    scenariomax_index: str,
    womd_index: str,
) -> dict:
    offset_path = default_offset_index_path(scenariomax_tfrecord)
    out = {
        "scenariomax_tfrecord": scenariomax_tfrecord,
        "scenariomax_offset_index": offset_path,
        "scenariomax_offset_index_exists": offset_index_exists(scenariomax_tfrecord),
        "scenariomax_sqlite": scenariomax_index,
        "scenariomax_sqlite_rows": _sqlite_count(scenariomax_index, "scenariomax_index"),
        "womd_sqlite": womd_index,
        "womd_sqlite_rows": _sqlite_count(womd_index, "scenario_index"),
    }
    if out["scenariomax_offset_index_exists"]:
        out["scenariomax_offset_index_bytes"] = os.path.getsize(offset_path)
    missing = []
    if not out["scenariomax_offset_index_exists"]:
        missing.append("ScenarioMax .offsets.npy (python -m rl.tfrecord_fast build ...)")
    if out["scenariomax_sqlite_rows"] <= 0:
        missing.append("ScenarioMax scenario_id sqlite")
    if out["womd_sqlite_rows"] <= 0:
        missing.append("WOMD scenario_id sqlite")
    out["ok"] = not missing
    out["missing"] = missing
    return out


def _lookup_scenariomax(conn: sqlite3.Connection, scenario_id: str) -> tuple[int, str] | None:
    row = conn.execute(
        "SELECT record_index, tfrecord_path FROM scenariomax_index WHERE scenario_id = ?",
        (scenario_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def check_mapping(
    pseudo_dir: Path,
    *,
    scenariomax_index: str,
    limit: int | None,
) -> dict:
    conn = sqlite3.connect(scenariomax_index)
    meta_paths = sorted(pseudo_dir.glob("*.pseudo_gt.json"))
    if limit is not None:
        meta_paths = meta_paths[:limit]

    rows: list[dict] = []
    mismatches = 0
    missing_sql = 0
    for meta_path in meta_paths:
        meta = json.load(open(meta_path))
        sid = meta.get("tf2_scenario_id")
        stored_idx = meta.get("scenariomax_record_index")
        stored_tf = meta.get("scenariomax_tfrecord") or meta.get("scenariomax_tfrecord_path")
        hit = _lookup_scenariomax(conn, sid) if sid else None
        ok = False
        err = None
        if not sid:
            err = "missing tf2_scenario_id"
        elif hit is None:
            err = "scenario_id not in sqlite"
            missing_sql += 1
        else:
            idx_sql, tf_sql = hit
            ok = int(stored_idx) == idx_sql and (stored_tf is None or str(stored_tf) == tf_sql)
            if not ok:
                mismatches += 1
                err = f"stored idx={stored_idx} tf={stored_tf} vs sql idx={idx_sql} tf={tf_sql}"
        rows.append(
            {
                "stem": meta_path.name.replace(".pseudo_gt.json", ""),
                "tf2_scenario_id": sid,
                "mapping_ok": ok,
                "error": err,
            }
        )
    conn.close()
    n = len(rows)
    return {
        "n": n,
        "mapping_match_rate": float(np.mean([r["mapping_ok"] for r in rows])) if n else 0.0,
        "sqlite_misses": missing_sql,
        "index_mismatches": mismatches,
        "ok": mismatches == 0 and missing_sql == 0,
        "rows": rows,
    }


def _splice_log_only(state, traj_bt5: np.ndarray, start_t: int):
    """Old (broken) replay: splice log only; sim stays zero after t=0."""
    segment = np.asarray(traj_bt5[start_t:], dtype=np.float32)
    return apply_ego_replacements_to_expanded_state(
        state, start_t_b=int(start_t), traj_bt5=segment[None, ...]
    )


def _splice_log_and_sim(state, traj_bt5: np.ndarray, start_t: int):
    """Fixed replay: mirror spliced log into sim for Waymax metrics."""
    segment = np.asarray(traj_bt5[start_t:], dtype=np.float32)
    spliced = apply_ego_replacements_to_expanded_state(
        state, start_t_b=int(start_t), traj_bt5=segment[None, ...]
    )
    return dataclasses.replace(spliced, sim_trajectory=spliced.log_trajectory)


def _find_brax_wrapper(env) -> BraxWrapper:
    cur = env
    while cur is not None:
        if isinstance(cur, BraxWrapper):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("BraxWrapper not found in env chain.")


def _vmax_pointwise_accuracy(env, state, *, start_t: int, termination_keys: tuple[str, ...]) -> float:
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
    return float(check_episode_success(batch_metrics, termination_keys))


def _rollout_offroad_overlap(state, start_t: int) -> tuple[bool, bool]:
    rollout = rollout_predicted_trajectories_with_metrics(state, metric_names=("overlap", "offroad"))
    off = np.asarray(rollout["metric_timeseries"]["offroad"][start_t:, 0])
    ov = np.asarray(rollout["metric_timeseries"]["overlap"][start_t:, 0])
    return bool(off.sum() > 0), bool(ov.sum() > 0)


def check_metrics(
    pseudo_dir: Path,
    *,
    run_dir: str,
    failure_dir: Path,
    limit: int | None,
) -> dict:
    from waymax import dynamics

    cfg = yaml.safe_load(open(Path(run_dir) / ".hydra/config.yaml"))
    termination_keys = tuple(cfg["termination_keys"])
    env = make_env_for_evaluation(
        max_num_objects=int(cfg["max_num_objects"]),
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=cfg["observation_type"],
        observation_config=cfg["observation_config"],
        termination_keys=list(termination_keys),
        noisy_init=False,
    )
    max_num_objects = int(cfg["max_num_objects"])

    npz_paths = sorted(pseudo_dir.glob("*.trajectory.npz"))
    if limit is not None:
        npz_paths = npz_paths[:limit]

    rows: list[dict] = []
    t0 = time.time()
    for i, npz_path in enumerate(npz_paths):
        npz = np.load(npz_path)
        traj = np.asarray(npz["predicted_trajectory"], dtype=np.float32)
        start_t = int(npz["start_timestep"])
        meta_path = npz_path.with_name(npz_path.name.replace(".trajectory.npz", ".pseudo_gt.json"))
        meta = json.load(open(meta_path))
        failure_json = meta.get("failure_json") or str(
            failure_dir / npz_path.name.replace(".trajectory.npz", ".json")
        )
        orig = json.load(open(failure_json))

        smx_state = load_waymax_state(
            str(meta["scenariomax_tfrecord"]),
            int(meta["scenariomax_record_index"]),
            num_paths=SCENARIOMAX_NUM_PATHS,
            num_points=SCENARIOMAX_NUM_POINTS_PER_PATH,
            max_num_objects=max_num_objects,
        )
        broken = _splice_log_only(smx_state, traj, start_t)
        fixed = _splice_log_and_sim(smx_state, traj, start_t)

        broken_acc = _vmax_pointwise_accuracy(env, broken, start_t=start_t, termination_keys=termination_keys)
        fixed_acc = _vmax_pointwise_accuracy(env, fixed, start_t=start_t, termination_keys=termination_keys)
        smx_off, smx_ov = _rollout_offroad_overlap(fixed, start_t)

        womd_state = load_waymax_state(
            str(orig["tfrecord"]),
            int(orig["scenario_idx"]),
            num_paths=LOCAL_NUM_PATHS,
            num_points=LOCAL_NUM_POINTS_PER_PATH,
            max_num_objects=128,
        )
        womd_fixed = _splice_log_and_sim(womd_state, traj, start_t)
        womd_off, womd_ov = _rollout_offroad_overlap(womd_fixed, start_t)

        rows.append(
            {
                "stem": npz_path.stem,
                "broken_vmax_accuracy": broken_acc,
                "fixed_vmax_accuracy": fixed_acc,
                "smx_rollout_offroad": smx_off,
                "smx_rollout_overlap": smx_ov,
                "womd_rollout_offroad": womd_off,
                "womd_rollout_overlap": womd_ov,
                "false_offroad_from_bug": bool(broken_acc < 1.0 and fixed_acc >= 1.0),
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  metrics {i + 1}/{len(npz_paths)} ({time.time() - t0:.0f}s)", flush=True)

    n = len(rows)
    summary = {
        "n": n,
        "elapsed_s": time.time() - t0,
        "broken_vmax_accuracy_rate": float(np.mean([r["broken_vmax_accuracy"] for r in rows])) if n else 0.0,
        "fixed_vmax_accuracy_rate": float(np.mean([r["fixed_vmax_accuracy"] for r in rows])) if n else 0.0,
        "smx_rollout_offroad_rate": float(np.mean([r["smx_rollout_offroad"] for r in rows])) if n else 0.0,
        "womd_rollout_offroad_rate": float(np.mean([r["womd_rollout_offroad"] for r in rows])) if n else 0.0,
        "false_offroad_from_bug_count": int(sum(r["false_offroad_from_bug"] for r in rows)),
        "false_offroad_from_bug_rate": float(np.mean([r["false_offroad_from_bug"] for r in rows])) if n else 0.0,
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    args = _parse_args()
    pseudo_dir = Path(args.pseudo_gt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pseudo_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"pseudo_gt_dir": str(pseudo_dir), "checks": args.checks}

    if "indices" in args.checks:
        print("=== indices ===", flush=True)
        report["indices"] = check_indices(
            scenariomax_tfrecord=args.scenariomax_tfrecord,
            scenariomax_index=args.scenariomax_index,
            womd_index=args.womd_index,
        )
        print(json.dumps(report["indices"], indent=2))

    if "mapping" in args.checks:
        print("=== mapping ===", flush=True)
        report["mapping"] = check_mapping(
            pseudo_dir,
            scenariomax_index=args.scenariomax_index,
            limit=args.limit,
        )
        m = report["mapping"]
        print(
            f"  n={m['n']} match_rate={m['mapping_match_rate']:.3f} "
            f"sqlite_misses={m['sqlite_misses']} mismatches={m['index_mismatches']}"
        )
        mapping_csv = out_dir / "verify_mapping.csv"
        if m["rows"]:
            with open(mapping_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(m["rows"][0].keys()))
                w.writeheader()
                w.writerows(m["rows"])
            print(f"  wrote {mapping_csv}")

    if "metrics" in args.checks:
        if not args.run_dir:
            raise SystemExit("--run-dir is required when --checks includes metrics")
        print("=== metrics ===", flush=True)
        report["metrics"] = check_metrics(
            pseudo_dir,
            run_dir=args.run_dir,
            failure_dir=Path(args.failure_dir),
            limit=args.limit,
        )
        ms = report["metrics"]["summary"]
        print(json.dumps(ms, indent=2))
        metrics_csv = out_dir / "verify_metrics.csv"
        metric_rows = report["metrics"]["rows"]
        if metric_rows:
            with open(metrics_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
                w.writeheader()
                w.writerows(metric_rows)
            print(f"  wrote {metrics_csv}")

    summary_path = out_dir / "verify_summary.json"
    # Drop bulky per-row lists from top-level JSON (kept in CSVs).
    slim = {k: v for k, v in report.items() if k not in ("mapping", "metrics")}
    if "mapping" in report:
        slim["mapping"] = {k: v for k, v in report["mapping"].items() if k != "rows"}
    if "metrics" in report:
        slim["metrics"] = report["metrics"]["summary"]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
