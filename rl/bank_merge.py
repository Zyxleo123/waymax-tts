#!/usr/bin/env python3
"""Merge several banks into one, keeping the best trajectory per scenario.

Why this exists
---------------
The bank's premise is "a scenario only has to succeed once, at any point in training" --
a union over time. That union does not stop at a run boundary. Measured 2026-07-17 across
five runs of the 218-scene failure set: the best single run banked 202, but only **11**
scenarios were unbanked in *every* run. 47 are simply flaky -- solved in some runs, not
others. Reading one manifest at a time throws that away for free.

"Best" = lowest peak raw yaw rate over the kept prefix (`rl.bank.max_yaw_rate`), the same
statistic the live bank upgrades on, so a merged bank and a resumed run agree on what
"better" means. Every trajectory is truncated at closest approach to the goal on the way
through, and `--repair` additionally runs `rl.repair.path_savgol_w7` (see that module for
why the shimmy is fixed post-hoc rather than in the reward).

The source banks must all have been harvested from the SAME tfrecord in file order --
scenario identity is the record index, so mixing datasets would silently file trajectories
under the wrong scene.

Report only::

    python -m rl.bank_merge --banks /path/run1/bank /path/run2/bank

Write the merged, truncated, repaired bank::

    python -m rl.bank_merge --banks /path/*/bank --out-dir /path/bank_merged --repair --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "V-Max"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl.bank import MAX_YAW_RATE_RAD_S, goal_cut_index, max_yaw_rate, truncate_at  # noqa: E402
from rl.bank_regate import DEFAULT_DATASET, read_goals  # noqa: E402
from rl.repair import repair_path_savgol  # noqa: E402

TRAJ_KEYS = ("x", "y", "yaw", "vel_x", "vel_y", "valid")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--banks", nargs="+", required=True, help="Bank dirs (or their manifests).")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Tfrecord all the banks were harvested from.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument("--repair", action="store_true", help="Apply rl.repair.path_savgol_w7 before scoring.")
    p.add_argument("--start-t", type=int, default=11, help="First policy-controlled step (91 - scenario_length).")
    p.add_argument("--max-num-objects", type=int, default=32)
    p.add_argument("--chunk", type=int, default=4)
    return p.parse_args()


def _manifest_of(path: str) -> tuple[Path, dict] | None:
    """None when there is no manifest -- `--banks */bank` legitimately globs aborted runs.

    Skipping is safe *because* it is loud: a bank with no manifest banked nothing (the
    manifest is written on every update), so there is no harvest to lose.
    """
    p = Path(path)
    mp = p if p.is_file() else p / "bank_manifest.json"
    if not mp.is_file():
        print(f"  skip: no bank_manifest.json at {path} (aborted run?)")
        return None
    return mp.parent, json.load(open(mp))


def main() -> int:
    args = _parse_args()
    banks = [b for b in (_manifest_of(x) for x in args.banks) if b is not None]
    if not banks:
        raise SystemExit("No banks with a manifest among --banks.")

    counts = {int(m["num_scenarios"]) for _, m in banks}
    if len(counts) != 1:
        raise SystemExit(f"Banks disagree on num_scenarios ({counts}); they are not the same dataset.")
    num_scenarios = counts.pop()

    goals = read_goals(args.dataset, num_scenarios, args.max_num_objects, args.chunk)

    # scenario -> list of (quality, source_name, entry, trajectory)
    best: dict[int, dict] = {}
    per_source_wins: dict[str, int] = {}

    for bank_dir, man in banks:
        source = bank_dir.parent.name
        per_source_wins.setdefault(source, 0)
        for k, entry in man["banked"].items():
            i = int(k)
            npz = Path(entry.get("path", bank_dir / "trajectories" / f"scenario_{i:05d}.npz"))
            if not npz.is_file():
                npz = bank_dir / "trajectories" / f"scenario_{i:05d}.npz"
                if not npz.is_file():
                    print(f"  WARN: {source} lists scenario {i} but its npz is missing; skipping")
                    continue
            d = np.load(npz)
            traj = {k2: np.asarray(d[k2])[None, ...] for k2 in TRAJ_KEYS}
            traj = {k2: (v.astype(bool) if k2 == "valid" else v.astype(np.float64)) for k2, v in traj.items()}

            if args.repair:
                traj = repair_path_savgol(traj, window=7, start_t=args.start_t)

            xy = np.stack([traj["x"], traj["y"]], -1).astype(float)
            dist = np.linalg.norm(xy - goals[i][None, None, :], axis=-1)
            cut = goal_cut_index(dist, np.asarray(traj["valid"]), start_t=args.start_t)
            traj = truncate_at(traj, cut)
            q = float(max_yaw_rate(np.asarray(traj["yaw"], float), cut, start_t=args.start_t)[0])

            prev = best.get(i)
            if prev is None or q < prev["quality"]:
                best[i] = {
                    "quality": q,
                    "source": source,
                    "step": int(entry.get("step", -1)),
                    "cut_step": int(cut[0]),
                    "min_dist_m": float(dist[0, args.start_t:].min()),
                    "traj": {k2: v[0] for k2, v in traj.items()},
                }

    for i, e in best.items():
        per_source_wins[e["source"]] += 1

    n = len(best)
    q = np.array([e["quality"] for e in best.values()])
    print(f"merged {len(banks)} banks over {num_scenarios} scenarios"
          f"{'  (repaired: path_savgol_w7)' if args.repair else ''}\n")
    for (bank_dir, man) in banks:
        src = bank_dir.parent.name
        print(f"  {src[:52]:52s} banked {man['num_banked']:3d}  -> best-of for {per_source_wins.get(src,0):3d}")
    print()
    print(f"  MERGED: {n}/{num_scenarios} banked ({n/num_scenarios:.1%})"
          f"   |  never banked anywhere: {num_scenarios - n}")
    print(f"  missing: {sorted(set(range(num_scenarios)) - set(best))}")
    print()
    print(f"  peak yaw rate: p50 {np.median(q):.2f}  p90 {np.percentile(q,90):.2f}  max {q.max():.2f} rad/s"
          f"   | <= {MAX_YAW_RATE_RAD_S}: {int((q <= MAX_YAW_RATE_RAD_S).sum())}/{n}")
    md = np.array([e["min_dist_m"] for e in best.values()])
    print(f"  closest approach to goal: p50 {np.median(md):.2f} m  max {md.max():.2f} m"
          f"   | within 2 m: {int((md < 2).sum())}/{n}")

    if args.write:
        out_dir = Path(args.out_dir or "bank_merged")
        traj_dir = out_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        banked = {}
        for i, e in sorted(best.items()):
            path = traj_dir / f"scenario_{i:05d}.npz"
            np.savez_compressed(path, **{k2: (v.astype(np.float32) if k2 != "valid" else v)
                                         for k2, v in e["traj"].items()})
            banked[str(i)] = {
                "step": e["step"], "path": str(path), "quality": e["quality"],
                "cut_step": e["cut_step"], "source": e["source"],
            }
        payload = {
            "num_scenarios": num_scenarios,
            "num_banked": len(banked),
            "banked": banked,
            "remaining": sorted(set(range(num_scenarios)) - set(best)),
            "merged_from": [str(b) for b, _ in banks],
            "repaired": bool(args.repair),
            "truncated": True,
        }
        tmp = out_dir / "bank_manifest.json.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, out_dir / "bank_manifest.json")
        print(f"\nwrote {len(banked)} trajectories -> {out_dir}")
    else:
        print("\n(report only; pass --write to produce the merged bank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
