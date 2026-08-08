#!/usr/bin/env python
"""Select failing scenarios from an evaluation CSV, and label the rendered videos.

Two modes:

  --mode indexes   print the space-separated scenario indexes of every NON-CLEAN
                   episode, and write a manifest CSV describing each one.
  --mode label     rename V-Max's `mp4/eval_<idx>.mp4` files into a flat, readable
                   `<bucket>__scenario<idx>.mp4` using that manifest.

Clean success is the user's definition — reached goal AND no overlap AND no offroad.
`run_red_light` also terminates the episode, so it is recorded in the bucket label
but does not by itself make an episode a failure.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def bucket_of(row, goal_col: str) -> str:
    """Terminal cause + goal status. Violations are exclusive (first one ends the episode)."""
    goal = row[goal_col] > 0.5
    parts = [k for k in ("overlap", "offroad", "run_red_light") if row.get(k, 0) > 0.5]
    if not parts:
        return "clean" if goal else "nogoal-safe"
    tag = "+".join({"overlap": "overlap", "offroad": "offroad", "run_red_light": "redlight"}[p] for p in parts)
    label = f"{tag}-{'aftergoal' if goal else 'nogoal'}"
    # A red light alone is NOT a clean-success failure under the reported metric
    # (goal AND no overlap AND no offroad). It still ends the episode, so keep the
    # descriptive label, but mark it clean so the rendered set matches the sweep.
    if goal and parts == ["run_red_light"]:
        return "clean"
    return label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path, help="evaluation_episodes.csv from the batch-1 metrics pass")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--mode", choices=["indexes", "label"], required=True)
    p.add_argument("--goal-col", default="reached_goal")
    p.add_argument("--mp4-dir", type=Path, help="label mode: where V-Max wrote eval_<idx>.mp4")
    p.add_argument("--out-dir", type=Path, help="label mode: destination for renamed videos")
    args = p.parse_args()

    if args.mode == "indexes":
        df = pd.read_csv(args.csv)
        df["bucket"] = df.apply(lambda r: bucket_of(r, args.goal_col), axis=1)
        fail = df[df.bucket != "clean"].copy()
        keep = [c for c in ("scenario_index", "bucket", args.goal_col, "overlap", "offroad",
                            "run_red_light", "at_fault_collision", "episode_length",
                            "min_distance_to_goal", "final_distance_to_goal") if c in fail.columns]
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        fail[keep].to_csv(args.manifest, index=False)
        # stdout is consumed by the job as the --scenario_indexes argument list
        print(" ".join(str(i) for i in fail["scenario_index"].tolist()))
        return

    # label mode
    man = pd.read_csv(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    moved = missing = 0
    for _, r in man.iterrows():
        idx = int(r["scenario_index"])
        src = args.mp4_dir / f"eval_{idx}.mp4"
        if not src.exists():
            missing += 1
            continue
        shutil.move(str(src), str(args.out_dir / f"{r['bucket']}__scenario{idx:04d}.mp4"))
        moved += 1
    print(f"labelled {moved} videos into {args.out_dir}" + (f" ({missing} expected but missing)" if missing else ""))


if __name__ == "__main__":
    main()
