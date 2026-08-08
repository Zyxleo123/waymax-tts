#!/usr/bin/env python
"""Partition evaluation episodes by *terminal cause*, not by co-occurring flags.

`termination_keys = [offroad, overlap, run_red_light]` and BraxWrapper.termination
sums them and stops the episode the moment any one fires
(`vmax/simulator/wrappers/interfaces/brax.py:165`). So an episode can only ever
carry two violation flags if they fired on the *same* step — combination buckets
like "overlap+offroad" are degenerate by construction, not informative.

This report therefore uses a mutually exclusive partition:

    clean success                  goal reached, no violation
    no-goal (safe)                 ran to the horizon, never within the radius
    overlap  / offroad / red-light terminated by that violation, each split by
                                   whether the goal had already been reached
    multi (same step)              >=2 keys fired on the same step

Buckets sum to 100% of episodes. Reads only saved CSVs; no simulator.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


VIOLATIONS = ("overlap", "offroad", "run_red_light")
SHORT = {"overlap": "overlap", "offroad": "offroad", "run_red_light": "redlight"}


def _step_of(stem: str) -> int:
    if stem == "model_final":
        return 1 << 62
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else -1


def categorize(df: pd.DataFrame, goal_col: str) -> pd.DataFrame:
    """Attach a `bucket` column giving each episode's mutually exclusive category."""
    flags = {v: df[v].to_numpy() > 0.5 for v in VIOLATIONS}
    goal = df[goal_col].to_numpy() > 0.5
    n_violations = sum(f.astype(int) for f in flags.values())

    bucket = pd.Series("", index=df.index, dtype=object)
    bucket[(n_violations == 0) & goal] = "clean success"
    bucket[(n_violations == 0) & ~goal] = "no-goal (safe)"
    for key, f in flags.items():
        sel = (n_violations == 1) & f
        bucket[sel & goal] = f"{SHORT[key]} (goal reached first)"
        bucket[sel & ~goal] = f"{SHORT[key]} (no goal)"
    bucket[n_violations > 1] = "multi (same step)"

    out = df.copy()
    out["bucket"] = bucket
    out["_goal"] = goal
    return out


ORDER = [
    "clean success",
    "no-goal (safe)",
    "overlap (no goal)",
    "overlap (goal reached first)",
    "offroad (no goal)",
    "offroad (goal reached first)",
    "redlight (no goal)",
    "redlight (goal reached first)",
    "multi (same step)",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-out", required=True, type=Path)
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--goal-col", default="reached_goal")
    p.add_argument("--detail-checkpoint", default=None, help="Extra drill-down for this stem")
    args = p.parse_args()

    rows, detail_frames = [], {}
    for csv in sorted(args.eval_out.glob("ai/ds/*.tfrecord/*/*/evaluation_episodes.csv")):
        split = csv.parents[2].name.replace(".tfrecord", "")
        stem = csv.parents[0].name
        df = pd.read_csv(csv)
        if df.empty:
            continue
        cat = categorize(df, args.goal_col)
        counts = cat["bucket"].value_counts()
        row = {"split": split, "checkpoint": stem, "step": _step_of(stem), "n": len(cat)}
        for b in ORDER:
            row[b] = int(counts.get(b, 0))
        rows.append(row)
        if args.detail_checkpoint and stem == args.detail_checkpoint:
            detail_frames[split] = cat

    if not rows:
        print(f"No evaluations under {args.eval_out}")
        return

    table = pd.DataFrame(rows).sort_values(["split", "step"]).reset_index(drop=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.report_dir / "failure_buckets.csv", index=False)

    lines: list[str] = []
    for split, g in table.groupby("split"):
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"SPLIT: {split}   —  % of all episodes, buckets are mutually exclusive and sum to 100")
        lines.append("=" * 100)
        hdr = f"{'checkpoint':<18}{'n':>5}{'clean':>8}{'noGoal':>8}{'ovlp':>7}{'ovlpG':>7}{'offr':>7}{'offrG':>7}{'red':>7}{'redG':>7}{'multi':>7}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for _, r in g.iterrows():
            n = r["n"]
            def pct(b): return 100 * r[b] / n
            lines.append(
                f"{r['checkpoint']:<18}{n:>5}"
                f"{pct('clean success'):>7.1f}%{pct('no-goal (safe)'):>7.1f}%"
                f"{pct('overlap (no goal)'):>6.1f}%{pct('overlap (goal reached first)'):>6.1f}%"
                f"{pct('offroad (no goal)'):>6.1f}%{pct('offroad (goal reached first)'):>6.1f}%"
                f"{pct('redlight (no goal)'):>6.1f}%{pct('redlight (goal reached first)'):>6.1f}%"
                f"{pct('multi (same step)'):>6.1f}%",
            )

    for split, cat in detail_frames.items():
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"DRILL-DOWN: {args.detail_checkpoint} on {split}")
        lines.append("=" * 100)
        n = len(cat)
        for b in ORDER:
            sub = cat[cat["bucket"] == b]
            if sub.empty:
                continue
            lines.append(f"{b:<32} {len(sub):>5}  ({100 * len(sub) / n:>5.1f}%)  median ep_len {sub['episode_length'].median():>5.1f}")

        safe_miss = cat[cat["bucket"] == "no-goal (safe)"]
        if not safe_miss.empty:
            lines.append("")
            lines.append(f"  'no-goal (safe)' — how far short? (n={len(safe_miss)}, goal radius 2.0 m)")
            for col in ("min_distance_to_goal", "final_distance_to_goal"):
                q = safe_miss[col].quantile([0.25, 0.5, 0.75]).round(2).tolist()
                lines.append(f"    {col:<24} p25/p50/p75 = {q[0]:>7} / {q[1]:>7} / {q[2]:>7} m")
            for radius, col in ((3.0, "reached_goal_3m"), (5.0, "reached_goal_5m")):
                if col in safe_miss.columns:
                    share = 100 * (safe_miss[col] > 0.5).mean()
                    lines.append(f"    would count at {radius:.0f} m radius: {share:.1f}%")
            if "sdc_progression" in safe_miss.columns:
                lines.append(f"    median sdc_progression   {safe_miss['sdc_progression'].median():.3f}")

        coll = cat[cat["bucket"].str.startswith("overlap")]
        if not coll.empty and "at_fault_collision" in coll.columns:
            lines.append("")
            lines.append(f"  collisions (n={len(coll)}): at-fault {100 * (coll['at_fault_collision'] > 0.5).mean():.1f}%")

    text = "\n".join(lines)
    (args.report_dir / "failure_buckets.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
