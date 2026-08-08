#!/usr/bin/env python
"""Aggregate the per-checkpoint V-Max evaluation CSVs into a comparable report.

Reads every ``evaluation_episodes.csv`` written by ``vmax.scripts.evaluate`` under
``--eval-out`` and reports, per (checkpoint, split):

  clean success = reached_goal AND NOT overlap AND NOT offroad

and, among the episodes that are *not* clean successes, the rate of each failure
mode and of every combination of them. Rates over non-clean episodes sum to 1.

Pure post-processing over saved CSVs — no GPU, no simulator, so the tables and
plots can be re-rendered any number of times without re-running the sweep.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# The three failure axes the report is built from, in a fixed order so the
# combination labels are stable across runs.
AXES = ("no_goal", "overlap", "offroad")

COMBOS = (
    ("no_goal",),
    ("overlap",),
    ("offroad",),
    ("no_goal", "overlap"),
    ("no_goal", "offroad"),
    ("overlap", "offroad"),
    ("no_goal", "overlap", "offroad"),
)


def _step_of(stem: str) -> int:
    """Training step encoded in a checkpoint name; model_final sorts last."""
    if stem == "model_final":
        return 1 << 62
    match = re.search(r"(\d+)", stem)
    return int(match.group(1)) if match else -1


def find_csvs(eval_out: Path) -> list[tuple[str, str, Path]]:
    """Return (split, checkpoint_stem, csv_path) for every finished evaluation."""
    found = []
    for csv in sorted(eval_out.glob("ai/ds/*.tfrecord/*/*/evaluation_episodes.csv")):
        split = csv.parents[2].name.replace(".tfrecord", "")
        stem = csv.parents[0].name
        found.append((split, stem, csv))
    return found


def summarize(df: pd.DataFrame, goal_col: str) -> dict:
    """Clean-success rate plus the failure-combination breakdown for one CSV."""
    for column in (goal_col, "overlap", "offroad"):
        if column not in df.columns:
            raise KeyError(f"column '{column}' missing; have {sorted(df.columns)}")

    flags = pd.DataFrame(
        {
            "no_goal": df[goal_col].to_numpy() < 0.5,
            "overlap": df["overlap"].to_numpy() > 0.5,
            "offroad": df["offroad"].to_numpy() > 0.5,
        },
    )

    n = len(df)
    clean = ~flags.any(axis=1)
    n_clean = int(clean.sum())
    n_dirty = n - n_clean

    row = {
        "n_scenarios": n,
        "n_clean_success": n_clean,
        "clean_success_rate": n_clean / n if n else float("nan"),
        "n_non_clean": n_dirty,
        # Marginal rates over ALL episodes, for context.
        "rate_all_no_goal": float(flags["no_goal"].mean()) if n else float("nan"),
        "rate_all_overlap": float(flags["overlap"].mean()) if n else float("nan"),
        "rate_all_offroad": float(flags["offroad"].mean()) if n else float("nan"),
    }

    dirty = flags[~clean]
    # Marginal rates among non-clean episodes (these overlap, they do not sum to 1).
    for axis in AXES:
        row[f"nonclean_any_{axis}"] = float(dirty[axis].mean()) if n_dirty else float("nan")
    # Exclusive combinations among non-clean episodes (these DO sum to 1).
    for combo in COMBOS:
        mask = pd.Series(True, index=dirty.index)
        for axis in AXES:
            mask &= dirty[axis] if axis in combo else ~dirty[axis]
        label = "+".join(combo)
        row[f"nonclean_only_{label}"] = float(mask.mean()) if n_dirty else float("nan")
        row[f"n_nonclean_only_{label}"] = int(mask.sum())

    # Extra context that is cheap to carry and often explains the failures.
    for extra in ("run_red_light", "episode_length", "accuracy", "min_distance_to_goal"):
        if extra in df.columns:
            row[f"mean_{extra}"] = float(df[extra].mean())

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-out", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument(
        "--goal-col",
        default="reached_goal",
        help="Goal column; reached_goal_2m/3m/5m give a looser radius (default: reached_goal)",
    )
    args = parser.parse_args()

    csvs = find_csvs(args.eval_out)
    if not csvs:
        print(f"No evaluation_episodes.csv found under {args.eval_out} — nothing to aggregate.")
        return

    rows = []
    for split, stem, csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # a truncated CSV from a killed job
            print(f"SKIP {split}/{stem}: unreadable ({exc})")
            continue
        if df.empty:
            print(f"SKIP {split}/{stem}: empty")
            continue
        try:
            row = summarize(df, args.goal_col)
        except KeyError as exc:
            print(f"SKIP {split}/{stem}: {exc}")
            continue
        rows.append({"split": split, "checkpoint": stem, "step": _step_of(stem), **row})

    if not rows:
        print("No usable evaluations.")
        return

    table = pd.DataFrame(rows).sort_values(["split", "step"]).reset_index(drop=True)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / "checkpoint_summary.csv"
    table.to_csv(csv_path, index=False)
    with (args.report_dir / "checkpoint_summary.jsonl").open("w") as handle:
        for row in table.to_dict(orient="records"):
            handle.write(json.dumps(row) + "\n")

    lines: list[str] = []
    for split, group in table.groupby("split"):
        best = group.loc[group["clean_success_rate"].idxmax()]
        lines.append("")
        lines.append("=" * 108)
        lines.append(f"SPLIT: {split}   ({int(group['n_scenarios'].iloc[0])} scenarios)")
        lines.append("=" * 108)
        header = (
            f"{'checkpoint':<22} {'n':>6} {'clean%':>8} {'non-clean':>10}  "
            f"{'|among non-clean:':<18}{'goal-':>7}{'ovlp':>7}{'offr':>7}"
            f"{'g+ov':>7}{'g+of':>7}{'ov+of':>7}{'all3':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in group.iterrows():
            lines.append(
                f"{r['checkpoint']:<22} {int(r['n_scenarios']):>6} "
                f"{100 * r['clean_success_rate']:>7.2f}% {int(r['n_non_clean']):>10}  "
                f"{'':<18}"
                f"{100 * r['nonclean_only_no_goal']:>6.1f}%"
                f"{100 * r['nonclean_only_overlap']:>6.1f}%"
                f"{100 * r['nonclean_only_offroad']:>6.1f}%"
                f"{100 * r['nonclean_only_no_goal+overlap']:>6.1f}%"
                f"{100 * r['nonclean_only_no_goal+offroad']:>6.1f}%"
                f"{100 * r['nonclean_only_overlap+offroad']:>6.1f}%"
                f"{100 * r['nonclean_only_no_goal+overlap+offroad']:>6.1f}%",
            )
        lines.append("")
        lines.append(
            f"BEST on {split}: {best['checkpoint']} — clean success "
            f"{100 * best['clean_success_rate']:.2f}% "
            f"({int(best['n_clean_success'])}/{int(best['n_scenarios'])})",
        )

    lines.append("")
    lines.append("Columns: 'goal-' = failed to reach goal only; 'ovlp' = collision only;")
    lines.append("'offr' = offroad only; the rest are the combinations. These are exclusive")
    lines.append("shares of the non-clean episodes and sum to 100%.")
    lines.append(f"Goal column used: {args.goal_col}")

    report = "\n".join(lines)
    (args.report_dir / "checkpoint_summary.txt").write_text(report + "\n")
    print(report)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
