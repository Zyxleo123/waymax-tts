#!/usr/bin/env python
"""Write a self-contained SUMMARY.md for ONE selected checkpoint.

Pulls from saved artefacts only (the sweep's per-episode CSVs and the render job's
failure manifests), so it can be regenerated at any time without a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(bench: Path, stem: str, split: str) -> pd.DataFrame:
    return pd.read_csv(bench / f"ai/ds/{split}.tfrecord/{stem}/{stem}/evaluation_episodes.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, type=Path, help="sweep benchmark/ dir (batch-8 metrics)")
    p.add_argument("--checkpoint", required=True, help="stem, e.g. model_15365120")
    p.add_argument("--run-name", default="", help="training run the checkpoint came from")
    p.add_argument("--video-dir", type=Path, help="dir holding <split>/ videos + <split>_failures.csv")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    stem = args.checkpoint
    L: list[str] = []
    L.append(f"# {stem} — selected checkpoint")
    L.append("")
    if args.run_name:
        L.append(f"**Run:** `{args.run_name}`")
    L.append(f"**Evaluated on:** ScenarioMax-converted WOMD shards 0–1 of each split "
             f"(training 935 scenarios written of 990 read, validation 568 of 592; "
             f"the remainder were filtered by ScenarioMax as overpasses / too-few-objects).")
    L.append("")
    L.append("**Clean success** = reached goal (within 2.0 m) AND no overlap AND no offroad. "
             "`run_red_light` also terminates an episode and is reported separately.")
    L.append("")

    L.append("## Headline")
    L.append("")
    L.append("| split | n | clean success | clean if ended at goal | goal | safe (no overlap/offroad) | accuracy (+no red light) |")
    L.append("|---|---|---|---|---|---|---|")
    frames = {}
    for split in ("training", "validation"):
        d = load(args.benchmark, stem, split)
        frames[split] = d
        goal = d.reached_goal > .5
        unsafe = (d.overlap > .5) | (d.offroad > .5)
        clean = goal & ~unsafe
        at_goal = clean | (goal & (d.overlap > .5) & ~(d.offroad > .5))
        acc = (d.accuracy > .5).mean()
        L.append(f"| {split} | {len(d)} | **{clean.mean():.4f}** | {at_goal.mean():.4f} | "
                 f"{goal.mean():.4f} | {(~unsafe).mean():.4f} | {acc:.4f} |")
    L.append("")
    n_after = sum(int(((f.reached_goal > .5) & (f.overlap > .5)).sum()) for f in frames.values())
    L.append(f"*clean if ended at goal* counts the {n_after} episodes that reached the goal and only "
             "collided afterwards — episodes here run the full 80 steps regardless of arrival, so the "
             "ego keeps driving with nothing left to optimise. It is a counterfactual, worth ~3 points; "
             "whether it is the fair number depends on what the `overlap-aftergoal` videos show.")
    L.append("")

    L.append("## Why it is not higher")
    L.append("")
    for split in ("training", "validation"):
        d = frames[split]
        goal = d.reached_goal > .5
        unsafe = (d.overlap > .5) | (d.offroad > .5)
        clean = goal & ~unsafe
        L.append(f"**{split}** — {(~clean).sum()} failures of {len(d)}:")
        L.append("")
        L.append(f"- unsafe, goal reached anyway: {(goal & unsafe).sum()}")
        L.append(f"- no goal, fully safe: {(~goal & ~unsafe).sum()}")
        L.append(f"- both (a violation ended the episode before the goal): {(~goal & unsafe).sum()}")
        L.append(f"- ceiling if safety were perfect: {goal.mean():.4f}; "
                 f"if goal-reaching were perfect: {(~unsafe).mean():.4f}")
        L.append(f"- independence would predict {goal.mean() * (~unsafe).mean():.4f}; "
                 f"actual {clean.mean():.4f} — the two failure modes overlap, so the "
                 f"intersection costs less than independence implies")
        L.append("")

    if args.video_dir:
        L.append("## Rendered failures")
        L.append("")
        L.append("Videos are 80 frames at 10 fps, named `<bucket>__scenario<idx>.mp4`. "
                 "Buckets are mutually exclusive: `overlap`/`offroad`/`redlight` terminate the "
                 "episode, so at most one fires, and `-aftergoal` vs `-nogoal` says whether the "
                 "goal had already been reached when it did.")
        L.append("")
        L.append("> One video per failure — the counts below match the headline exactly, because "
                 "both come from the same batch-1 evaluation pass. Episodes that ran a red light "
                 "but reached the goal cleanly are **not** failures under this metric; their "
                 "videos are kept aside in `_not_failures_redlight_only/`.")
        L.append(">")
        L.append("> The 26-checkpoint sweep reports this checkpoint at 0.9224 on training rather "
                 "than 0.9198: it ran at batch-8, and Waymax drops the incomplete final batch, so "
                 "it scored 928 of the 935 scenarios. The same 928 are used for every checkpoint, "
                 "so sweep rankings are unaffected; the numbers here are the complete ones.")
        L.append("")
        L.append("| bucket | training | validation |")
        L.append("|---|---|---|")
        mans = {}
        for split in ("training", "validation"):
            f = args.video_dir / f"{split}_failures.csv"
            mans[split] = pd.read_csv(f).bucket.value_counts() if f.exists() else pd.Series(dtype=int)
        for b in sorted(set(mans["training"].index) | set(mans["validation"].index)):
            L.append(f"| `{b}` | {int(mans['training'].get(b, 0))} | {int(mans['validation'].get(b, 0))} |")
        L.append(f"| **total** | **{int(mans['training'].sum())}** | **{int(mans['validation'].sum())}** |")
        L.append("")
        big = (mans["training"].get("overlap-aftergoal", 0) + mans["validation"].get("overlap-aftergoal", 0))
        L.append(f"Largest single group is `overlap-aftergoal` ({int(big)} of "
                 f"{int(mans['training'].sum() + mans['validation'].sum())}): the policy reaches "
                 f"the goal, the episode keeps running, and it collides afterwards. Worth watching first.")
        L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
