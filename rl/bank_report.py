#!/usr/bin/env python3
"""Report what a bank has harvested, and why the rest is unbanked.

`bank_manifest.json` records, per scenario, whether the last deterministic rollout
reached the goal / went offroad / collided. That partitions the not-yet-banked
scenarios by cause, which is what decides where the next run's budget goes:

  * never reaches the goal  -> a behaviour problem; more RL may help
  * reaches but goes offroad -> a precision/geometry problem; offroad_margin,
    off_route=0, and the road-edge clearance measurements are the levers
  * reaches but collides     -> check reactive-IDM vs log-replay before tuning

Usage:
    python -m rl.bank_report <run_dir_or_bank_dir> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _find_manifest(path: str) -> str:
    for cand in (
        path,
        os.path.join(path, "bank_manifest.json"),
        os.path.join(path, "bank", "bank_manifest.json"),
    ):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"No bank_manifest.json under {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Run dir, bank dir, or the manifest itself.")
    p.add_argument("--csv", help="Also write a per-scenario table here.")
    args = p.parse_args()

    manifest_path = _find_manifest(args.path)
    with open(manifest_path) as f:
        m = json.load(f)

    n = m["num_scenarios"]
    banked = {int(k): v for k, v in m["banked"].items()}
    remaining = m["remaining"]
    last = m.get("last_eval")

    print(f"manifest : {manifest_path}")
    print(f"banked   : {len(banked)}/{n} ({100 * len(banked) / n:.1f}%)")
    print(f"remaining: {len(remaining)}")

    if banked:
        steps = sorted(v["step"] for v in banked.values())
        print(f"\nbanked at: first={steps[0]} median={steps[len(steps) // 2]} last={steps[-1]}")
        at_first = sum(1 for s in steps if s == steps[0])
        print(f"  {at_first} of {len(banked)} banked at the earliest eval "
              f"(i.e. solved with little or no RL)")

    if not last:
        print("\nNo per-scenario flags recorded (run predates flag logging).")
        return

    per = last["per_scenario"]
    print(f"\n--- why the {len(remaining)} unbanked are unbanked (flags @ step {last['step']}) ---")

    buckets: dict[str, list[int]] = {
        "never reaches goal": [],
        "reaches, offroad only": [],
        "reaches, collision only": [],
        "reaches, offroad + collision": [],
        "clean but not banked (?)": [],
    }
    for i in remaining:
        f = per.get(str(i))
        if f is None:
            continue
        reached, off, ovl = f["reached_goal"] > 0.5, f["offroad"] > 0.5, f["overlap"] > 0.5
        if not reached:
            buckets["never reaches goal"].append(i)
        elif off and ovl:
            buckets["reaches, offroad + collision"].append(i)
        elif off:
            buckets["reaches, offroad only"].append(i)
        elif ovl:
            buckets["reaches, collision only"].append(i)
        else:
            # Should be empty: clean implies banked. Non-empty means the last eval
            # solved it after the manifest was written, or something is off.
            buckets["clean but not banked (?)"].append(i)

    for name, idxs in buckets.items():
        if not idxs:
            continue
        pct = 100 * len(idxs) / max(len(remaining), 1)
        print(f"  {name:32s} {len(idxs):4d}  ({pct:5.1f}% of unbanked)  e.g. {idxs[:8]}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("scenario_idx,banked,banked_step,reached_goal,offroad,overlap\n")
            for i in range(n):
                fl = per.get(str(i), {})
                f.write(
                    f"{i},{int(i in banked)},{banked.get(i, {}).get('step', '')},"
                    f"{fl.get('reached_goal', '')},{fl.get('offroad', '')},{fl.get('overlap', '')}\n"
                )
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
