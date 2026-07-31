#!/usr/bin/env python3
"""Compare vendored-ES reproduction outcomes against the original failure-set labels.

Note: the original 24-failure run scanned all scenes sequentially (batch offset
affects diffusion noise), so exact bit-match is not expected. This reports
per-scene outcomes and agreement, and doubles as the Stage-0 instrumentation table.
"""
from __future__ import annotations
import argparse, glob, json, os, re

# Original labels from simulation_es_training_20260723_184324/20260723_184447
ORIG_FAIL = {
    "tfrecord-00000-of-01000": {71, 92, 113, 134, 187, 252, 315, 377},
    "tfrecord-00001-of-01000": {45, 62, 147, 158, 200, 243, 289, 360, 463},
    "tfrecord-00002-of-01000": {0, 13, 20, 24, 70, 94, 97},
}
ORIG_CONTROL = {
    "tfrecord-00000-of-01000": {0, 1, 2, 3},
    "tfrecord-00001-of-01000": {0, 1, 2, 3},
    "tfrecord-00002-of-01000": {1, 2, 3, 4},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {}
    for f in glob.glob(os.path.join(args.out, "**", "*.scenario_*.json"), recursive=True):
        base = os.path.basename(f)
        if base == "summary.json":
            continue
        m = re.search(r"(tfrecord-\d+-of-\d+)\.scenario_(\d+)\.json", base)
        if not m:
            continue
        rec, idx = m.group(1), int(m.group(2))
        d = json.load(open(f))
        rows[(rec, idx)] = d

    def label(rec, idx):
        if idx in ORIG_FAIL.get(rec, set()):
            return "FAIL"
        if idx in ORIG_CONTROL.get(rec, set()):
            return "ctrl-SUCCESS"
        return "?"

    print(f"{'tfrecord':<24} {'idx':>4} {'orig':>12} {'repro':>7} "
          f"{'reached':>7} {'overlap':>7} {'offroad':>7} {'goal_d':>7}")
    agree_fail = tot_fail = agree_ctrl = tot_ctrl = 0
    for (rec, idx) in sorted(rows):
        d = rows[(rec, idx)]
        orig = label(rec, idx)
        repro = "SUCCESS" if d.get("success") else "FAIL"
        print(f"{rec:<24} {idx:>4} {orig:>12} {repro:>7} "
              f"{str(d.get('goal_reached')):>7} {str(d.get('overlap')):>7} "
              f"{str(d.get('offroad')):>7} {d.get('min_goal_distance_m', -1):>7.2f}")
        if orig == "FAIL":
            tot_fail += 1
            agree_fail += int(repro == "FAIL")
        elif orig == "ctrl-SUCCESS":
            tot_ctrl += 1
            agree_ctrl += int(repro == "SUCCESS")

    print("\n=== agreement with original labels (informational; batching differs) ===")
    print(f"failures reproduced as FAIL:      {agree_fail}/{tot_fail}")
    print(f"controls reproduced as SUCCESS:   {agree_ctrl}/{tot_ctrl}")
    # Stage-0 headline: how many failures reach goal but violate a safety term
    reach_unsafe = sum(
        1 for (rec, idx), d in rows.items()
        if idx in ORIG_FAIL.get(rec, set()) and d.get("goal_reached")
        and (d.get("overlap") or d.get("offroad"))
    )
    print(f"failures that REACH goal but are unsafe (offroad/overlap): {reach_unsafe}")


if __name__ == "__main__":
    main()
