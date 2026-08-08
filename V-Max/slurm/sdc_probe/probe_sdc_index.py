# Copyright 2025 Valeo.

"""Measure where the SDC sits in the raw WOMD tf_example object ordering.

Waymax's WOMD loader truncates objects with a plain ``v[:max_num_objects]``
(``waymax/dataloader/womd_dataloader.py``, next to a standing
``TODO(b/246965197) check sdc included if it is needed``). ScenarioMax hides this
by moving the SDC to index 0 and sorting the rest by distance before writing, so
raw WOMD is the only path where the slice can decide anything.

This job answers the two questions that decide whether that matters:

  1. How often is ``sdc_track_index >= max_num_objects``?  That is an outright
     dropped ego -- ``is_sdc`` becomes all-false for the scenario.
  2. How many valid objects does the slice discard?  Even with the SDC retained,
     raw keeps the first N in file order rather than the N nearest, so the
     LQ observation's nearest-16 selection picks from a different pool.

Reads the tf_example fields directly (``state/is_sdc``, ``state/type``); it never
imports waymax, so it needs no GPU and cannot be perturbed by the loader itself.

Writes one JSONL row per scenario plus a JSON summary, so the numbers can be
re-tabulated without re-reading the shards.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf


# Raw WOMD 1.3.1 tf_example stores a fixed 128 object slots per scenario.
WOMD_MAX_OBJECTS = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, help="Glob for the tf_example shards to read.")
    parser.add_argument("--num_shards", type=int, default=20, help="How many shards to sample (0 = all).")
    parser.add_argument(
        "--max_num_objects",
        type=int,
        default=64,
        help="The truncation the training run uses; the threshold this probe scores against.",
    )
    parser.add_argument("--out_dir", required=True, help="Directory for rows.jsonl and summary.json.")
    return parser.parse_args()


def _feature_spec() -> dict:
    return {
        "state/is_sdc": tf.io.FixedLenFeature([WOMD_MAX_OBJECTS], tf.int64),
        "state/type": tf.io.FixedLenFeature([WOMD_MAX_OBJECTS], tf.float32),
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    shards = sorted(glob.glob(args.shards))
    if not shards:
        raise SystemExit(f"No shards matched: {args.shards}")
    if args.num_shards > 0:
        shards = shards[: args.num_shards]

    print(f"Reading {len(shards)} shard(s), threshold max_num_objects={args.max_num_objects}", flush=True)

    spec = _feature_spec()
    rows_path = os.path.join(args.out_dir, "rows.jsonl")

    sdc_indices: list[int] = []
    valid_counts: list[int] = []
    n_no_sdc = 0
    n_dropped = 0
    n_scenarios = 0
    per_shard = Counter()

    with open(rows_path, "w") as fout:
        for shard in shards:
            for record in tf.data.TFRecordDataset(shard, compression_type=""):
                parsed = tf.io.parse_single_example(record, spec)
                is_sdc = parsed["state/is_sdc"].numpy()
                # `state/type` is -1 for an unfilled object slot; anything >= 0 is a real track.
                n_valid = int((parsed["state/type"].numpy() >= 0).sum())

                where = np.flatnonzero(is_sdc == 1)
                n_scenarios += 1

                if where.size == 0:
                    # No ego flagged at all -- a malformed record, not a truncation effect.
                    n_no_sdc += 1
                    sdc_index = -1
                else:
                    sdc_index = int(where[0])
                    sdc_indices.append(sdc_index)
                    if sdc_index >= args.max_num_objects:
                        n_dropped += 1
                        per_shard[os.path.basename(shard)] += 1

                valid_counts.append(n_valid)
                fout.write(
                    json.dumps(
                        {
                            "shard": os.path.basename(shard),
                            "sdc_index": sdc_index,
                            "num_valid_objects": n_valid,
                            "num_sdc_flagged": int(where.size),
                        },
                    )
                    + "\n",
                )

            print(f"  {os.path.basename(shard)}: {n_scenarios} scenarios so far", flush=True)

    idx = np.asarray(sdc_indices)
    valid = np.asarray(valid_counts)
    # How many real tracks the slice throws away, per scenario.
    truncated = np.clip(valid - args.max_num_objects, 0, None)

    summary = {
        "num_scenarios": n_scenarios,
        "num_shards": len(shards),
        "max_num_objects": args.max_num_objects,
        "sdc_dropped_count": n_dropped,
        "sdc_dropped_frac": (n_dropped / n_scenarios) if n_scenarios else 0.0,
        "scenarios_without_sdc_flag": n_no_sdc,
        "sdc_index": {
            "max": int(idx.max()) if idx.size else None,
            "mean": float(idx.mean()) if idx.size else None,
            "p50": float(np.percentile(idx, 50)) if idx.size else None,
            "p95": float(np.percentile(idx, 95)) if idx.size else None,
            "p99": float(np.percentile(idx, 99)) if idx.size else None,
            "frac_nonzero": float((idx != 0).mean()) if idx.size else None,
        },
        "valid_objects": {
            "mean": float(valid.mean()) if valid.size else None,
            "p50": float(np.percentile(valid, 50)) if valid.size else None,
            "p95": float(np.percentile(valid, 95)) if valid.size else None,
            "max": int(valid.max()) if valid.size else None,
        },
        "objects_discarded_by_slice": {
            "frac_scenarios_losing_any": float((truncated > 0).mean()) if truncated.size else None,
            "mean_lost": float(truncated.mean()) if truncated.size else None,
            "p95_lost": float(np.percentile(truncated, 95)) if truncated.size else None,
            "max_lost": int(truncated.max()) if truncated.size else None,
        },
        "worst_shards": per_shard.most_common(10),
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as fout:
        json.dump(summary, fout, indent=2)

    print("\n===== SDC index probe =====")
    print(json.dumps(summary, indent=2))
    print(f"\nRows:    {rows_path}")
    print(f"Summary: {summary_path}")

    if n_dropped:
        print(
            f"\nVERDICT: the SDC is sliced away in {n_dropped}/{n_scenarios} scenarios "
            f"({100 * n_dropped / n_scenarios:.2f}%) at max_num_objects={args.max_num_objects}.",
        )
    else:
        print(
            f"\nVERDICT: the SDC survives the slice in all {n_scenarios} scenarios "
            f"(max index {int(idx.max()) if idx.size else 'n/a'}). "
            "Claim 3's dropped-ego half does not fire; the nearest-vs-first "
            "neighbour-selection half may still.",
        )


if __name__ == "__main__":
    main()
