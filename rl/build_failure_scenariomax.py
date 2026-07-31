#!/usr/bin/env python3
"""Extract a ScenarioMax tfrecord containing only the failure-case scenes.

Copies the raw serialized ``tf.Example`` bytes for each failure's
``scenariomax_record_index`` (from ``rl/trace_failure_to_womd.py --check-scenariomax``
output) out of the full ScenarioMax training tfrecord, byte-for-byte. No
re-serialization, so the result is layout-identical to ``path_dataset`` and can be fed
straight into V-Max's ``simulator.make_data_generator`` -- usable as an overfit
training set and/or a joint-validation eval set against any repro_sac_v2* checkpoint.

Uses ``rl/tfrecord_fast.py``'s byte-offset sidecar index (``<tfrecord>.offsets.npy``)
for O(1) random-access reads -- the source tfrecord is ~930GB / 400k+ records, so a
plain sequential ``TFRecordDataset`` scan up to the highest wanted index is not
viable. Build the offset index once if missing::

    python -m rl.tfrecord_fast build /zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord

Some raw failures never made it into ScenarioMax at all: its tfexample converter
(``ScenarioMax/scenariomax/unified_to_tfexample/converter/roadgraph.py:_detect_overpass``)
hard-excludes any scene where two road-edge points sit within 0.8m in XY but >4m
apart in Z (its roadgraph format has no notion of road level, so it can't represent
stacked/overpass geometry). This script also classifies and documents those via
``--excluded-manifest`` (investigated 2026-07-09: every excluded case checked out as a
genuine multi-level structure -- highway interchange, spiral ramp, or mountain-road
switchback-over-switchback -- not a false positive from map noise, so none are
force-converted; see ``SPECIAL_NOTES`` below for the specific cases that got a closer
look).

Usage::

    python rl/build_failure_scenariomax.py \\
        --trace-json /zfsauton/scratch/yixiz/failure_womd_trace.json \\
        --output /zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord \\
        --excluded-manifest /zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/excluded_overpass.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_SCENARIOMAX_DIR = os.path.join(os.path.dirname(_RL_DIR), "ScenarioMax")
for _p in (_RL_DIR, _SCENARIOMAX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tfrecord_fast import offset_index_exists, read_tfrecord_bytes  # noqa: E402

DEFAULT_TRACE_JSON = "/zfsauton/scratch/yixiz/failure_womd_trace.json"
DEFAULT_OUTPUT = "/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord"
DEFAULT_EXCLUDED_MANIFEST = "/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/excluded_overpass.json"

# Cases given a closer look during the 2026-07-09 investigation, keyed by tf2_scenario_id
# (stable across trace reruns, unlike json_path). Everything else gets DEFAULT_NOTE.
SPECIAL_NOTES = {
    "be6a02b25aa8697c": (  # 00004-of-01000.scenario_029
        "Investigated as a possible false positive (isolated stray segment in a top-down "
        "plot). The offending pair is two points WITHIN the same road_edge feature (136 "
        "pts), spanning z=14.4-28.4m at a nearly fixed (x,y) -- i.e. the boundary line loops "
        "back near its own start at much greater height, consistent with a spiral/helix "
        "ramp. The identical feature (same bbox/z-range) also appears in scenario_218 "
        "(different WOMD scenario clip, different shard) at the same real-world location, "
        "which rules out one-off map digitization noise: a real, persistent structure "
        "reproduces across independent extracts; noise would not. Treated as genuine -- "
        "excluded."
    ),
    "a5ef0762f85d139e": (  # 00004-of-01000.scenario_218
        "Same real-world location and identical road_edge feature geometry as "
        "scenario_029 (00004 shard, tf2_scenario_id be6a02b25aa8697c) -- see that entry. "
        "Corroborates a genuine persistent spiral/helix ramp rather than a one-off "
        "artifact. Excluded."
    ),
    "d84a652fc06c2b37": (  # 00002-of-01000.scenario_252
        "Investigated as a possible false positive (winding road, gradual color gradient "
        "in plot, no clean 'flat road under flat road' crossing pattern). Global road-edge "
        "elevation span is 96.9m across the scene -- consistent with a mountain/hillside "
        "road with switchbacks. The flagged pair is a climbing feature (104.4-113.0m) "
        "closely overlapping in XY with a flatter lower feature (104.3-105.9m), exactly "
        "what a switchback-over-switchback crossing looks like on a steep mountain road: a "
        "real, physically-stacked road relationship, just not a purpose-built bridge. "
        "Flattening would still misplace the ego's elevation/lane. Excluded."
    ),
}

DEFAULT_NOTE = (
    "Road-edge geometry shows a clean 'flat road at one elevation crossing a flat road at "
    "another elevation' pattern typical of a highway interchange / overpass. ScenarioMax's "
    "roadgraph format has no notion of road level/layer, so it cannot represent two stacked "
    "road surfaces at the same (x,y) -- flattening would risk assigning the ego to the "
    "wrong level for offroad/collision checks. Excluded rather than guessed at."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-json", default=DEFAULT_TRACE_JSON)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument(
        "--excluded-manifest",
        default=DEFAULT_EXCLUDED_MANIFEST,
        help="Where to document scenariomax_hit=False failures with classification "
        "evidence. Pass '' to skip.",
    )
    return p.parse_args()


def _write_clean_tfrecord(trace: dict, output: str) -> int:
    records = [
        r for r in trace["records"] if r.get("scenariomax_hit") and r.get("error") is None
    ]
    if not records:
        raise RuntimeError(
            "No scenariomax-mapped failures in trace json. Run "
            "trace_failure_to_womd.py --check-scenariomax first."
        )

    by_source: dict[str, set[int]] = {}
    for r in records:
        by_source.setdefault(r["scenariomax_tfrecord_path"], set()).add(
            int(r["scenariomax_record_index"])
        )

    for src_path in by_source:
        if not offset_index_exists(src_path):
            raise RuntimeError(
                f"No byte-offset index for {src_path}. Build it once with:\n"
                f"  python -m rl.tfrecord_fast build {src_path}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # Write to a temp path and atomically replace `output` at the end. A rebuild that
    # writes `output` in place (TFRecordWriter truncates on open) will corrupt reads for
    # any training job already streaming from that path -- os.replace() swaps the
    # directory entry in one step instead of mutating the bytes a live reader sees.
    tmp_output = f"{output}.tmp.{os.getpid()}"
    written = 0
    with tf.io.TFRecordWriter(tmp_output) as writer:
        for src_path, wanted in by_source.items():
            print(f"Reading {len(wanted)} records from {src_path} via offset index ...")
            for idx in sorted(wanted):
                writer.write(read_tfrecord_bytes(src_path, idx))
                written += 1
    os.replace(tmp_output, output)

    print(f"Wrote {written}/{len(records)} failure scenes -> {output}")
    if written != len(records):
        print("WARNING: some indices were not found.")
    return written


def _road_edge_z_spans(scenario) -> tuple[float | None, float | None]:
    """(global z span, max single-feature z span) across all road_edge features."""
    spans = []
    all_z = []
    for mf in scenario.map_features:
        if mf.HasField("road_edge"):
            zs = np.array([p.z for p in mf.road_edge.polyline])
            if len(zs):
                spans.append(float(zs.max() - zs.min()))
                all_z.append(zs)
    if not all_z:
        return None, None
    combined = np.concatenate(all_z)
    return float(combined.max() - combined.min()), max(spans)


def _write_excluded_manifest(trace: dict, clean_count: int, manifest_path: str) -> None:
    import scenariomax.raw_to_unified.datasets.waymo.waymo_protos.scenario_pb2 as scenario_pb2

    missing = [r for r in trace["records"] if r.get("scenariomax_hit") is False]
    entries = []
    for r in missing:
        path, idx = r["raw_tfrecord_path"], r["raw_record_index"]
        raw = next(iter(tf.data.TFRecordDataset([path]).skip(idx).take(1))).numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw)
        global_span, max_single_span = _road_edge_z_spans(scenario)

        entries.append(
            {
                "failure_json": r["json_path"],
                "tf2_scenario_id": r["tf2_scenario_id"],
                "raw_shard": r.get("raw_shard"),
                "raw_record_index": r.get("raw_record_index"),
                "raw_tfrecord_path": r.get("raw_tfrecord_path"),
                "exception": "OverpassException",
                "road_edge_global_z_span_m": round(global_span, 2) if global_span is not None else None,
                "road_edge_max_single_feature_z_span_m": (
                    round(max_single_span, 2) if max_single_span is not None else None
                ),
                "classification": "genuine_multilevel_structure",
                "note": SPECIAL_NOTES.get(r["tf2_scenario_id"], DEFAULT_NOTE),
            }
        )

    manifest = {
        "description": (
            "Failure-case scenarios excluded from ScenarioMaxWaymoFailures/failures.tfrecord "
            "because ScenarioMax's tfexample converter raises OverpassException on them "
            "(scenariomax/unified_to_tfexample/converter/roadgraph.py:_detect_overpass). "
            "These scenes are absent from the full ScenarioMaxWaymo/training.tfrecord too "
            "-- the filter runs on the whole corpus, not just the failure subset -- so no "
            "repro_sac_v2* checkpoint has ever seen them in ScenarioMax layout either. "
            "Investigated 2026-07-09 for false positives (map noise, mountain roads) that "
            "could safely be force-converted; none were found -- see per-entry notes. No "
            "overrides applied."
        ),
        "clean_tfrecord": os.path.abspath(DEFAULT_OUTPUT),
        "clean_count": clean_count,
        "total_raw_failures": len(trace["records"]),
        "excluded_count": len(entries),
        "excluded": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    tmp_manifest = f"{manifest_path}.tmp.{os.getpid()}"
    json.dump(manifest, open(tmp_manifest, "w"), indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"Wrote {manifest_path} ({len(entries)} excluded entries documented)")


def main() -> int:
    args = _parse_args()
    trace = json.load(open(args.trace_json))

    written = _write_clean_tfrecord(trace, args.output)

    if args.excluded_manifest:
        _write_excluded_manifest(trace, written, args.excluded_manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
