#!/usr/bin/env python3
"""Trace RL failure JSONs (#2 tfexample) back to WOMD scenario protobuf (#1).

Each failure file stores ``tfrecord`` + ``scenario_idx`` — a **record position inside
one tfexample shard**, not a global scenario number.  This script:

1. Reads ``scenario/id`` from the #2 tfexample record referenced by each failure.
2. Looks up that id in the scenario sidecar SQLite (if present).
3. Finds the matching record in #1 scenario protobuf shards (cached index).
4. Reports whether #1 and #2 agree at the same (shard, record_index).

Run on a login / IO node with TensorFlow installed, e.g.::

    conda activate waymax
    python rl/trace_failure_to_womd.py \\
        --failure-dir /zfsauton/scratch/mineuih/waymax_rs/failure_samples \\
        --output /zfsauton/scratch/yixiz/failure_womd_trace.json

Build (or refresh) the #1 id index once — slow, ~486k records / 1000 shards::

    python rl/trace_failure_to_womd.py --build-index-only

Typical runtime for tracing 245 failures with a warm index: a few minutes.

Quick smoke test (first 10 failures, one pass over #1 shards, no index file)::

    python rl/trace_failure_to_womd.py --scan-without-index --limit 10
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import tensorflow as tf

_RL_DIR = Path(__file__).resolve().parent
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))
from womd_compare import scenario_id_from_scenario_bytes

# Defaults match eshau scratch layout used by waymax_rs.
DEFAULT_FAILURE_DIR = "/zfsauton/scratch/mineuih/waymax_rs/failure_samples"
DEFAULT_TFEXAMPLE_DIR = "/zfsauton/scratch/eshau/womd/tf_example/training"
DEFAULT_SCENARIO_DIR = "/zfsauton/scratch/eshau/womd/scenario/training"
DEFAULT_SIDECAR_DB = (
    "/zfsauton/scratch/eshau/womd/scenario_sidecar/training/training.scenario_store.sqlite3"
)
DEFAULT_INDEX_CACHE = "/zfsauton/scratch/yixiz/womd_scenario_id_index.sqlite3"
DEFAULT_SCENARIOMAX_DIR = "/zfsauton/scratch/yixiz/ScenarioMaxWaymo"
DEFAULT_SCENARIOMAX_TFRECORD = (
    "/zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord"
)
DEFAULT_SCENARIOMAX_INDEX = "/zfsauton/scratch/yixiz/scenariomax_scenario_id_index.sqlite3"

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


@dataclasses.dataclass
class FailureRecord:
    json_path: str
    tfrecord: str
    scenario_idx: int
    tf2_scenario_id: str | None = None
    raw_scenario_id_same_slot: str | None = None
    raw_shard: int | None = None
    raw_record_index: int | None = None
    raw_tfrecord_path: str | None = None
    sidecar_hit: bool | None = None
    same_slot_match: bool | None = None
    indexed_match: bool | None = None
    scenariomax_hit: bool | None = None
    scenariomax_tfrecord_path: str | None = None
    scenariomax_record_index: int | None = None
    error: str | None = None


def _load_raw_scenario_id(tfrecord_path: str, record_index: int) -> str:
    raw = next(iter(tf.data.TFRecordDataset([tfrecord_path]).skip(record_index).take(1)))
    return scenario_id_from_scenario_bytes(raw.numpy())


def _shard_from_tfrecord_path(path: str) -> int | None:
    m = SHARD_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def _raw_tfrecord_path(scenario_dir: str, shard: int, num_shards: int = 1000) -> str:
    return os.path.join(
        scenario_dir,
        f"training.tfrecord-{shard:05d}-of-{num_shards:05d}",
    )


def _iter_failure_jsons(failure_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(failure_dir, "*.json")))
    return [p for p in paths if not p.endswith(".instructions.json")]


def _load_tf2_scenario_id(tfrecord_path: str, scenario_idx: int) -> str:
    raw = next(iter(tf.data.TFRecordDataset([tfrecord_path]).skip(scenario_idx).take(1)))
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    return example.features.feature["scenario/id"].bytes_list.value[0].decode()


def _open_index_db(index_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scenario_index (
            scenario_id TEXT PRIMARY KEY,
            shard INTEGER NOT NULL,
            record_index INTEGER NOT NULL,
            tfrecord_path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scenario_index_shard ON scenario_index(shard)"
    )
    conn.commit()
    return conn


def _index_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM scenario_index").fetchone()
    return int(row[0]) if row else 0


def build_scenario_index(
    scenario_dir: str,
    index_path: str,
    *,
    shard_limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """Scan #1 protobuf shards and cache scenario_id -> (shard, record_index)."""
    if overwrite and os.path.exists(index_path):
        os.remove(index_path)

    conn = _open_index_db(index_path)
    if not overwrite and _index_count(conn) > 0:
        count = _index_count(conn)
        print(f"Index already has {count} ids at {index_path}; use --overwrite-index to rebuild.")
        conn.close()
        return count

    conn.execute("DELETE FROM scenario_index")
    conn.commit()

    shard_files = sorted(
        f
        for f in os.listdir(scenario_dir)
        if f.startswith("training.tfrecord-") and f.endswith(f"-of-01000")
    )
    if shard_limit is not None:
        shard_files = shard_files[:shard_limit]

    total = 0
    t0 = time.time()
    for file_name in shard_files:
        m = SHARD_RE.search(file_name)
        if not m:
            continue
        shard = int(m.group(1))
        path = os.path.join(scenario_dir, file_name)
        batch: list[tuple[str, int, int, str]] = []
        for record_index, data in enumerate(tf.data.TFRecordDataset(path)):
            sid = scenario_id_from_scenario_bytes(data.numpy())
            batch.append((sid, shard, record_index, path))
            if len(batch) >= 500:
                conn.executemany(
                    "INSERT OR REPLACE INTO scenario_index VALUES (?,?,?,?)",
                    batch,
                )
                conn.commit()
                total += len(batch)
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO scenario_index VALUES (?,?,?,?)",
                batch,
            )
            conn.commit()
            total += len(batch)
        if shard % 50 == 0:
            elapsed = time.time() - t0
            print(f"  indexed shard {shard:05d}  total_ids={total}  elapsed={elapsed:.0f}s", flush=True)

    conn.close()
    print(f"Built index with {total} scenario ids -> {index_path}")
    return total


def lookup_index(conn: sqlite3.Connection, scenario_id: str) -> tuple[int, int, str] | None:
    row = conn.execute(
        "SELECT shard, record_index, tfrecord_path FROM scenario_index WHERE scenario_id=?",
        (scenario_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1]), str(row[2])


def lookup_sidecar(sidecar_db: str, scenario_id: str) -> bool:
    if not os.path.isfile(sidecar_db):
        return False
    conn = sqlite3.connect(sidecar_db)
    row = conn.execute(
        "SELECT 1 FROM payloads WHERE scenario_id=? LIMIT 1",
        (scenario_id,),
    ).fetchone()
    conn.close()
    return row is not None


def _scenariomax_tfrecord_paths(scenariomax_dir: str) -> list[str]:
    if not os.path.isdir(scenariomax_dir):
        return []
    patterns = [
        os.path.join(scenariomax_dir, "training.tfrecord"),
        os.path.join(scenariomax_dir, "waymo", "waymo_*", "training.tfrecord"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    return paths


def build_scenariomax_id_set(scenariomax_dir: str) -> set[str]:
    """Load all scenario/id values from ScenarioMax output (one linear scan)."""
    ids: set[str] = set()
    for path in _scenariomax_tfrecord_paths(scenariomax_dir):
        print(f"  scanning ScenarioMax ids in {path} ...", flush=True)
        for raw in tf.data.TFRecordDataset(path):
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            ids.add(example.features.feature["scenario/id"].bytes_list.value[0].decode())
    return ids


def _open_scenariomax_index_db(index_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(index_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scenariomax_index (
            scenario_id TEXT PRIMARY KEY,
            record_index INTEGER NOT NULL,
            tfrecord_path TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _scenariomax_index_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM scenariomax_index").fetchone()
    return int(row[0]) if row else 0


def build_scenariomax_index(
    tfrecord_paths: list[str],
    index_path: str,
    *,
    overwrite: bool = False,
) -> int:
    """Scan ScenarioMax tfexample(s) and cache scenario_id -> record_index."""
    if not tfrecord_paths:
        raise RuntimeError("No ScenarioMax tfrecord paths to index.")

    if overwrite and os.path.exists(index_path):
        os.remove(index_path)

    conn = _open_scenariomax_index_db(index_path)
    if not overwrite and _scenariomax_index_count(conn) > 0:
        count = _scenariomax_index_count(conn)
        print(
            f"ScenarioMax index already has {count} ids at {index_path}; "
            "use --overwrite-scenariomax-index to rebuild."
        )
        conn.close()
        return count

    conn.execute("DELETE FROM scenariomax_index")
    conn.commit()

    total = 0
    t0 = time.time()
    for path in tfrecord_paths:
        print(f"  indexing ScenarioMax tfrecord {path} ...", flush=True)
        batch: list[tuple[str, int, str]] = []
        for record_index, raw in enumerate(tf.data.TFRecordDataset(path)):
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            sid = example.features.feature["scenario/id"].bytes_list.value[0].decode()
            batch.append((sid, record_index, path))
            if len(batch) >= 500:
                conn.executemany(
                    "INSERT OR REPLACE INTO scenariomax_index VALUES (?,?,?)",
                    batch,
                )
                conn.commit()
                total += len(batch)
                batch.clear()
            if record_index and record_index % 50000 == 0:
                print(
                    f"    record {record_index}  total_ids={total + len(batch)}  "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO scenariomax_index VALUES (?,?,?)",
                batch,
            )
            conn.commit()
            total += len(batch)

    conn.close()
    print(f"Built ScenarioMax index with {total} scenario ids -> {index_path}")
    return total


def lookup_scenariomax_index(
    conn: sqlite3.Connection, scenario_id: str
) -> tuple[int, str] | None:
    row = conn.execute(
        "SELECT record_index, tfrecord_path FROM scenariomax_index WHERE scenario_id=?",
        (scenario_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def scan_scenario_ids_in_raw(
    scenario_dir: str,
    target_ids: set[str],
    *,
    shard_limit: int | None = None,
) -> dict[str, tuple[int, int, str]]:
    """One linear pass over #1 shards; stop early when all ``target_ids`` are found."""
    shard_files = sorted(
        f
        for f in os.listdir(scenario_dir)
        if f.startswith("training.tfrecord-") and f.endswith("-of-01000")
    )
    if shard_limit is not None:
        shard_files = shard_files[:shard_limit]

    found: dict[str, tuple[int, int, str]] = {}
    remaining = set(target_ids)
    t0 = time.time()
    for file_name in shard_files:
        m = SHARD_RE.search(file_name)
        if not m:
            continue
        shard = int(m.group(1))
        path = os.path.join(scenario_dir, file_name)
        for record_index, data in enumerate(tf.data.TFRecordDataset(path)):
            sid = scenario_id_from_scenario_bytes(data.numpy())
            if sid in remaining:
                found[sid] = (shard, record_index, path)
                remaining.remove(sid)
                if not remaining:
                    print(
                        f"  found all {len(target_ids)} ids after shard {shard:05d} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
                    return found
        if shard % 50 == 0:
            print(
                f"  scanned through shard {shard:05d}  "
                f"found {len(found)}/{len(target_ids)}  elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    print(
        f"  scan done: found {len(found)}/{len(target_ids)}  "
        f"shards={len(shard_files)}  elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    return found


def trace_failures_scan(
    failure_dir: str,
    scenario_dir: str,
    *,
    sidecar_db: str | None = DEFAULT_SIDECAR_DB,
    shard_limit: int | None = None,
    limit: int | None = None,
) -> list[FailureRecord]:
    """Trace failures by scanning #1 shards once (no sqlite index required)."""
    failure_paths = _iter_failure_jsons(failure_dir)
    if limit is not None:
        failure_paths = failure_paths[:limit]

    # Load #2 ids first so we can scan #1 in a single pass.
    pending: list[FailureRecord] = []
    target_ids: set[str] = set()
    for json_path in failure_paths:
        rec = FailureRecord(json_path=json_path, tfrecord="", scenario_idx=-1)
        try:
            payload = json.load(open(json_path))
            rec.tfrecord = str(payload["tfrecord"])
            rec.scenario_idx = int(payload["scenario_idx"])
            rec.tf2_scenario_id = _load_tf2_scenario_id(rec.tfrecord, rec.scenario_idx)
            target_ids.add(rec.tf2_scenario_id)

            shard = _shard_from_tfrecord_path(rec.tfrecord)
            if shard is not None:
                raw_path = _raw_tfrecord_path(scenario_dir, shard)
                if os.path.isfile(raw_path):
                    rec.raw_scenario_id_same_slot = _load_raw_scenario_id(
                        raw_path, rec.scenario_idx
                    )
                    rec.same_slot_match = rec.raw_scenario_id_same_slot == rec.tf2_scenario_id

            if sidecar_db:
                rec.sidecar_hit = lookup_sidecar(sidecar_db, rec.tf2_scenario_id)
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
        pending.append(rec)

    print(f"Scanning #1 for {len(target_ids)} scenario ids ...", flush=True)
    located = scan_scenario_ids_in_raw(
        scenario_dir, target_ids, shard_limit=shard_limit
    )

    for rec in pending:
        if rec.error or not rec.tf2_scenario_id:
            continue
        hit = located.get(rec.tf2_scenario_id)
        if hit is not None:
            rec.raw_shard, rec.raw_record_index, rec.raw_tfrecord_path = hit
            rec.indexed_match = True
        else:
            rec.indexed_match = False

    return pending


def trace_failures(
    failure_dir: str,
    scenario_dir: str,
    index_path: str,
    *,
    sidecar_db: str | None = DEFAULT_SIDECAR_DB,
    scenariomax_conn: sqlite3.Connection | None = None,
    limit: int | None = None,
) -> list[FailureRecord]:
    index_conn = _open_index_db(index_path)
    if _index_count(index_conn) == 0:
        index_conn.close()
        raise RuntimeError(
            f"Index {index_path} is empty. Run with --build-index-only first."
        )

    results: list[FailureRecord] = []
    failure_paths = _iter_failure_jsons(failure_dir)
    if limit is not None:
        failure_paths = failure_paths[:limit]

    for json_path in failure_paths:
        rec = FailureRecord(json_path=json_path, tfrecord="", scenario_idx=-1)
        try:
            payload = json.load(open(json_path))
            rec.tfrecord = str(payload["tfrecord"])
            rec.scenario_idx = int(payload["scenario_idx"])

            rec.tf2_scenario_id = _load_tf2_scenario_id(rec.tfrecord, rec.scenario_idx)

            shard = _shard_from_tfrecord_path(rec.tfrecord)
            if shard is not None:
                raw_path = _raw_tfrecord_path(scenario_dir, shard)
                if os.path.isfile(raw_path):
                    rec.raw_tfrecord_path = raw_path
                    rec.raw_scenario_id_same_slot = _load_raw_scenario_id(
                        raw_path, rec.scenario_idx
                    )
                    rec.same_slot_match = (
                        rec.raw_scenario_id_same_slot == rec.tf2_scenario_id
                    )

            if sidecar_db:
                rec.sidecar_hit = lookup_sidecar(sidecar_db, rec.tf2_scenario_id)

            hit = lookup_index(index_conn, rec.tf2_scenario_id)
            if hit is not None:
                rec.raw_shard, rec.raw_record_index, rec.raw_tfrecord_path = hit
                rec.indexed_match = True
            else:
                rec.indexed_match = False

            if scenariomax_conn is not None:
                smx_hit = lookup_scenariomax_index(scenariomax_conn, rec.tf2_scenario_id)
                if smx_hit is not None:
                    rec.scenariomax_record_index, rec.scenariomax_tfrecord_path = smx_hit
                    rec.scenariomax_hit = True
                else:
                    rec.scenariomax_hit = False
        except Exception as exc:  # noqa: BLE001 — collect per-file errors for report
            rec.error = f"{type(exc).__name__}: {exc}"
        results.append(rec)

    index_conn.close()
    return results


def _write_csv(rows: list[FailureRecord], csv_path: str) -> None:
    fieldnames = [
        "json_path",
        "tfrecord",
        "scenario_idx",
        "tf2_scenario_id",
        "raw_scenario_id_same_slot",
        "same_slot_match",
        "sidecar_hit",
        "indexed_match",
        "raw_shard",
        "raw_record_index",
        "raw_tfrecord_path",
        "scenariomax_hit",
        "scenariomax_tfrecord_path",
        "scenariomax_record_index",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in rows:
            writer.writerow({k: getattr(rec, k) for k in fieldnames})


def _summarize(rows: list[FailureRecord]) -> dict:
    ok = [r for r in rows if r.error is None]
    return {
        "total_failures": len(rows),
        "errors": sum(1 for r in rows if r.error),
        "with_tf2_scenario_id": sum(1 for r in ok if r.tf2_scenario_id),
        "same_slot_match": sum(1 for r in ok if r.same_slot_match is True),
        "same_slot_mismatch": sum(1 for r in ok if r.same_slot_match is False),
        "sidecar_hit": sum(1 for r in ok if r.sidecar_hit),
        "indexed_in_scenario_dir": sum(1 for r in ok if r.indexed_match),
        "indexed_missing_in_scenario_dir": sum(1 for r in ok if r.indexed_match is False),
        "scenariomax_hit": sum(1 for r in ok if r.scenariomax_hit),
        "scenariomax_missing": sum(1 for r in ok if r.scenariomax_hit is False),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failure-dir", default=DEFAULT_FAILURE_DIR)
    p.add_argument("--tfexample-dir", default=DEFAULT_TFEXAMPLE_DIR)
    p.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR)
    p.add_argument("--sidecar-db", default=DEFAULT_SIDECAR_DB)
    p.add_argument("--no-sidecar", action="store_true")
    p.add_argument("--index-cache", default=DEFAULT_INDEX_CACHE)
    p.add_argument("--scenariomax-dir", default=DEFAULT_SCENARIOMAX_DIR)
    p.add_argument(
        "--scenariomax-tfrecord",
        default=DEFAULT_SCENARIOMAX_TFRECORD,
        help="ScenarioMax tfexample used for pseudo-GT rollouts.",
    )
    p.add_argument("--scenariomax-index", default=DEFAULT_SCENARIOMAX_INDEX)
    p.add_argument("--build-scenariomax-index-only", action="store_true")
    p.add_argument("--overwrite-scenariomax-index", action="store_true")
    p.add_argument(
        "--check-scenariomax",
        action="store_true",
        help="Look up each failure's scenario id in the ScenarioMax sqlite index.",
    )
    p.add_argument("--build-index-only", action="store_true")
    p.add_argument(
        "--scan-without-index",
        action="store_true",
        help="Scan #1 shards directly (no sqlite index). Use with --limit for smoke tests.",
    )
    p.add_argument("--overwrite-index", action="store_true")
    p.add_argument(
        "--shard-limit",
        type=int,
        default=None,
        help="Max #1 shards to scan (index build or --scan-without-index).",
    )
    p.add_argument("--limit", type=int, default=None, help="Trace only first N failures.")
    p.add_argument(
        "--output",
        default="failure_womd_trace.json",
        help="JSON summary + per-failure records.",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path (defaults to <output>.csv).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sidecar_db = None if args.no_sidecar else args.sidecar_db

    if args.build_index_only:
        build_scenario_index(
            args.scenario_dir,
            args.index_cache,
            shard_limit=args.shard_limit,
            overwrite=args.overwrite_index,
        )
        return 0

    if args.build_scenariomax_index_only:
        paths = _scenariomax_tfrecord_paths(args.scenariomax_dir)
        if not paths and os.path.isfile(args.scenariomax_tfrecord):
            paths = [args.scenariomax_tfrecord]
        build_scenariomax_index(
            paths,
            args.scenariomax_index,
            overwrite=args.overwrite_scenariomax_index,
        )
        return 0

    scenariomax_conn: sqlite3.Connection | None = None
    if args.check_scenariomax:
        if not os.path.isfile(args.scenariomax_index):
            paths = _scenariomax_tfrecord_paths(args.scenariomax_dir)
            if not paths and os.path.isfile(args.scenariomax_tfrecord):
                paths = [args.scenariomax_tfrecord]
            print(f"Building ScenarioMax index at {args.scenariomax_index} ...")
            build_scenariomax_index(paths, args.scenariomax_index)
        scenariomax_conn = _open_scenariomax_index_db(args.scenariomax_index)
        print(f"  ScenarioMax indexed ids: {_scenariomax_index_count(scenariomax_conn)}")

    print("Tracing failures...")
    print(f"  failure_dir   = {args.failure_dir}")
    print(f"  scenario_dir  = {args.scenario_dir}")
    if args.scan_without_index:
        print(f"  mode          = scan-without-index (shard_limit={args.shard_limit})")
        print(f"  sidecar_db    = {sidecar_db or '(disabled)'}")
        rows = trace_failures_scan(
            args.failure_dir,
            args.scenario_dir,
            sidecar_db=sidecar_db,
            shard_limit=args.shard_limit,
            limit=args.limit,
        )
    else:
        print(f"  index_cache   = {args.index_cache}")
        print(f"  sidecar_db    = {sidecar_db or '(disabled)'}")
        rows = trace_failures(
            args.failure_dir,
            args.scenario_dir,
            args.index_cache,
            sidecar_db=sidecar_db,
            scenariomax_conn=scenariomax_conn,
            limit=args.limit,
        )

    if scenariomax_conn is not None:
        scenariomax_conn.close()

    summary = _summarize(rows)
    out = {
        "summary": summary,
        "records": [dataclasses.asdict(r) for r in rows],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    csv_path = args.csv or (args.output.rsplit(".", 1)[0] + ".csv")
    _write_csv(rows, csv_path)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {args.output}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
