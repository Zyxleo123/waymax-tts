#!/usr/bin/env python3
"""Build a ScenarioMax tfrecord shard for the ES oracle scenes (Path 2).

**No TensorFlow import** (TF CUDA init hung previous CPU jobs for hours).
Reads ``scenario/id`` from native WOMD tf_example shards via a raw TFRecord
scan, looks them up in the ScenarioMax id index, and byte-copies matching
records from the big ScenarioMax training tfrecord (``rl/tfrecord_fast``).

Writes under ``--out-dir`` (must be on scratch):

  * ``es_scenes.tfrecord``
  * ``es_scenes_manifest.json``

Example::

    python es_baseline/experiments/build_es_scenariomax.py \\
      --out-dir /zfsauton/scratch/yixiz/ScenarioMaxWaymoES
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RL = _REPO / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))
from tfrecord_fast import offset_index_exists, read_tfrecord_bytes  # noqa: E402

ES_SCENES = {
    "00000": [71, 92, 113, 134, 187, 252, 315, 377, 0, 1, 2, 3],
    "00001": [45, 62, 147, 158, 200, 243, 289, 360, 463, 0, 1, 2, 3],
    "00002": [0, 13, 20, 24, 70, 94, 97, 1, 2, 3, 4],
}

DEFAULT_REPRO = _REPO / "es_baseline" / "repro"
DEFAULT_SMX = "/zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord"
DEFAULT_SMX_INDEX = "/zfsauton/scratch/yixiz/scenariomax_scenario_id_index.sqlite3"
DEFAULT_OUT = "/zfsauton/scratch/yixiz/ScenarioMaxWaymoES"
SCRATCH_PREFIXES = ("/zfsauton/scratch/", "/scratch/")

_RECORD_HEADER = struct.Struct("<Q")
_RECORD_LEN_CRC = struct.Struct("<I")
_RECORD_DATA_CRC = struct.Struct("<I")
_HEX_ID = re.compile(rb"[0-9a-f]{16}")

# TFRecord wire format uses masked CRC-32C (Castagnoli), not zlib CRC-32.
try:
    from google_crc32c import value as _crc32c  # type: ignore
except ImportError:  # pragma: no cover
    def _crc32c_table() -> list[int]:
        poly = 0x82F63B78
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
            table.append(crc)
        return table

    _CRC32C_TABLE = _crc32c_table()

    def _crc32c(data: bytes) -> int:
        crc = 0xFFFFFFFF
        for b in data:
            crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
        return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _require_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    s = str(resolved)
    if not any(s.startswith(p) for p in SCRATCH_PREFIXES):
        raise SystemExit(
            f"Refusing to write large artifacts outside scratch: {resolved}\n"
            f"Pass --out-dir under /zfsauton/scratch/..."
        )
    return resolved


def _masked_crc(data: bytes) -> int:
    crc = int(_crc32c(data)) & 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _write_tfrecord(path: Path, records: list[bytes]) -> None:
    with open(path, "wb") as f:
        for data in records:
            length = len(data)
            f.write(_RECORD_HEADER.pack(length))
            f.write(_RECORD_LEN_CRC.pack(_masked_crc(_RECORD_HEADER.pack(length))))
            f.write(data)
            f.write(_RECORD_DATA_CRC.pack(_masked_crc(data)))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-dir", type=Path, default=DEFAULT_REPRO)
    p.add_argument("--scenariomax-tfrecord", default=DEFAULT_SMX)
    p.add_argument("--scenariomax-index", default=DEFAULT_SMX_INDEX)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _tfexample_path(repro_dir: Path, shard: str) -> Path:
    nested = repro_dir / f"tf{shard}" / "training" / f"training_tfexample.tfrecord-{shard}-of-01000"
    flat = repro_dir / f"tf{shard}" / f"training_tfexample.tfrecord-{shard}-of-01000"
    if nested.is_file() or nested.is_symlink():
        return nested.resolve()
    if flat.is_file() or flat.is_symlink():
        return flat.resolve()
    raise FileNotFoundError(f"missing tfrecord for shard {shard}: tried {nested} and {flat}")


def _scenario_id_from_example(raw: bytes) -> str | None:
    """Pull 16-hex WOMD scenario id near the feature key (no protobuf/TF)."""
    for key in (b"scenario/id", b"state/id"):
        idx = raw.find(key)
        if idx < 0:
            continue
        m = _HEX_ID.search(raw, idx, idx + 128)
        if m:
            return m.group(0).decode("ascii")
    return None


def _read_needed_records(path: Path, indices: set[int]) -> dict[int, bytes]:
    want = set(indices)
    max_i = max(want)
    out: dict[int, bytes] = {}
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        i = 0
        while i <= max_i:
            pos = f.tell()
            if pos >= file_size:
                break
            header = f.read(_RECORD_HEADER.size)
            if len(header) < _RECORD_HEADER.size:
                break
            (length,) = _RECORD_HEADER.unpack(header)
            if length <= 0:
                raise ValueError(f"bad length at record {i} byte {pos} in {path}")
            f.read(_RECORD_LEN_CRC.size)
            data = f.read(length)
            if len(data) < length:
                raise ValueError(f"truncated record {i} at byte {pos} in {path}")
            f.read(_RECORD_DATA_CRC.size)
            if i in want:
                out[i] = data
                if len(out) == len(want):
                    break
            i += 1
    missing = want - set(out)
    if missing:
        raise IndexError(f"{path}: missing record indices {sorted(missing)}")
    return out


def _lookup_smx(con: sqlite3.Connection, scenario_id: str) -> int | None:
    row = con.execute(
        "SELECT record_index FROM scenariomax_index WHERE scenario_id = ?",
        (scenario_id,),
    ).fetchone()
    return int(row[0]) if row else None


def main() -> None:
    args = _parse()
    out_dir = _require_scratch(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tf = out_dir / "es_scenes.tfrecord"
    out_man = out_dir / "es_scenes_manifest.json"

    if not offset_index_exists(args.scenariomax_tfrecord):
        raise SystemExit(
            f"Missing offsets for {args.scenariomax_tfrecord}. Build once with:\n"
            f"  python -m rl.tfrecord_fast build {args.scenariomax_tfrecord}"
        )

    con = sqlite3.connect(args.scenariomax_index)
    records = []
    wanted_smx: list[tuple[int, dict]] = []

    for shard, idxs in ES_SCENES.items():
        path = _tfexample_path(args.repro_dir, shard)
        want = set(idxs)
        print(f"scanning (raw) {path} for {sorted(want)}", flush=True)
        raw_by_idx = _read_needed_records(path, want)
        for i in sorted(want):
            sid = _scenario_id_from_example(raw_by_idx[i])
            smx_idx = _lookup_smx(con, sid) if sid else None
            row = {
                "tf_example_shard": shard,
                "tf_example_idx": int(i),
                "tfrecord": str(path),
                "scenario_id": sid,
                "scenariomax_hit": smx_idx is not None,
                "scenariomax_record_index": smx_idx,
                "scenariomax_tfrecord_path": args.scenariomax_tfrecord if smx_idx is not None else None,
            }
            records.append(row)
            if smx_idx is not None:
                wanted_smx.append((smx_idx, row))
            print(
                f"  {shard}/{i}: id={sid}  smx={smx_idx if smx_idx is not None else 'MISS'}",
                flush=True,
            )

    con.close()

    wanted_smx.sort(key=lambda x: x[0])
    seen: set[int] = set()
    uniq: list[tuple[int, dict]] = []
    for smx_idx, row in wanted_smx:
        if smx_idx in seen:
            continue
        seen.add(smx_idx)
        uniq.append((smx_idx, row))

    print(f"\nwriting {len(uniq)} ScenarioMax records -> {out_tf}", flush=True)
    payloads: list[bytes] = []
    for out_i, (smx_idx, row) in enumerate(uniq):
        payloads.append(read_tfrecord_bytes(args.scenariomax_tfrecord, smx_idx))
        row["es_scenes_record_index"] = out_i
    _write_tfrecord(out_tf, payloads)

    n_hit = sum(1 for r in records if r["scenariomax_hit"])
    n_miss = len(records) - n_hit
    manifest = {
        "summary": {
            "n_es_scenes": len(records),
            "scenariomax_hit": n_hit,
            "scenariomax_miss": n_miss,
            "unique_smx_records_written": len(uniq),
            "out_tfrecord": str(out_tf),
            "source_scenariomax": args.scenariomax_tfrecord,
        },
        "records": records,
    }
    out_man.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out_man}", flush=True)
    print(f"hit={n_hit}  miss={n_miss}  (misses are usually ScenarioMax overpass exclusions)", flush=True)


if __name__ == "__main__":
    main()
