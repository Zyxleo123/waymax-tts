#!/usr/bin/env python3
"""Fast random access into large TFRecord files via byte-offset sidecar indexes.

``TFRecordDataset.skip(N)`` is O(N) and impractical for ScenarioMax's monolithic
``training.tfrecord`` (400k+ records, ~900GB). Build a ``.offsets.npy`` once, then
read any record in O(1).

Example::

    python -m rl.tfrecord_fast build \\
        /zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord

    python -m rl.tfrecord_fast read \\
        /zfsauton/scratch/yixiz/ScenarioMaxWaymo/training.tfrecord 231012
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_RECORD_HEADER = struct.Struct("<Q")
_RECORD_LEN_CRC = struct.Struct("<I")
_RECORD_DATA_CRC = struct.Struct("<I")
_RECORD_META_SIZE = _RECORD_HEADER.size + _RECORD_LEN_CRC.size + _RECORD_DATA_CRC.size


def default_offset_index_path(tfrecord_path: str) -> str:
    return f"{tfrecord_path}.offsets.npy"


def offset_index_exists(tfrecord_path: str, offset_index: str | None = None) -> bool:
    path = offset_index or default_offset_index_path(tfrecord_path)
    return os.path.isfile(path)


def load_offset_index(tfrecord_path: str, offset_index: str | None = None) -> np.ndarray:
    path = offset_index or default_offset_index_path(tfrecord_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"TFRecord offset index not found: {path}\n"
            f"Build it once with:\n"
            f"  python -m rl.tfrecord_fast build {tfrecord_path}"
        )
    offsets = np.load(path, mmap_mode="r")
    if offsets.ndim != 1 or offsets.dtype != np.int64:
        raise ValueError(f"Invalid offset index {path}: expected 1-D int64 array.")
    return offsets


def build_offset_index(
    tfrecord_path: str,
    output_path: str | None = None,
    *,
    progress_every: int = 50000,
) -> np.ndarray:
    """Scan ``tfrecord_path`` once and write byte offsets for each record."""
    tfrecord_path = os.path.abspath(tfrecord_path)
    output_path = output_path or default_offset_index_path(tfrecord_path)
    file_size = os.path.getsize(tfrecord_path)
    offsets: list[int] = []
    t0 = time.time()
    with open(tfrecord_path, "rb") as f:
        while True:
            pos = f.tell()
            if pos >= file_size:
                break
            header = f.read(_RECORD_HEADER.size)
            if len(header) < _RECORD_HEADER.size:
                break
            (length,) = _RECORD_HEADER.unpack(header)
            if length <= 0 or f.tell() + length + _RECORD_DATA_CRC.size > file_size:
                raise ValueError(
                    f"Corrupt TFRecord at byte {pos} in {tfrecord_path}: length={length}"
                )
            f.seek(_RECORD_LEN_CRC.size + length + _RECORD_DATA_CRC.size, os.SEEK_CUR)
            offsets.append(pos)
            n = len(offsets)
            if progress_every and n % progress_every == 0:
                pct = 100.0 * pos / file_size
                rate = pos / max(time.time() - t0, 1e-6) / 1e6
                print(
                    f"  indexed {n} records  {pct:.1f}%  {rate:.0f} MB/s",
                    flush=True,
                )
    arr = np.asarray(offsets, dtype=np.int64)
    tmp = f"{output_path}.tmp.{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, output_path)
    elapsed = time.time() - t0
    print(
        f"Wrote {len(arr)} offsets -> {output_path}  "
        f"({file_size / 1e9:.2f} GB in {elapsed / 60:.1f} min)"
    )
    return arr


def read_tfrecord_bytes(
    tfrecord_path: str,
    record_index: int,
    *,
    offset_index: str | None = None,
) -> bytes:
    """Read one serialized TFExample without ``TFRecordDataset.skip``."""
    if record_index < 0:
        raise ValueError(f"record_index must be >= 0, got {record_index}")
    offsets = load_offset_index(tfrecord_path, offset_index)
    if record_index >= len(offsets):
        raise IndexError(
            f"record_index {record_index} out of range for {tfrecord_path} "
            f"(index has {len(offsets)} records)."
        )
    with open(tfrecord_path, "rb") as f:
        f.seek(int(offsets[record_index]))
        header = f.read(_RECORD_HEADER.size)
        (length,) = _RECORD_HEADER.unpack(header)
        f.read(_RECORD_LEN_CRC.size)
        data = f.read(length)
        if len(data) != length:
            raise IOError(f"Short read at record {record_index} in {tfrecord_path}")
        f.read(_RECORD_DATA_CRC.size)
    return data


def read_tfrecord_bytes_or_scan(
    tfrecord_path: str,
    record_index: int,
    *,
    offset_index: str | None = None,
    slow_threshold: int = 1000,
) -> bytes:
    """Use offset index when available; otherwise scan (only for small indices)."""
    path = offset_index or default_offset_index_path(tfrecord_path)
    if os.path.isfile(path):
        return read_tfrecord_bytes(tfrecord_path, record_index, offset_index=path)
    if record_index >= slow_threshold:
        raise RuntimeError(
            f"Cannot load record {record_index} from {tfrecord_path} without a byte "
            f"offset index (would require scanning {record_index} records).\n"
            f"Build the index once (takes ~1h on the monolithic ScenarioMax file):\n"
            f"  python -m rl.tfrecord_fast build {tfrecord_path}\n"
            f"Or visualize trajectory only:\n"
            f"  python -m rl.visualize_scenario ... --scenariomax --no-map --mode static"
        )
    import tensorflow as tf

    return next(iter(tf.data.TFRecordDataset([tfrecord_path]).skip(record_index).take(1))).numpy()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build .offsets.npy for a TFRecord file.")
    b.add_argument("tfrecord")
    b.add_argument("--output", default=None)

    r = sub.add_parser("read", help="Read one record and print byte length.")
    r.add_argument("tfrecord")
    r.add_argument("record_index", type=int)
    r.add_argument("--offset-index", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "build":
        build_offset_index(args.tfrecord, args.output)
        return 0
    if args.cmd == "read":
        raw = read_tfrecord_bytes(args.tfrecord, args.record_index, offset_index=args.offset_index)
        print(len(raw))
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
