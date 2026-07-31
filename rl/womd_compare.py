"""Utilities to compare local WOMD tf_example data vs ScenarioMax-converted tfexample.

Local RL code uses Waymo's pre-built ``tf_example/training/training_tfexample.*``
(WOD 1.3.1: 45×800 SDC paths). ScenarioMax converts ``scenario/training/training.*``
protobuf TFRecords into V-Max-style tfexample (10×300 paths, SDC at object slot 0).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Iterable

import numpy as np
import tensorflow as tf

# Waymax / WOMD 1.3.1 official tfexample layout (``waymax_rs`` default).
LOCAL_NUM_PATHS = 45
LOCAL_NUM_POINTS_PER_PATH = 800

# ScenarioMax + V-Max training layout.
SCENARIOMAX_NUM_PATHS = 10
SCENARIOMAX_NUM_POINTS_PER_PATH = 300

DEFAULT_LOCAL_TFEXAMPLE = (
    "/zfsauton/scratch/eshau/womd/tf_example/training/"
    "training_tfexample.tfrecord-00000-of-01000"
)
DEFAULT_RAW_SCENARIO = (
    "/zfsauton/scratch/eshau/womd/scenario/training/training.tfrecord-00000-of-01000"
)
DEFAULT_SCENARIOMAX_SMOKE = "/zfsauton/scratch/yixiz/ScenarioMaxWaymo_smoke/training.tfrecord"


def _read_varint(data: bytes, i: int) -> tuple[int | None, int | None]:
    result = 0
    shift = 0
    while i < len(data):
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 63:
            return None, None
    return None, None


def scenario_id_from_scenario_bytes(data: bytes) -> str:
    """Extract ``scenario_id`` (Waymo Scenario proto field 5) without importing pb2."""
    i = 0
    n = len(data)
    while i < n:
        tag, i = _read_varint(data, i)
        if i is None:
            break
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            _, i = _read_varint(data, i)
            if i is None:
                break
        elif wire_type == 1:
            i += 8
        elif wire_type == 2:
            length, i = _read_varint(data, i)
            if i is None or i + length > n:
                break
            chunk = data[i : i + length]
            i += length
            if field_num == 5:
                return chunk.decode("utf-8")
        elif wire_type == 5:
            i += 4
        else:
            break
    raise ValueError("scenario_id (field 5) not found in Scenario protobuf")


def iter_scenario_ids(tfrecord_path: str, *, limit: int | None = None) -> list[str]:
    """Return ``scenario/id`` strings in TFRecord order."""
    ids: list[str] = []
    for i, raw in enumerate(tf.data.TFRecordDataset(tfrecord_path)):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        sid = example.features.feature["scenario/id"].bytes_list.value[0].decode()
        ids.append(sid)
        if limit is not None and i + 1 >= limit:
            break
    return ids


def path_xyz_float_count(example: tf.train.Example) -> int:
    feature = example.features.feature["path_samples/xyz"].float_list.value
    return len(feature)


def load_example(tfrecord_path: str, index: int) -> tf.train.Example:
    raw = next(iter(tf.data.TFRecordDataset([tfrecord_path]).skip(index).take(1)))
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    return example


def scenario_id_at(tfrecord_path: str, index: int) -> str:
    return iter_scenario_ids(tfrecord_path, limit=index + 1)[index]


def iter_raw_scenario_ids(tfrecord_path: str, *, limit: int | None = None) -> list[str]:
    """Load ``scenario_id`` from raw WOMD scenario protobuf TFRecords."""
    ids: list[str] = []
    for i, data in enumerate(tf.data.TFRecordDataset(tfrecord_path)):
        ids.append(scenario_id_from_scenario_bytes(data.numpy()))
        if limit is not None and i + 1 >= limit:
            break
    return ids


def load_waymax_state(
    tfrecord_path: str,
    index: int,
    *,
    num_paths: int,
    num_points: int,
    max_num_objects: int = 128,
    batch_dims: tuple[int, ...] = (1,),
):
    """Load one ``SimulatorState`` via Waymax preprocessing.

    Preprocess a single TFExample first, then add batch dims. Feeding a
    batched serialized example breaks ``max_num_objects`` truncation (Waymax
    slices axis 0, which is batch when examples are pre-batched).
    """
    import jax.numpy as jnp
    import jax.tree_util as jtu
    import tensorflow as tf
    from waymax import config as waymax_config
    from waymax import dataloader
    from waymax.dataloader import womd_factories

    cfg = waymax_config.DatasetConfig(
        path=tfrecord_path,
        max_num_objects=int(max_num_objects),
        num_paths=num_paths,
        num_points_per_path=num_points,
        include_sdc_paths=True,
        batch_dims=batch_dims,
    )
    from rl.tfrecord_fast import read_tfrecord_bytes_or_scan

    raw = read_tfrecord_bytes_or_scan(tfrecord_path, int(index))
    processed = dataloader.preprocess_serialized_womd_data(tf.constant(raw), cfg)
    processed_np = jtu.tree_map(lambda t: t.numpy(), processed)
    if batch_dims:
        processed_np = jtu.tree_map(
            lambda x: jnp.asarray(x)[(None,) * len(batch_dims)],
            processed_np,
        )
    else:
        processed_np = jtu.tree_map(jnp.asarray, processed_np)
    return womd_factories.simulator_state_from_womd_dict(
        processed_np, include_sdc_paths=cfg.include_sdc_paths
    )


def sdc_object_index(state, *, timestep: int = 10) -> int:
    valid = np.asarray(state.log_trajectory.valid[0, :, timestep])
    is_sdc = np.asarray(state.object_metadata.is_sdc[0])
    mask = valid.astype(bool) & is_sdc.astype(bool)
    if not mask.any():
        raise ValueError("no valid SDC at requested timestep")
    return int(np.argmax(mask))


def sdc_xy_slice(state, *, timestep: int = 10, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    idx = sdc_object_index(state, timestep=timestep)
    t0 = timestep
    t1 = min(timestep + steps, state.log_trajectory.x.shape[-1])
    x = np.asarray(state.log_trajectory.x[0, idx, t0:t1])
    y = np.asarray(state.log_trajectory.y[0, idx, t0:t1])
    return x, y


def id_overlap(a: Iterable[str], b: Iterable[str]) -> tuple[int, int, int]:
    sa, sb = set(a), set(b)
    return len(sa & sb), len(sa), len(sb)


def sorted_raw_training_files(raw_dir: str) -> list[str]:
    files = [
        f
        for f in os.listdir(raw_dir)
        if f.startswith("training.tfrecord-") and "tfrecord" in f
    ]
    return sorted(files)
