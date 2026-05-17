from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import tensorflow as tf
from waymax import config as waymax_config
from waymax import dataloader
from waymax.dataloader import womd_factories


def load_scenario_state_fast(cfg: waymax_config.DatasetConfig, scenario_index: int):
    """Loads one scenario directly from TFRecord by skipping raw records first."""
    raw_ds = tf.data.TFRecordDataset([cfg.path]).skip(int(scenario_index)).take(1)
    iterator = iter(raw_ds)
    try:
        serialized = next(iterator)
    except StopIteration as exc:
        raise ValueError(
            f"scenario_index {scenario_index} is out of range for tfrecord '{cfg.path}'."
        ) from exc

    serialized = tf.expand_dims(serialized, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized, cfg)
    return womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )


def load_scenario_state_batch_fast(
    cfg: waymax_config.DatasetConfig, scenario_indices: Iterable[int]
):
    """Loads a batch of scenarios from one TFRecord in a single dataset scan."""
    indices = [int(i) for i in scenario_indices]
    if not indices:
        raise ValueError("scenario_indices must be non-empty.")
    if any(i < 0 for i in indices):
        raise ValueError(f"scenario_indices must be >= 0, got {indices}.")

    unique_indices = sorted(set(indices))
    needed = set(unique_indices)
    selected: list[tf.Tensor] = []
    selected_indices: list[int] = []

    raw_ds = tf.data.TFRecordDataset([cfg.path])
    for raw_idx, serialized in enumerate(raw_ds):
        if raw_idx in needed:
            selected.append(serialized)
            selected_indices.append(raw_idx)
            if len(selected) == len(needed):
                break

    missing = sorted(needed.difference(selected_indices))
    if missing:
        raise ValueError(
            f"scenario_index values {missing} are out of range for tfrecord '{cfg.path}'."
        )

    serialized_batch = tf.stack(selected, axis=0)
    processed = dataloader.preprocess_serialized_womd_data(serialized_batch, cfg)
    state_batch = womd_factories.simulator_state_from_womd_dict(
        processed, include_sdc_paths=cfg.include_sdc_paths
    )
    scenario_to_batch_idx = {scenario_idx: i for i, scenario_idx in enumerate(selected_indices)}
    return state_batch, scenario_to_batch_idx
