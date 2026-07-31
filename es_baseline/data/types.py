from __future__ import annotations

from typing import Any

import jax
import torch
from flax import struct
from dataclasses import dataclass




@struct.dataclass
class PreprocessConfig:
    model_dt: float = 0.2
    world_dt_fallback: float = 0.1
    max_range: float = 30.0
    ego_range: float = 100.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_tl_points: int = 16
    num_map_type_classes: int = 21
    predict_horizon: int = 25
    map_unknown_type_index: int = 20
    ema_decay: float = 0.999
    max_segments: int = 128
    max_points_per_segment: int = 128
    num_object_types: int = 8
    inst_dim: int = 768


@struct.dataclass
class PreprocessBatch:
    features: dict[str, jax.Array]
    aux: dict[str, jax.Array]


@struct.dataclass
class CacheLoaderConfig:
    cache_paths: tuple[str, ...]
    annotation_dir: str | None = None
    preprocess_cfg: PreprocessConfig = PreprocessConfig()
    batch_size: int = 1
    shuffle_seed: int = 42
    anchor_steps: tuple[int, ...] = (0, 10, 20, 30, 40)
    language_label: str | None = "qa" # (qa, instruction, None)
    backend: str = "torch" # (torch, jax)
    include_inst_features: bool = False
    drop_last: bool = False
    

@struct.dataclass
class CacheBatch:
    features: dict[str, torch.Tensor | jax.Array]
    aux: dict[str, torch.Tensor | jax.Array]
    prompts: list[str] = None
    answers: list[str] = None
    keys: list[str] = None

