from __future__ import annotations

from typing import Any

import jax
from flax import struct


@struct.dataclass
class PreprocessConfig:
    model_dt: float = 0.2
    world_dt_fallback: float = 0.1
    max_range: float = 30.0
    ego_range: float = 100.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_map_points: int = 1024
    max_tl_points: int = 16
    num_map_type_classes: int = 21
    predict_horizon: int = 25
    map_unknown_type_index: int = 20
    ema_decay: float = 0.999


@struct.dataclass
class PreprocessBatch:
    features: dict[str, jax.Array]
    aux: dict[str, jax.Array]