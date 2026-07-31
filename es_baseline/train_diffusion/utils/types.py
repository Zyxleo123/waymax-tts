from __future__ import annotations

from typing import Any

import jax
import optax
from flax import struct


@struct.dataclass
class TrainState:
    # `model` uses Any to avoid import cycle with model.DiffusionPolicy.
    model: Any
    opt_state: optax.OptState
    ema_params: Any
    rng_key: jax.Array
    global_step: jax.Array
    epoch: jax.Array
