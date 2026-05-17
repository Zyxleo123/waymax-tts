from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
@dataclass
class PlannerResult:
    start_t_b: jax.Array
    trajectory_norm_btd: jax.Array
    trajectory_world_bt5: jax.Array
    world_t_seconds_bt: jax.Array
    world_t_valid_bt: jax.Array
    aux: dict[str, jax.Array]

class AbstractPlanner:
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        pass

    def plan_trajectory(
        self,
        **kwargs: Any,
    ) -> PlannerResult:
        pass