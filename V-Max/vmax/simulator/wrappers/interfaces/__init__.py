# Copyright 2025 Valeo.

"""Interface wrappers for different environment standards."""

from .brax import AutoResetWrapper, BraxWrapper, EnvTransition, VmapWrapper
from .multi_agent import MultiAgentBraxWrapper


# gymnasium is only needed by ``make_gym_env``. Importing it eagerly makes the whole
# simulator unimportable in environments that do not ship it (the Diffusion-ES stack
# runs V-Max's env in-process for online SAC initialization and has no gymnasium).
try:
    from .gym import GymWrapper
except ImportError:  # pragma: no cover - depends on the environment
    GymWrapper = None


__all__ = [
    "AutoResetWrapper",
    "BraxWrapper",
    "EnvTransition",
    "GymWrapper",
    "MultiAgentBraxWrapper",
    "VmapWrapper",
]
