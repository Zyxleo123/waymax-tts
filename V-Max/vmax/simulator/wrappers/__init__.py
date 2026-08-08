# Copyright 2025 Valeo.

"""Module for wrappers that enhance the behavior of environments."""

# Base wrapper
from .base import Wrapper

# Interface wrappers
from .interfaces.brax import AutoResetWrapper, BraxWrapper, EnvTransition, VmapWrapper
from .interfaces.multi_agent import MultiAgentBraxWrapper

# Optional: see vmax/simulator/wrappers/interfaces/__init__.py.
try:
    from .interfaces.gym import GymWrapper
except ImportError:  # pragma: no cover - depends on the environment
    GymWrapper = None

# Observation wrappers
from .observation import ObservationWrapper

# Reward wrappers
from .reward import RewardCustomWrapper, RewardLinearWrapper

# State modification wrappers
from .state.noisy_init import NoisyInitWrapper
from .state.sdc_path import SDCPathWrapper


__all__ = [
    # Interfaces
    "AutoResetWrapper",
    "BraxWrapper",
    "EnvTransition",
    "GymWrapper",
    "MultiAgentBraxWrapper",
    # State
    "NoisyInitWrapper",
    # Observation
    "ObservationWrapper",
    # Reward
    "RewardCustomWrapper",
    "RewardLinearWrapper",
    "SDCPathWrapper",
    "VmapWrapper",
    # Base
    "Wrapper",
]
