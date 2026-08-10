"""Unified simulator-reward fine-tuning framework for the pretrained diffusion policy.

This package is the RL/search trainer for the diffusion motion policy. It is
deliberately *separate* from ``train_diffusion/train_diffusion_policy.py`` (which
only updates instruction/subgoal/FiLM parameters): the modules here treat one
sampled trajectory as the outer action and optimize against the *simulator*
reward.

Layout (Phase 1 -- the trustworthy environment):

* ``checkpoint``  -- load the pretrained architecture from checkpoint *metadata*
  (not config defaults), exposing both raw and EMA parameters.
* ``reward``      -- the single batched simulator-reward function every method
  consumes.
* ``env``         -- :class:`~rl.diffusion_ft.env.DiffusionWaymaxEnv`, a batched
  environment whose outer action is one sampled diffusion trajectory.
"""

from __future__ import annotations

__all__ = [
    "checkpoint",
    "reward",
    "env",
]
