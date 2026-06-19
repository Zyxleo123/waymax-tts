"""V-Max RL integration for the diffusion-refinement loop.

This package wires Valeo's `V-Max <https://github.com/valeoai/v-max>`_ JAX-native
RL/IL algorithms (specifically the **BC_SAC** hybrid: behavior-cloning warm-up +
SAC) into this repository so we can:

1. **BC warm-up** a driving policy on the *expert* WOMD data that are **not**
   failure cases (these stand in for "data we already handle"); and
2. **SAC RL** that same policy on the harvested **failure cases** (the scenarios
   where the base diffusion planner failed), using reward only -- we pretend we
   have no expert demonstrations for them; and
3. export clean policy rollouts on the failure cases to **fine-tune the diffusion
   planner**.

The heavy lifting (networks, losses, replay buffer, pmap training loop) is reused
verbatim from the vendored ``V-Max/`` checkout. This package only adds:

* :mod:`rl.vmax_rl.data` -- data generators that restrict V-Max training to the
  failure set (RL) and the non-failure expert set (imitation);
* :mod:`rl.vmax_rl.bc_sac_dual` -- a BC_SAC trainer that draws imitation and RL
  batches from *two different* data sources;
* :mod:`rl.vmax_rl.refine` -- the CLI training entrypoint;
* :mod:`rl.vmax_rl.evaluate` -- clean-success evaluation + diffusion-rollout
  export.
"""

from rl.vmax_rl import compat  # noqa: F401  (installs JAX>=0.7 compat shims on import)

__all__ = ["compat"]
