"""DPPO actor: frozen base denoiser + trainable late-step denoiser copy.

Phase 2, item 4. The pretrained diffusion policy is split so that only the last
``k_trainable`` denoising steps are learnable:

* the **scene tokenizer and all instruction/subgoal modules are frozen** -- their
  output (the condition tensors ``cond1``/``cond2``/``cond2_mask``) is computed
  once by the frozen policy and passed in as arrays, so no gradient can reach
  them;
* the **frozen base denoiser** drives the early (high-noise) denoising steps and
  never updates;
* a **trainable copy** of the denoiser drives the final ``k_trainable`` steps and
  is the only thing PPO optimizes.

The actor exposes:

* :meth:`collect` -- run :meth:`GaussianDiffusion.sample_with_trace` (frozen early,
  trainable late, RL sampling mode) to produce a trajectory and the recorded
  transitions with old-policy log-probs;
* :meth:`ppo_update` -- one clipped-PPO gradient step on the trainable denoiser
  only, from stored transitions + advantages;
* :meth:`state_dict` / :meth:`load_state_dict` -- checkpoint just the trainable
  params, optimizer, and rng, so a run resumes bit-for-bit.

Gradient isolation is structural: the loss is differentiated w.r.t. the trainable
denoiser's ``nnx.Param`` state alone; everything else is captured as a constant.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def _clone_module(module: nnx.Module) -> nnx.Module:
    """Deep-copy an nnx module (independent parameter arrays)."""
    gdef, state = nnx.split(module)
    state_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), state)
    return nnx.merge(gdef, state_copy)


class DiffusionRLActor:
    """Wraps a :class:`GaussianDiffusion` for DPPO fine-tuning of late steps."""

    def __init__(
        self,
        diffusion,
        *,
        k_trainable: int = 10,
        prefix_len: int | None = None,
        sigma_sample_floor: float = 0.05,
        sigma_logprob_floor: float = 0.1,
        actor_lr: float = 1e-5,
        grad_clip_norm: float = 1.0,
        ppo_clip: float = 0.01,
        seed: int = 0,
    ):
        self.diffusion = diffusion
        self.k_trainable = int(k_trainable)
        self.prefix_len = prefix_len
        self.sigma_sample_floor = float(sigma_sample_floor)
        self.sigma_logprob_floor = float(sigma_logprob_floor)
        self.ppo_clip = float(ppo_clip)
        self._rng = jax.random.PRNGKey(int(seed))

        # Frozen base denoiser = the diffusion's own denoise_fn (never updated).
        self.frozen_denoise = diffusion.denoise_fn
        # Trainable copy drives the last k_trainable steps.
        self.trainable_denoise = _clone_module(self.frozen_denoise)

        # Split the trainable denoiser into (graphdef, params, non-params). Only
        # `params` is optimized; everything else is a constant to the loss.
        self._tgdef, self._tparams, self._tnonparam = nnx.split(
            self.trainable_denoise, nnx.Param, ...
        )

        self.tx = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            optax.adam(float(actor_lr)),
        )
        self._opt_state = self.tx.init(self._tparams)

    # ------------------------------------------------------------------ #
    def _trainable_fn(self, params):
        """Reconstitute the trainable denoiser from a params pytree."""
        return nnx.merge(self._tgdef, params, self._tnonparam)

    # ------------------------------------------------------------------ #
    def collect(self, cond1, cond2, cond2_mask, *, rng=None):
        """Sample a trajectory + record the last k_trainable transitions.

        Uses the frozen denoiser for early steps and the current trainable copy
        for late steps, in RL sampling mode. Returns ``(trajectory, trace)``.
        """
        if rng is None:
            self._rng, rng = jax.random.split(self._rng)
        shape = (cond1.shape[0], self.diffusion.denoise_fn.in_ch, self.diffusion.denoise_fn.horizon)
        traj, trace = self.diffusion.sample_with_trace(
            shape, cond1, cond2, cond2_mask, rng=rng,
            k_trainable=self.k_trainable,
            trainable_denoise_fn=self._trainable_fn(self._tparams),
            rl_mode=True,
            sigma_sample_floor=self.sigma_sample_floor,
            sigma_logprob_floor=self.sigma_logprob_floor,
            prefix_len=self.prefix_len,
        )
        return traj, trace

    # ------------------------------------------------------------------ #
    def log_probs(self, params, cond1, cond2, cond2_mask, trace):
        """Per-step transition log-probs ``[K, B]`` under a given params pytree."""
        fn = self._trainable_fn(params)
        k = int(trace["timestep"].shape[0])
        lps = []
        for j in range(k):
            lp = self.diffusion.transition_log_prob(
                trace["x_t"][j], trace["x_prev"][j], trace["timestep"][j],
                cond1, cond2, cond2_mask,
                sigma_floor=self.sigma_logprob_floor, prefix_len=self.prefix_len,
                denoise_fn=fn,
            )
            lps.append(lp)
        return jnp.stack(lps, axis=0)

    # ------------------------------------------------------------------ #
    def _expert_loss(self, params, expert):
        """Ordinary diffusion loss of the trainable denoiser on expert targets.

        ``expert`` is ``(target_btd, cond1, cond2, cond2_mask, rng)`` -- the
        normalized expert (log) trajectory and its frozen condition. Anchors the
        trainable copy to the demonstration (DPPO + expert anchor)."""
        target_btd, ec1, ec2, emask, erng = expert
        fn = self._trainable_fn(params)
        x0 = jnp.transpose(target_btd, (0, 2, 1))  # [B,T,D] -> [B,D,T]
        return jnp.mean(self.diffusion.loss_per_example(x0, ec1, ec2, emask, rng=erng, denoise_fn=fn))

    def _ppo_loss(self, params, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b,
                  expert=None, expert_weight=0.0):
        new_lp = self.log_probs(params, cond1, cond2, cond2_mask, trace)  # [K, B]
        ratio = jnp.exp(new_lp - old_log_probs_kb)                        # [K, B]
        adv = adv_b[None, :]                                              # [1, B]
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
        surrogate = jnp.minimum(unclipped, clipped)
        pg_loss = -jnp.mean(surrogate)
        approx_kl = jnp.mean(old_log_probs_kb - new_lp)
        # `expert` is a Python-level None-or-tuple at trace time, so a plain branch
        # is correct (and keeps the expert forward out of the graph when unused).
        if expert is not None:
            expert_loss = self._expert_loss(params, expert)
        else:
            expert_loss = jnp.asarray(0.0, dtype=jnp.float32)
        loss = pg_loss + expert_weight * expert_loss
        return loss, {"loss": loss, "pg_loss": pg_loss, "expert_loss": expert_loss,
                      "approx_kl": approx_kl, "mean_ratio": jnp.mean(ratio)}

    def ppo_update(self, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b,
                   *, expert=None, expert_weight=0.0):
        """One clipped-PPO gradient step on the trainable denoiser only.

        With ``expert`` given, adds ``expert_weight`` * (expert diffusion loss) --
        the DPPO + expert-anchor variant.
        """
        (loss, metrics), grads = jax.value_and_grad(self._ppo_loss, has_aux=True)(
            self._tparams, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b,
            expert, expert_weight,
        )
        updates, self._opt_state = self.tx.update(grads, self._opt_state, self._tparams)
        self._tparams = optax.apply_updates(self._tparams, updates)
        metrics = dict(metrics)
        metrics["grad_global_norm"] = optax.global_norm(grads)
        return metrics

    # ------------------------------------------------------------------ #
    def frozen_params_unchanged(self, reference_state) -> bool:
        """True iff the frozen base denoiser still equals ``reference_state``.

        Used to assert the update touched only the trainable copy.
        """
        _, cur = nnx.split(self.frozen_denoise)
        leaves_a = jax.tree_util.tree_leaves(cur)
        leaves_b = jax.tree_util.tree_leaves(reference_state)
        return all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves_a, leaves_b))

    def frozen_state(self):
        _, st = nnx.split(self.frozen_denoise)
        return jax.tree_util.tree_map(lambda x: jnp.array(x), st)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        return {
            "tparams": self._tparams,
            "opt_state": self._opt_state,
            "rng": self._rng,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._tparams = sd["tparams"]
        self._opt_state = sd["opt_state"]
        self._rng = sd["rng"]
