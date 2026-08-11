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
        sigma_sample_floor: float = 0.1,
        sigma_logprob_floor: float = 0.1,
        actor_lr: float = 3e-5,
        grad_clip_norm: float = 1.0,
        ppo_clip: float = 0.1,
        pg_scale: float | None = None,
        seed: int = 0,
    ):
        self.diffusion = diffusion
        self.k_trainable = int(k_trainable)
        self.prefix_len = prefix_len
        self.sigma_sample_floor = float(sigma_sample_floor)
        self.sigma_logprob_floor = float(sigma_logprob_floor)
        # PPO requires the behavior policy (what `sample_with_trace` draws from)
        # and the density the ratio is computed under to be the *same* Gaussian.
        # A smaller sample floor than log-prob floor makes every recorded action
        # off-policy w.r.t. the density being optimized, biasing the gradient with
        # no importance correction. Enforce equality rather than silently mixing.
        if abs(self.sigma_sample_floor - self.sigma_logprob_floor) > 1e-9:
            raise ValueError(
                "sigma_sample_floor and sigma_logprob_floor must be equal "
                f"(got {self.sigma_sample_floor} vs {self.sigma_logprob_floor}); "
                "sampling and log-prob must describe the same policy."
            )
        self.ppo_clip = float(ppo_clip)
        # `None` = auto: undo the 1/(channels*prefix_len) averaging that
        # `_reduce_logdensity` applies, so the policy-gradient term is comparable
        # in magnitude to the (unscaled) expert diffusion loss. Resolved on the
        # first `ppo_update` once the trace shape is known.
        self.pg_scale = None if pg_scale is None else float(pg_scale)
        # The number of scalar log-densities `_reduce_logdensity` averages over
        # (channels * executed prefix). Recorded so the reported per-transition
        # KL/ratio can be recovered from the averaged quantity that PPO clips on.
        # Resolved alongside `pg_scale` on the first `ppo_update`.
        self._reduce_n: float | None = None
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
        # Behavior params = the params `collect` last sampled under. The KL gate
        # measures the candidate policy against *this* snapshot (the policy that
        # generated the data), not against the drifting current params. Refreshed
        # in `collect`; seeded here so a `ppo_update` before any `collect` is safe.
        self._behavior_tparams = self._tparams

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
        # Freeze the behavior policy for this data batch: the KL gate compares the
        # PPO candidate against these params, so it must be the params that draw
        # the trajectory. Collection within an iteration never updates params, so
        # every replan step in the iteration shares one snapshot.
        self._behavior_tparams = self._tparams
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
    def act(self, cond1, cond2, cond2_mask, *, rng, deterministic: bool = False):
        """Sample a trajectory for *evaluation* only (no trace, no training).

        ``deterministic=True`` runs the standard (non-RL) sampler over the
        trainable late steps -- the trajectory the fine-tuned policy would emit at
        deployment, with no exploration-floor inflation. ``deterministic=False``
        draws from the same stochastic policy PPO optimizes. Both are reproducible
        for a fixed ``rng``.
        """
        shape = (cond1.shape[0], self.diffusion.denoise_fn.in_ch, self.diffusion.denoise_fn.horizon)
        traj, _ = self.diffusion.sample_with_trace(
            shape, cond1, cond2, cond2_mask, rng=rng,
            k_trainable=self.k_trainable,
            trainable_denoise_fn=self._trainable_fn(self._tparams),
            rl_mode=not deterministic,
            sigma_sample_floor=self.sigma_sample_floor,
            sigma_logprob_floor=self.sigma_logprob_floor,
            prefix_len=self.prefix_len,
        )
        return traj

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
    def _expert_loss(self, params, expert, alive_b):
        """Ordinary diffusion loss of the trainable denoiser on expert targets.

        ``expert`` is ``(target_btd, cond1, cond2, cond2_mask, rng)`` -- the
        normalized expert (log) trajectory and its frozen condition. Anchors the
        trainable copy to the demonstration (DPPO + expert anchor). Averaged over
        the *alive* samples only, so padded post-terminal replans do not tug the
        anchor toward a stale expert target."""
        target_btd, ec1, ec2, emask, erng = expert
        fn = self._trainable_fn(params)
        x0 = jnp.transpose(target_btd, (0, 2, 1))  # [B,T,D] -> [B,D,T]
        lpe = self.diffusion.loss_per_example(x0, ec1, ec2, emask, rng=erng, denoise_fn=fn)  # [B]
        return jnp.sum(lpe * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

    @staticmethod
    def _wmean_kb(x_kb, w_b):
        """Mean of ``[K, B]`` over the alive samples (weight ``[B]``, broadcast over K)."""
        w = w_b[None, :]
        return jnp.sum(x_kb * w) / jnp.maximum(jnp.sum(w) * x_kb.shape[0], 1.0)

    def _ppo_loss(self, params, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
                  expert=None, expert_weight=0.0):
        new_lp = self.log_probs(params, cond1, cond2, cond2_mask, trace)  # [K, B]
        log_ratio = new_lp - old_log_probs_kb                             # [K, B]
        ratio = jnp.exp(log_ratio)                                        # [K, B]
        adv = adv_b[None, :]                                              # [1, B]
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
        surrogate = jnp.minimum(unclipped, clipped)
        # All reductions are over the *alive* samples: post-terminal padded replans
        # carry adv==0 but still have a ratio, so an unweighted mean would dilute
        # the PG term and mis-report KL/clip_frac. `_wmean_kb` divides by the alive
        # count, not the padded n*B.
        # `_reduce_logdensity` averages the per-element log density over
        # channels*prefix_len, which keeps `ratio` near 1 (intended) but shrinks
        # the PG gradient by that same factor. `pg_scale` restores it so the PG
        # term is not swamped by the expert anchor; the clip still acts on the
        # un-scaled ratio, so the trust region is unchanged.
        pg_raw = -self._wmean_kb(surrogate, alive_b)
        pg_loss = self.pg_scale * pg_raw
        # Schulman's nonnegative KL estimator `ratio - 1 - log(ratio)` -- always
        # >= 0 and lower-variance than the raw `old - new` difference. Reported for
        # diagnostics; the *gate* uses the exact Gaussian KL in `ppo_update`.
        approx_kl = self._wmean_kb(ratio - 1.0 - log_ratio, alive_b)
        # `ratio` above is on the *averaged* log density (what the clip sees). The
        # per-transition ratio compounds over all `reduce_n` coordinates, so report
        # the trust region as the policy actually moves it: log_ratio * reduce_n.
        lr_t = log_ratio * self._reduce_n
        ratio_t = jnp.exp(lr_t)
        approx_kl_transition = self._wmean_kb(ratio_t - 1.0 - lr_t, alive_b)
        # `expert` is a Python-level None-or-tuple at trace time, so a plain branch
        # is correct (and keeps the expert forward out of the graph when unused).
        if expert is not None:
            expert_loss = self._expert_loss(params, expert, alive_b)
        else:
            expert_loss = jnp.asarray(0.0, dtype=jnp.float32)
        expert_term = expert_weight * expert_loss
        loss = pg_loss + expert_term
        clip_frac = self._wmean_kb((jnp.abs(ratio - 1.0) > self.ppo_clip).astype(jnp.float32), alive_b)
        return loss, {"loss": loss, "pg_loss": pg_loss, "pg_raw": pg_raw,
                      "expert_loss": expert_loss, "expert_term": expert_term,
                      "approx_kl": approx_kl, "approx_kl_transition": approx_kl_transition,
                      "mean_ratio": self._wmean_kb(ratio, alive_b),
                      "mean_ratio_transition": self._wmean_kb(ratio_t, alive_b),
                      "clip_frac": clip_frac}

    # ------------------------------------------------------------------ #
    def _posterior_means_std(self, params, cond1, cond2, cond2_mask, trace):
        """Per-transition posterior mean and std, both ``[K, B, C, T]``.

        The std is the (param-independent) schedule posterior std; the mean is the
        denoiser-dependent posterior mean. Used by the exact Gaussian KL gate.
        """
        fn = self._trainable_fn(params)
        k = int(trace["timestep"].shape[0])
        means, stds = [], []
        for j in range(k):
            mean, std, _ = self.diffusion._posterior_mean_std(
                trace["x_t"][j], trace["timestep"][j], cond1, cond2, cond2_mask, denoise_fn=fn)
            means.append(mean)
            stds.append(std)
        return jnp.stack(means, axis=0), jnp.stack(stds, axis=0)

    def _transition_kl(self, mu_old, mu_new, std, alive_b):
        """Exact KL(old || candidate) of the reverse-diffusion policy, in nats.

        Both policies are diagonal Gaussians with the *same* (param-independent)
        std, so the KL is ``0.5 * sum((mu_old - mu_new)/sigma)^2`` over the executed
        prefix coordinates -- no ``exp`` of a large log-ratio, so it cannot overflow
        or go negative. Summed over the ``K`` late steps (the trajectory KL
        factorizes over the reverse chain) and averaged over the alive samples.
        """
        std_lp = jnp.maximum(std, self.sigma_logprob_floor)          # [K, B, C, T]
        per_elt = 0.5 * ((mu_new - mu_old) / std_lp) ** 2            # [K, B, C, T]
        t_dim = per_elt.shape[3]
        if self.prefix_len is None:
            kl_kb = jnp.sum(per_elt, axis=(2, 3))                    # [K, B]
        else:
            mt = (jnp.arange(t_dim) < int(self.prefix_len)).astype(per_elt.dtype)
            kl_kb = jnp.sum(per_elt * mt[None, None, None, :], axis=(2, 3))
        kl_b = jnp.sum(kl_kb, axis=0)                                # [B], trajectory KL
        return jnp.sum(kl_b * alive_b) / jnp.maximum(jnp.sum(alive_b), 1.0)

    def ppo_update(self, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
                   *, expert=None, expert_weight=0.0, target_kl=None):
        """One clipped-PPO gradient step on the trainable denoiser only.

        With ``expert`` given, adds ``expert_weight`` * (expert diffusion loss) --
        the DPPO + expert-anchor variant.

        The trust-region gate is on the *candidate*: grads are formed at the current
        params, a candidate params + optimizer state are constructed, and the exact
        Gaussian KL of the candidate policy against the behavior snapshot
        (``_behavior_tparams``, frozen in ``collect``) is measured. If it exceeds
        ``target_kl`` the candidate is discarded and ``metrics["applied"] == 0.0``,
        so the step that would cross the trust region is never committed -- unlike a
        pre-update KL, which is ~0 at epoch 0 and only rejects the *next* step.
        ``metrics["kl_gauss"]`` is the realized post-update KL (0 when discarded).
        """
        if self.pg_scale is None or self._reduce_n is None:
            # trace["x_t"] is [K, B, C, T]; the log density was averaged over
            # C and the executed prefix (or all of T when prefix_len is None).
            _, _, c, t_dim = trace["x_t"].shape
            pl = t_dim if self.prefix_len is None else int(self.prefix_len)
            self._reduce_n = float(int(c) * pl)
            if self.pg_scale is None:
                self.pg_scale = self._reduce_n
        (loss, metrics), grads = jax.value_and_grad(self._ppo_loss, has_aux=True)(
            self._tparams, cond1, cond2, cond2_mask, trace, old_log_probs_kb, adv_b, alive_b,
            expert, expert_weight,
        )
        metrics = dict(metrics)
        # Build the candidate (params + optimizer state) but do not commit yet.
        cand_updates, cand_opt_state = self.tx.update(grads, self._opt_state, self._tparams)
        cand_params = optax.apply_updates(self._tparams, cand_updates)
        # Exact KL of the candidate against the behavior policy (the data-generating
        # snapshot), not against the current params.
        mu_old, std = self._posterior_means_std(self._behavior_tparams, cond1, cond2, cond2_mask, trace)
        mu_cand, _ = self._posterior_means_std(cand_params, cond1, cond2, cond2_mask, trace)
        kl_gauss = self._transition_kl(mu_old, mu_cand, std, alive_b)
        applied = target_kl is None or float(np.asarray(kl_gauss)) <= float(target_kl)
        if applied:
            self._opt_state = cand_opt_state
            self._tparams = cand_params
        metrics["grad_global_norm"] = optax.global_norm(grads)
        metrics["kl_gauss"] = kl_gauss if applied else jnp.asarray(0.0, dtype=jnp.float32)
        metrics["kl_gauss_candidate"] = kl_gauss
        metrics["applied"] = jnp.asarray(1.0 if applied else 0.0, dtype=jnp.float32)
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
