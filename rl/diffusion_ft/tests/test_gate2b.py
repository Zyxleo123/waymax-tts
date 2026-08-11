"""Gate 2 (RL-actor): frozen/trainable split, gradient isolation, PPO direction.

Tiny synthetic model (fast, CPU). Covers the remaining Gate 2 items:

* Only the trainable denoiser copy receives gradients (the frozen base denoiser is
  unchanged after an update; the trainable params move).
* A positive-advantage PPO update increases the sampled transitions' log-prob.
* Checkpoint/resume at an iteration boundary reproduces the next rollout.
* Sampling and log-prob use the same sigma floor (single-Gaussian policy).
* KL early-stop declines the step that would cross target_kl (no extra update).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned
from rl.diffusion_ft.rl_actor import DiffusionRLActor


def _tiny_actor(**kw):
    C, T, cond_dim, hidden = 5, 8, 16, 32
    rngs = nnx.Rngs(0)
    unet = UNet1DConditioned(
        in_ch=C, cond_dim=cond_dim, cond2_dim=cond_dim, time_emb_dim=cond_dim,
        horizon=T, hidden_dim=hidden, base_ch=16, rngs=rngs,
    )
    diff = GaussianDiffusion(unet, timesteps=10, predict_type="v")
    B = 3
    cond1 = jax.random.normal(jax.random.PRNGKey(1), (B, 4, cond_dim))
    cond2 = jax.random.normal(jax.random.PRNGKey(2), (B, cond_dim))
    mask = jnp.ones((B,), dtype=jnp.float32)
    actor = DiffusionRLActor(diff, k_trainable=4, seed=0, **kw)
    return actor, cond1, cond2, mask


def _close(a, b, atol=1e-5):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol))


def gate_gradient_isolation() -> bool:
    actor, c1, c2, m = _tiny_actor(actor_lr=1e-3, ppo_clip=0.5)
    frozen_before = actor.frozen_state()
    tparams_before = jax.tree_util.tree_map(lambda x: jnp.array(x), actor._tparams)

    traj, trace = actor.collect(c1, c2, m, rng=jax.random.PRNGKey(100))
    old_lp = trace["log_prob"]
    adv = jnp.ones((c1.shape[0],))
    metrics = actor.ppo_update(c1, c2, m, trace, old_lp, adv)

    frozen_unchanged = actor.frozen_params_unchanged(frozen_before)
    tparams_moved = not all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(tparams_before),
                        jax.tree_util.tree_leaves(actor._tparams))
    )
    gnorm = float(np.asarray(metrics["grad_global_norm"]))
    print(f"  frozen base unchanged={frozen_unchanged}  trainable moved={tparams_moved}  grad_norm={gnorm:.4e}")
    return bool(frozen_unchanged and tparams_moved and gnorm > 0)


def gate_positive_advantage_increases_prob() -> bool:
    actor, c1, c2, m = _tiny_actor(actor_lr=2e-3, ppo_clip=0.5)
    traj, trace = actor.collect(c1, c2, m, rng=jax.random.PRNGKey(101))
    old_lp = trace["log_prob"]                       # [K, B]
    adv = jnp.ones((c1.shape[0],))                   # positive advantage
    before = float(jnp.sum(actor.log_probs(actor._tparams, c1, c2, m, trace)))
    for _ in range(5):
        actor.ppo_update(c1, c2, m, trace, old_lp, adv)
    after = float(jnp.sum(actor.log_probs(actor._tparams, c1, c2, m, trace)))
    print(f"  sum log_prob before={before:.4f} after={after:.4f} increased={after > before}")
    return after > before


def gate_checkpoint_reproduces() -> bool:
    actor, c1, c2, m = _tiny_actor(actor_lr=2e-3, ppo_clip=0.5)
    rng = jax.random.PRNGKey(102)
    traj0, _ = actor.collect(c1, c2, m, rng=rng)
    sd = jax.tree_util.tree_map(lambda x: jnp.array(x), actor.state_dict())

    # Mutate the actor with an update, then confirm the rollout changed.
    _, trace = actor.collect(c1, c2, m, rng=rng)
    actor.ppo_update(c1, c2, m, trace, trace["log_prob"], jnp.ones((c1.shape[0],)))
    traj1, _ = actor.collect(c1, c2, m, rng=rng)
    changed = not _close(traj0, traj1, atol=1e-5)

    # Restore and confirm the rollout is reproduced.
    actor.load_state_dict(sd)
    traj2, _ = actor.collect(c1, c2, m, rng=rng)
    reproduced = _close(traj0, traj2, atol=0.0)
    print(f"  update changed rollout={changed}  restore reproduces rollout={reproduced}")
    return bool(changed and reproduced)


def gate_sigma_floors_must_match() -> bool:
    """Sampling and log-prob must describe the same policy (equal sigma floors).

    A smaller sample floor than log-prob floor makes every recorded action
    off-policy w.r.t. the density PPO optimizes; the actor rejects it up front.
    """
    try:
        _tiny_actor(sigma_sample_floor=0.05, sigma_logprob_floor=0.1)
    except ValueError:
        print("  mismatched floors rejected: True")
        return True
    print("  mismatched floors rejected: False")
    return False


def gate_kl_stop_declines_crossing_step() -> bool:
    """With a tiny target_kl, a drifted epoch declines the step (applied == 0).

    The gate must fire *before* the step that crosses the trust region, not one
    step after: the first update (KL 0) applies, later ones over target do not.
    """
    actor, c1, c2, m = _tiny_actor(actor_lr=5e-2, ppo_clip=0.5)
    _, trace = actor.collect(c1, c2, m, rng=jax.random.PRNGKey(103))
    old_lp = trace["log_prob"]
    adv = jnp.ones((c1.shape[0],))
    m0 = actor.ppo_update(c1, c2, m, trace, old_lp, adv, target_kl=1e-6)
    tparams_after_first = jax.tree_util.tree_map(lambda x: jnp.array(x), actor._tparams)
    m1 = actor.ppo_update(c1, c2, m, trace, old_lp, adv, target_kl=1e-6)
    unchanged = all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(tparams_after_first),
                        jax.tree_util.tree_leaves(actor._tparams))
    )
    applied0 = float(np.asarray(m0["applied"]))
    applied1 = float(np.asarray(m1["applied"]))
    print(f"  first applied={applied0}  second applied={applied1}  "
          f"params frozen after decline={unchanged}")
    return bool(applied0 == 1.0 and applied1 == 0.0 and unchanged)


def main() -> int:
    print(f"[gate2b] devices={jax.devices()}")
    checks = [
        ("gradient isolation (only trainable denoiser)", gate_gradient_isolation),
        ("positive advantage increases log-prob", gate_positive_advantage_increases_prob),
        ("checkpoint/resume reproduces rollout", gate_checkpoint_reproduces),
        ("sampling/log-prob sigma floors must match", gate_sigma_floors_must_match),
        ("KL early-stop declines the crossing step", gate_kl_stop_declines_crossing_step),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n=== {name} ===")
        try:
            results[name] = bool(fn())
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    print("\n=========== Gate 2 (RL actor) summary ===========")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allp = all(results.values())
    print(f"\nGate 2 RL-actor overall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
