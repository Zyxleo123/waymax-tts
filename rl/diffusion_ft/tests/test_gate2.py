"""Gate 2 (diffusion-API core): validate the RL extensions to the diffusion model.

Uses a *tiny* synthetic ``GaussianDiffusion`` (no checkpoint, small dims, few
timesteps) so the math/plumbing checks run fast on CPU. Covers the checkpoint-
independent Gate 2 items:

* ``loss == loss_per_example().mean()`` under the same rng.
* ``sample_with_trace(rl_mode=False).trajectory == sample()`` bit-for-bit (same rng).
* Recomputed transition log-probs equal the stored trace values (=> PPO ratio 1).
* ``sample_with_trace`` routes the late steps through a substitute (trainable)
  denoiser, and through the frozen one for the early steps.

The gradient-isolation and positive-advantage-update items live with the RL
policy wrapper (they need the two-network split + an optimizer) and are covered
separately once that module exists.
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


def _tiny():
    C, T, cond_dim, hidden = 5, 8, 16, 32
    rngs = nnx.Rngs(0)
    unet = UNet1DConditioned(
        in_ch=C, cond_dim=cond_dim, cond2_dim=cond_dim, time_emb_dim=cond_dim,
        horizon=T, hidden_dim=hidden, base_ch=16, rngs=rngs,
    )
    diff = GaussianDiffusion(unet, timesteps=10, predict_type="v")
    B = 3
    kc = jax.random.PRNGKey(1)
    cond1 = jax.random.normal(kc, (B, 4, cond_dim))
    cond2 = jax.random.normal(jax.random.PRNGKey(2), (B, cond_dim))
    mask = jnp.ones((B,), dtype=jnp.float32)
    return diff, (B, C, T), cond1, cond2, mask


def _close(a, b, atol=1e-5, rtol=1e-5):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))


def gate_loss_per_example(diff, shape, c1, c2, m) -> bool:
    B, C, T = shape
    x0 = jax.random.normal(jax.random.PRNGKey(10), shape)
    rng = jax.random.PRNGKey(11)
    l = float(diff.loss(x0, c1, c2, m, rng=rng))
    lpe = np.asarray(diff.loss_per_example(x0, c1, c2, m, rng=rng))
    ok = lpe.shape == (B,) and _close(l, lpe.mean())
    print(f"  loss={l:.6f} loss_per_example.mean={lpe.mean():.6f} shape={lpe.shape} ok={ok}")
    return ok


def gate_sample_matches(diff, shape, c1, c2, m) -> bool:
    rng = jax.random.PRNGKey(20)
    base = np.asarray(diff.sample(shape, c1, c2, m, rng=rng))
    ok_all = True
    for k in (0, 3, 10):
        traj = np.asarray(diff.sample_with_trace(shape, c1, c2, m, rng=rng, k_trainable=k, rl_mode=False)[0])
        max_abs = float(np.max(np.abs(base - traj)))
        # k=0 is the pure fori path -> bit-identical. k>0 splits the chain into a
        # fori prefix + an unrolled suffix, which XLA float-reassociates slightly;
        # require numerical (not bitwise) agreement there.
        tol = 0.0 if k == 0 else 1e-4
        ok = max_abs <= tol
        print(f"  k_trainable={k}: max|Δ|={max_abs:.2e} tol={tol:.0e} ok={ok}")
        ok_all = ok_all and ok
    return ok_all


def gate_logprob_consistent(diff, shape, c1, c2, m) -> bool:
    """Stored trace log-prob == recomputed transition_log_prob (ratio == 1)."""
    rng = jax.random.PRNGKey(30)
    k = 4
    # Sampling and log-prob must share the same floor (the policy is a single
    # Gaussian); the actor enforces this, so exercise the consistent setting here.
    _, tr = diff.sample_with_trace(
        shape, c1, c2, m, rng=rng, k_trainable=k, rl_mode=True,
        sigma_sample_floor=0.1, sigma_logprob_floor=0.1,
    )
    ok = True
    for j in range(k):
        recomputed = diff.transition_log_prob(
            tr["x_t"][j], tr["x_prev"][j], tr["timestep"][j], c1, c2, m, sigma_floor=0.1,
        )
        stored = tr["log_prob"][j]
        match = _close(recomputed, stored, atol=1e-5)
        ratio = np.exp(np.asarray(recomputed) - np.asarray(stored))
        print(f"  step {j}: logprob match={match} ratio(mean)={float(ratio.mean()):.6f}")
        ok = ok and match and _close(ratio, np.ones_like(ratio), atol=1e-4)
    return ok


def gate_trainable_denoiser_routing(diff, shape, c1, c2, m) -> bool:
    """Late steps use the substitute denoiser; early steps use the frozen one."""
    # A substitute that returns a constant (very different from the real UNet) so
    # its influence is detectable. Signature matches denoise_fn(x, t, c1, c2, mask).
    def const_fn(x, t, cc1, cc2, cc2m):
        return jnp.ones_like(x) * 7.0

    rng = jax.random.PRNGKey(40)
    base, _ = diff.sample_with_trace(shape, c1, c2, m, rng=rng, k_trainable=3, rl_mode=False)
    sub, _ = diff.sample_with_trace(
        shape, c1, c2, m, rng=rng, k_trainable=3, rl_mode=False, trainable_denoise_fn=const_fn,
    )
    changed = not _close(base, sub, atol=1e-4)
    # With k_trainable=0 the substitute is never used -> identical to frozen.
    none_used, _ = diff.sample_with_trace(shape, c1, c2, m, rng=rng, k_trainable=0, rl_mode=False, trainable_denoise_fn=const_fn)
    frozen, _ = diff.sample_with_trace(shape, c1, c2, m, rng=rng, k_trainable=0, rl_mode=False)
    unused_ok = _close(none_used, frozen, atol=0.0, rtol=0.0)
    print(f"  substitute changes late-step output={changed}; unused when k=0 -> identical={unused_ok}")
    return bool(changed and unused_ok)


def main() -> int:
    print(f"[gate2] devices={jax.devices()}")
    diff, shape, c1, c2, m = _tiny()
    checks = [
        ("loss_per_example == loss.mean", gate_loss_per_example),
        ("sample_with_trace matches sample", gate_sample_matches),
        ("recomputed logprob == stored (ratio 1)", gate_logprob_consistent),
        ("trainable-denoiser routing", gate_trainable_denoiser_routing),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n=== {name} ===")
        try:
            results[name] = bool(fn(diff, shape, c1, c2, m))
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    print("\n=========== Gate 2 (core) summary ===========")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allp = all(results.values())
    print(f"\nGate 2 core overall: {'PASS' if allp else 'FAIL'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
