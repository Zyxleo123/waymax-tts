"""State-only value network for DPPO, over stopped-gradient scene features.

The critic estimates ``V(s)`` from the frozen scene condition (the scene
tokenizer's output ``cond1``, mean-pooled over tokens). The input is
``stop_gradient``-ed so value learning never flows back into the (frozen) scene
tokenizer -- only the value MLP's own parameters train, with a separate
optimizer. This is the ``state-only value network using stopped-gradient scene
features`` from the DPPO plan.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class ValueMLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, feat: jnp.ndarray) -> jnp.ndarray:
        h = nnx.silu(self.fc1(feat))
        h = nnx.silu(self.fc2(h))
        return self.out(h)[..., 0]  # [B]


def pool_condition(cond1: jnp.ndarray) -> jnp.ndarray:
    """Mean-pool scene tokens ``[B, n_tokens, D]`` -> ``[B, D]`` (stopped grad)."""
    return jax.lax.stop_gradient(jnp.mean(cond1, axis=1))


class ValueCritic:
    """Value MLP + its own Adam optimizer, updated by MSE to returns."""

    def __init__(self, cond_dim: int, *, hidden: int = 256, lr: float = 1e-3,
                 grad_clip_norm: float = 1.0, seed: int = 0):
        self.model = ValueMLP(cond_dim, hidden, rngs=nnx.Rngs(int(seed)))
        self._gdef, self._params, self._nonparam = nnx.split(self.model, nnx.Param, ...)
        self.tx = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            optax.adam(float(lr)),
        )
        self._opt_state = self.tx.init(self._params)

    def _apply(self, params, feat):
        model = nnx.merge(self._gdef, params, self._nonparam)
        return model(feat)

    def value(self, cond1: jnp.ndarray) -> jnp.ndarray:
        """Predict ``V(s)`` ``[B]`` from scene tokens (no gradient tracked)."""
        return self._apply(self._params, pool_condition(cond1))

    def update(self, cond1: jnp.ndarray, returns_b: jnp.ndarray, *,
               weights: jnp.ndarray | None = None, epochs: int = 1) -> dict[str, Any]:
        """Fit ``V`` to ``returns_b`` for ``epochs`` gradient steps.

        ``weights`` is an optional ``[B]`` 0/1 mask (the ``alive`` flags): steps
        from already-terminated environments carry meaningless returns and must
        not be regressed on. The MSE is normalized by the weight mass, so masking
        does not silently shrink the gradient.

        A single step per iteration cannot track the target while the critic is
        still far from correct -- the GAE targets inflate faster than V rises --
        so this defaults to being called with ``epochs`` > 1.
        """
        feat = pool_condition(cond1)
        w = jnp.ones_like(returns_b) if weights is None else weights.astype(returns_b.dtype)
        w_sum = jnp.maximum(jnp.sum(w), 1.0)

        def loss_fn(params):
            pred = self._apply(params, feat)
            return jnp.sum(w * (pred - returns_b) ** 2) / w_sum

        first = None
        loss = jnp.asarray(0.0)
        for _ in range(max(1, int(epochs))):
            loss, grads = jax.value_and_grad(loss_fn)(self._params)
            updates, self._opt_state = self.tx.update(grads, self._opt_state, self._params)
            self._params = optax.apply_updates(self._params, updates)
            if first is None:
                first = loss
        # `value_loss` stays the pre-update fit (comparable across iterations);
        # `value_loss_final` shows whether the extra epochs actually helped.
        return {"value_loss": first, "value_loss_final": loss}

    def state_dict(self) -> dict[str, Any]:
        return {"params": self._params, "opt_state": self._opt_state}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._params = sd["params"]
        self._opt_state = sd["opt_state"]
