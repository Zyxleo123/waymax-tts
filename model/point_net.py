from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from flax import nnx

from .mlp import MLP


class PointNet(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        n_layer: int = 4,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",
    ) -> None:
        if pool_mode != "max":
            raise ValueError(f"pool_mode {pool_mode} not supported")

        self.pool_mode = pool_mode
        self.hidden_dim = hidden_dim
        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            rngs=rngs,
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
        )

        self.mlp_layers = [
            MLP(
                [hidden_dim, hidden_dim // 2],
                rngs=rngs,
                dropout_p=dropout_p,
                use_layernorm=use_layernorm,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(n_layer - 1)
        ]

        self.mlp_out = MLP(
            [hidden_dim, hidden_dim],
            rngs=rngs,
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
            end_layer_activation=end_layer_activation,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        valid: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"Input x should be 3D tensor, got {x.shape}")

        b, n, _ = x.shape
        if valid is None:
            valid = jnp.ones((b, n), dtype=bool)

        has_any = jnp.any(valid, axis=1)
        neg_inf = jnp.array(-1e9, dtype=x.dtype)

        h = self.input_mlp(x, deterministic=deterministic)

        for mlp in self.mlp_layers:
            feature_encoded = mlp(h, deterministic=deterministic)
            feature_pooled = jnp.where(valid[..., None], feature_encoded, neg_inf).max(axis=1, keepdims=True)
            feature_pooled = jnp.where(has_any[:, None, None], feature_pooled, jnp.zeros_like(feature_pooled))
            h = jnp.concatenate([feature_encoded, jnp.broadcast_to(feature_pooled, (b, n, feature_pooled.shape[-1]))], axis=-1)

        h = jnp.where(valid[..., None], h, neg_inf).max(axis=1, keepdims=False)
        h = jnp.where(has_any[:, None], h, jnp.zeros_like(h))
        return self.mlp_out(h, deterministic=deterministic)
