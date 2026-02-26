# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional

import jax.numpy as jnp
from flax import nnx

from .mlp import MLP


class PointNet(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int = 4,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.pool_mode = pool_mode
        self.hidden_dim = hidden_dim

        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
            rngs=rngs,
        )

        self.mlp_layers = [
            MLP(
                [hidden_dim, hidden_dim // 2],
                dropout_p=dropout_p,
                use_layernorm=use_layernorm,
                use_batchnorm=use_batchnorm,
                rngs=rngs,
            )
            for _ in range(n_layer - 1)
        ]
        self.mlp_out = MLP(
            [hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
            end_layer_activation=end_layer_activation,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        valid: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        assert x.ndim == 3, f"Input x should be 3D tensor, got {x.shape}"
        b, n, _ = x.shape
        if valid is None:
            valid = jnp.ones((b, n), dtype=jnp.bool_)

        has_any = jnp.any(valid, axis=1)
        x = self.input_mlp(x, deterministic=deterministic)
        neg_fill_value = jnp.finfo(x.dtype).min

        for mlp in self.mlp_layers:
            feature_encoded = mlp(x, deterministic=deterministic)
            masked = jnp.where(valid[..., None], feature_encoded, neg_fill_value)
            feature_pooled = jnp.max(masked, axis=1, keepdims=True)
            feature_pooled = jnp.where(has_any[:, None, None], feature_pooled, jnp.zeros_like(feature_pooled))
            x = jnp.concatenate((feature_encoded, jnp.broadcast_to(feature_pooled, feature_encoded.shape)), axis=-1)

        x = jnp.where(valid[..., None], x, neg_fill_value)
        x = jnp.max(x, axis=1)
        x = jnp.where(has_any[:, None], x, jnp.zeros_like(x))
        return self.mlp_out(x, deterministic=deterministic)
