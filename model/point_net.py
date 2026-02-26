# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Optional

import jax.numpy as jnp
from flax import linen as nn

from .mlp import MLP


class PointNet(nn.Module):
    input_dim: int
    hidden_dim: int
    n_layer: int = 4
    use_layernorm: bool = False
    use_batchnorm: bool = False
    end_layer_activation: bool = True
    dropout_p: Optional[float] = None
    pool_mode: str = "max"

    def setup(self) -> None:
        self.input_mlp = MLP(
            [self.input_dim, self.hidden_dim, self.hidden_dim],
            dropout_p=self.dropout_p,
            use_layernorm=self.use_layernorm,
            use_batchnorm=self.use_batchnorm,
        )
        self.mlp_layers: List[nn.Module] = [
            MLP(
                [self.hidden_dim, self.hidden_dim // 2],
                dropout_p=self.dropout_p,
                use_layernorm=self.use_layernorm,
                use_batchnorm=self.use_batchnorm,
                name=f"mlp_{i}",
            )
            for i in range(self.n_layer - 1)
        ]
        self.mlp_out = MLP(
            [self.hidden_dim, self.hidden_dim],
            dropout_p=self.dropout_p,
            use_layernorm=self.use_layernorm,
            use_batchnorm=self.use_batchnorm,
            end_layer_activation=self.end_layer_activation,
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
            feature_pooled = jnp.where(
                has_any[:, None, None],
                feature_pooled,
                jnp.zeros_like(feature_pooled),
            )
            x = jnp.concatenate((feature_encoded, jnp.broadcast_to(feature_pooled, feature_encoded.shape)), axis=-1)

        x = jnp.where(valid[..., None], x, neg_fill_value)
        x = jnp.max(x, axis=1)
        x = jnp.where(has_any[:, None], x, jnp.zeros_like(x))
        return self.mlp_out(x, deterministic=deterministic)

