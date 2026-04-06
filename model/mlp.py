from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from flax import nnx


def _get_activation(name: str):
    if name == "relu":
        return nnx.relu
    if name == "gelu":
        return nnx.gelu
    raise RuntimeError(f"activation {name} not implemented")


class MLP(nnx.Module):
    def __init__(
        self,
        fc_dims: Sequence[int],
        rngs: nnx.Rngs,
        dropout_p: Optional[float] = None,
        use_layernorm: bool = False,
        activation: str = "relu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> None:
        if len(fc_dims) < 2:
            raise ValueError("fc_dims should include input and output dimensions")
        if use_layernorm and use_batchnorm:
            raise ValueError("use_layernorm and use_batchnorm are mutually exclusive")

        self.input_dim = int(fc_dims[0])
        self.output_dim = int(fc_dims[-1])
        self.dropout_p = dropout_p
        self.use_layernorm = use_layernorm
        self.use_batchnorm = use_batchnorm
        self.end_layer_activation = end_layer_activation
        self.init_weight_norm = init_weight_norm
        self.init_bias = init_bias
        self.activation = _get_activation(activation)

        fc_layers = []
        norm_layers = []
        dropout_layers = []

        for i in range(len(fc_dims) - 1):
            in_dim = int(fc_dims[i])
            out_dim = int(fc_dims[i + 1])
            dense = nnx.Linear(in_dim, out_dim, rngs=rngs)

            if init_weight_norm:
                w = dense.kernel.value
                norm = jnp.linalg.norm(w, axis=0, keepdims=True)
                dense.kernel.value = w / jnp.maximum(norm, 1e-8)
            if init_bias is not None and i == len(fc_dims) - 2:
                dense.bias.value = jnp.full_like(dense.bias.value, init_bias)

            fc_layers.append(dense)

            is_last = i == len(fc_dims) - 2
            if (not is_last) or end_layer_activation:
                if use_layernorm:
                    norm_layers.append(nnx.LayerNorm(num_features=out_dim, rngs=rngs))
                elif use_batchnorm:
                    norm_layers.append(nnx.BatchNorm(num_features=out_dim, rngs=rngs))
                else:
                    norm_layers.append(None)

                if dropout_p is not None:
                    dropout_layers.append(nnx.Dropout(rate=dropout_p, rngs=rngs))
                else:
                    dropout_layers.append(None)

        # Register layer containers as NNX modules in a single assignment.
        self.fc_layers = nnx.List(fc_layers)
        self.norm_layers = nnx.List(norm_layers)
        self.dropout_layers = nnx.List(dropout_layers)

    def __call__(
        self,
        x: jnp.ndarray,
        valid_mask: Optional[jnp.ndarray] = None,
        fill_invalid: float = 0.0,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        leading_shape = x.shape[:-1]
        h = x.reshape((-1, x.shape[-1]))

        norm_dropout_idx = 0
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            is_last = i == len(self.fc_layers) - 1

            if (not is_last) or self.end_layer_activation:
                norm_layer = self.norm_layers[norm_dropout_idx]
                if norm_layer is not None:
                    if self.use_batchnorm:
                        h = norm_layer(h, use_running_average=deterministic)
                    else:
                        h = norm_layer(h)

                dropout_layer = self.dropout_layers[norm_dropout_idx]
                if dropout_layer is not None:
                    h = dropout_layer(h, deterministic=deterministic)

                if not is_last:
                    h = self.activation(h)
                norm_dropout_idx += 1

        h = h.reshape(leading_shape + (self.output_dim,))
        if valid_mask is not None:
            h = jnp.where(valid_mask[..., None], h, jnp.array(fill_invalid, dtype=h.dtype))
        if self.end_layer_activation:
            h = self.activation(h)
        return h