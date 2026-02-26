# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from flax import nnx


def _activation_fn(activation: str):
    if activation == "relu":
        return nnx.relu
    if activation == "gelu":
        return nnx.gelu
    raise RuntimeError(f"activation {activation} not implemented")


class MLP(nnx.Module):
    def __init__(
        self,
        fc_dims: Union[List[int], Tuple[int, ...]],
        dropout_p: Optional[float] = None,
        use_layernorm: bool = False,
        activation: str = "relu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_batchnorm: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        assert len(fc_dims) >= 2
        assert not (use_layernorm and use_batchnorm)
        if use_batchnorm:
            raise NotImplementedError("BatchNorm in this MLP is not supported in JAX path.")

        self.fc_dims = tuple(fc_dims)
        self.dropout_p = dropout_p
        self.use_layernorm = use_layernorm
        self.activation = activation
        self.end_layer_activation = end_layer_activation
        self.init_weight_norm = init_weight_norm

        self.layers = []
        self.norms = []
        self.dropouts = []

        for i in range(len(self.fc_dims) - 1):
            in_dim, out_dim = self.fc_dims[i], self.fc_dims[i + 1]
            bias_init = nnx.initializers.zeros
            if i == len(self.fc_dims) - 2 and init_bias is not None:
                bias_init = nnx.initializers.constant(init_bias)
            dense = nnx.Linear(in_dim, out_dim, use_bias=True, kernel_init=nnx.initializers.lecun_normal(), bias_init=bias_init, rngs=rngs)
            self.layers.append(dense)

            apply_post = (i < len(self.fc_dims) - 2) or self.end_layer_activation
            if apply_post and self.use_layernorm:
                self.norms.append(nnx.LayerNorm(num_features=out_dim, rngs=rngs))
            else:
                self.norms.append(None)

            if apply_post and self.dropout_p is not None:
                self.dropouts.append(nnx.Dropout(rate=self.dropout_p, rngs=rngs))
            else:
                self.dropouts.append(None)

    def __call__(
        self,
        x: jnp.ndarray,
        valid_mask: Optional[jnp.ndarray] = None,
        fill_invalid: float = 0.0,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        act = _activation_fn(self.activation)
        out = x
        n_layers = len(self.layers)

        for i in range(n_layers):
            out = self.layers[i](out)

            if self.init_weight_norm:
                out = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)

            apply_post = (i < n_layers - 1) or self.end_layer_activation
            if apply_post:
                if self.norms[i] is not None:
                    out = self.norms[i](out)
                if self.dropouts[i] is not None:
                    out = self.dropouts[i](out, deterministic=deterministic)
                out = act(out)

        if valid_mask is not None:
            out = jnp.where(valid_mask[..., None], out, jnp.asarray(fill_invalid, dtype=out.dtype))
        return out
