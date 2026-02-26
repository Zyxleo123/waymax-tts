# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn


def _activation_fn(activation: str):
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    raise RuntimeError(f"activation {activation} not implemented")


class MLP(nn.Module):
    fc_dims: Union[List[int], Tuple[int, ...]]
    dropout_p: Optional[float] = None
    use_layernorm: bool = False
    activation: str = "relu"
    end_layer_activation: bool = True
    init_weight_norm: bool = False
    init_bias: Optional[float] = None
    use_batchnorm: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        valid_mask: Optional[jnp.ndarray] = None,
        fill_invalid: float = 0.0,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        assert len(self.fc_dims) >= 2
        assert not (self.use_layernorm and self.use_batchnorm)
        if self.use_batchnorm:
            raise NotImplementedError("BatchNorm in this MLP is not supported in JAX path.")

        act = _activation_fn(self.activation)
        out = x
        n_layers = len(self.fc_dims) - 1

        for i in range(n_layers):
            is_last = i == n_layers - 1
            bias_init = nn.initializers.zeros
            if is_last and self.init_bias is not None:
                bias_init = nn.initializers.constant(self.init_bias)

            out = nn.Dense(
                features=self.fc_dims[i + 1],
                use_bias=True,
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=bias_init,
                name=f"fc_{i}",
            )(out)

            if self.init_weight_norm:
                # Matches the intent of the Torch path by normalizing output vectors.
                out = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)

            apply_post = (not is_last) or self.end_layer_activation
            if apply_post:
                if self.use_layernorm:
                    out = nn.LayerNorm(name=f"ln_{i}")(out)
                if self.dropout_p is not None:
                    out = nn.Dropout(rate=self.dropout_p, name=f"drop_{i}")(out, deterministic=deterministic)
                out = act(out)

        if valid_mask is not None:
            out = jnp.where(valid_mask[..., None], out, jnp.asarray(fill_invalid, dtype=out.dtype))
        return out

