"""JAX/Flax implementations of multi-head Self-Attention and Cross-Attention.

Provides:
- MultiHeadDotProductAttention: core attention implementation
- SelfAttention: wrapper using same tensor for q/k/v
- CrossAttention: wrapper using separate memory for k/v

This file uses Flax Linen (flax.linen). If your environment doesn't
have Flax installed, install it with `pip install flax`.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

try:
    import flax.linen as nn
except Exception as e:
    raise ImportError("flax is required for this module. Install with `pip install flax`.") from e


def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    # x: [batch, seq, d_model] -> [batch, heads, seq, head_dim]
    batch, seq, d_model = x.shape
    head_dim = d_model // num_heads
    x = x.reshape(batch, seq, num_heads, head_dim)
    return x.transpose(0, 2, 1, 3)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    # x: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
    batch, heads, seq, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch, seq, heads * head_dim)


class MultiHeadDotProductAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = False

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Compute multi-head attention.

        Args:
            query: [batch, seq_q, d_model]
            key:   [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask:  optional attention mask. Expected broadcasting shapes:
                   [batch, 1, 1, seq_k] or [batch, 1, seq_q, seq_k] or [batch, seq_q, seq_k]
                   mask should be 1.0 for allowed positions and 0.0 for masked positions.
            deterministic: if False applies dropout.
            return_attn: if True returns attention weights as second output.

        Returns:
            output: [batch, seq_q, d_model]
            attn_weights (optional): [batch, num_heads, seq_q, seq_k]
        """
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        q_proj = nn.Dense(self.d_model, use_bias=self.use_bias, name="q_proj")(query)
        k_proj = nn.Dense(self.d_model, use_bias=self.use_bias, name="k_proj")(key)
        v_proj = nn.Dense(self.d_model, use_bias=self.use_bias, name="v_proj")(value)

        q = _split_heads(q_proj, self.num_heads)  # [B, H, Tq, Dh]
        k = _split_heads(k_proj, self.num_heads)  # [B, H, Tk, Dh]
        v = _split_heads(v_proj, self.num_heads)  # [B, H, Tk, Dh]

        depth = q.shape[-1]
        # scaled dot-product
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(depth).astype(q.dtype)

        # causal mask
        if self.causal:
            seq_q = q.shape[2]
            seq_k = k.shape[2]
            causal_mask = jnp.tril(jnp.ones((seq_q, seq_k), dtype=jnp.bool_))
            causal_mask = causal_mask[jnp.newaxis, jnp.newaxis, :, :]
            # convert to additive mask
            scores = jnp.where(causal_mask, scores, -1e9)

        if mask is not None:
            # mask: 1 for keep, 0 for mask. Convert to additive
            # broadcast mask to [B, 1, Q, K] if needed
            mask = jnp.asarray(mask)
            if mask.ndim == 2:
                # [B, K] -> [B, 1, 1, K]
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                # [B, Q, K] -> [B, 1, Q, K]
                mask = mask[:, None, :, :]
            elif mask.ndim == 4:
                # assume already [B, 1, Q, K] or [B, H, Q, K]
                pass
            scores = jnp.where(mask.astype(bool), scores, -1e9)

        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=deterministic)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        out = _merge_heads(out)
        out = nn.Dense(self.d_model, use_bias=self.use_bias, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)

        if return_attn:
            return out, attn_weights
        return out, None


class SelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True
    causal: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        attn = MultiHeadDotProductAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            causal=self.causal,
        )
        return attn(x, x, x, mask=mask, deterministic=deterministic, return_attn=return_attn)


class CrossAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        memory: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attn: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Cross-attention where `memory` provides keys and values.

        Args:
            query: [batch, seq_q, d_model]
            memory: [batch, seq_k, d_model]
        """
        attn = MultiHeadDotProductAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            causal=False,
        )
        return attn(query, memory, memory, mask=mask, deterministic=deterministic, return_attn=return_attn)


__all__ = ["MultiHeadDotProductAttention", "SelfAttention", "CrossAttention"]
