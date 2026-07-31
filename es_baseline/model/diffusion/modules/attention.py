from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from model.diffusion.modules.mlp import MLP


class CrossAttention(nnx.Module):
    """Multi-head cross-attention: query from features, key/value from condition tokens.

    This implementation assumes `context_mask_bn` (bool array of shape [B, n_tokens])
    is always provided by the caller. There is no Python branching on mask presence
    to avoid jax.jit retrace overhead.
    """
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        rngs: nnx.Rngs = None,
    ):
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nnx.Linear(query_dim, query_dim, rngs=rngs)
        self.k_proj = nnx.Linear(context_dim, query_dim, rngs=rngs)
        self.v_proj = nnx.Linear(context_dim, query_dim, rngs=rngs)
        self.out_proj = MLP([query_dim, hidden_dim, query_dim], rngs=rngs)

    def __call__(self, query_btc: jnp.ndarray, context_bnc: jnp.ndarray, context_mask_bn: jnp.ndarray) -> jnp.ndarray:
        """Apply cross-attention.

        Args:
            query_btc: [B_q, T, C] query (may be B_q==1 and will be broadcast over batch)
            context_bnc: [B, n_tokens, C_context] condition tokens
            context_mask_bn: [B, n_tokens] boolean mask for condition tokens (True=keep)

        Returns:
            [B, T, C] attended features (batch size equals `context_bnc.shape[0]`)
        """
        # Allow query to be a single set of tokens that is broadcast across the batch.
        Bq, Tq, Cq = query_btc.shape
        Bc = context_bnc.shape[0]
        if Bq != Bc:
            if Bq == 1:
                query_btc = jnp.broadcast_to(query_btc, (Bc, Tq, Cq))
            else:
                raise ValueError("Batch dimension of query and context must match, or query batch==1 for broadcasting")

        B = context_bnc.shape[0]
        T = query_btc.shape[1]
        n_tokens = context_bnc.shape[1]

        # Project query, key, value
        Q = self.q_proj(query_btc)  # [B, T, C]
        K = self.k_proj(context_bnc)  # [B, n_tokens, C]
        V = self.v_proj(context_bnc)  # [B, n_tokens, C]

        # Reshape for multi-head attention: [B*num_heads, T, head_dim]
        Q = Q.reshape((B, T, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, T, self.head_dim))
        K = K.reshape((B, n_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, n_tokens, self.head_dim))
        V = V.reshape((B, n_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B * self.num_heads, n_tokens, self.head_dim))

        # Attention scores: [B*num_heads, T, n_tokens]
        attn_scores = (Q @ K.transpose((0, 2, 1))) * self.scale

        # Broadcast boolean mask -> [B*num_heads, 1, n_tokens]
        mask_bhn = jnp.tile(context_mask_bn[:, None, :], (1, self.num_heads, 1))  # [B, H, n]
        mask_flat = mask_bhn.reshape((B * self.num_heads, 1, n_tokens))
        attn_scores = jnp.where(mask_flat, attn_scores, jnp.array(-1e9, dtype=attn_scores.dtype))

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # [B*num_heads, T, n_tokens]

        # Apply attention to values: [B*num_heads, T, head_dim]
        out = attn_weights @ V

        # Reshape back: [B, T, C]
        out = out.reshape((B, self.num_heads, T, self.head_dim)).transpose((0, 2, 1, 3)).reshape((B, T, Cq))
        out = self.out_proj(out)
        return out


class CrossAttentionLayers(nnx.Module):
    """Stack of cross-attention layers with residual connections."""
    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 4, rngs: nnx.Rngs = None):
        self.attn_layers = nnx.List([CrossAttention(query_dim, context_dim, hidden_dim, num_heads, rngs=rngs) for _ in range(num_layers)])

    def __call__(self, query_btc: jnp.ndarray, context_bnc: jnp.ndarray, context_mask_bn: jnp.ndarray) -> jnp.ndarray:
        h = query_btc
        for layer in self.attn_layers:
            attn_out = layer(h, context_bnc, context_mask_bn)
            h = h + attn_out  # Residual connection
        return h


__all__ = ["CrossAttention", "CrossAttentionLayers"]
