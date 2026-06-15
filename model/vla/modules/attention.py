from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.vla.modules.mlp import MLP


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = MLP([dim, hidden_dim, dim])

    def forward(self, x_btc: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x_btc: [B, T, C] input features
            mask: [B, T] boolean mask (True=keep)
        """
        b, t, c = x_btc.shape

        qkv = self.qkv_proj(x_btc)
        qkv = qkv.reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :].expand(b, self.num_heads, t, t)
            attn_scores = attn_scores.masked_fill(~mask, torch.tensor(-1e9, dtype=attn_scores.dtype, device=attn_scores.device))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = attn_weights @ v

        out = out.transpose(1, 2).reshape(b, t, c)
        return self.out_proj(out)
    

class SelfAttentionLayers(nn.Module):
    """Stack of self-attention layers with residual connections."""

    def __init__(self, dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList([
            SelfAttention(dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x_btc: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = x_btc
        for layer in self.attn_layers:
            attn_out = layer(h, mask)
            h = h + attn_out
        return h


class CrossAttention(nn.Module):
    """Multi-head cross-attention: query from features, key/value from condition tokens."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = int(num_heads)
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = MLP([query_dim, hidden_dim, query_dim])

    def forward(self, query_btc: torch.Tensor, context_bnc: torch.Tensor, context_mask_bn: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            query_btc: [B_q, T, C] query (may be B_q==1 and will be broadcast over batch)
            context_bnc: [B, n_tokens, C_context] condition tokens
            context_mask_bn: [B, n_tokens] boolean mask for condition tokens (True=keep)
        """
        bq, tq, cq = query_btc.shape
        bc = context_bnc.shape[0]
        if bq != bc:
            if bq == 1:
                query_btc = query_btc.expand(bc, tq, cq)
            else:
                raise ValueError("Batch dimension of query and context must match, or query batch==1 for broadcasting")

        b = context_bnc.shape[0]
        t = query_btc.shape[1]
        n_tokens = context_bnc.shape[1]

        q = self.q_proj(query_btc)
        k = self.k_proj(context_bnc)
        v = self.v_proj(context_bnc)

        q = q.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, t, self.head_dim)
        k = k.reshape(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, n_tokens, self.head_dim)
        v = v.reshape(b, n_tokens, self.num_heads, self.head_dim).transpose(1, 2).reshape(b * self.num_heads, n_tokens, self.head_dim)

        attn_scores = (q @ k.transpose(1, 2)) * self.scale
        mask_flat = context_mask_bn[:, None, :].expand(b, self.num_heads, n_tokens).reshape(b * self.num_heads, 1, n_tokens)
        attn_scores = attn_scores.masked_fill(~mask_flat, torch.tensor(-1e9, dtype=attn_scores.dtype, device=attn_scores.device))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = attn_weights @ v

        out = out.reshape(b, self.num_heads, t, self.head_dim).transpose(1, 2).reshape(b, t, cq)
        return self.out_proj(out)


class CrossAttentionLayers(nn.Module):
    """Stack of cross-attention layers with residual connections."""

    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList([
            CrossAttention(query_dim, context_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, query_btc: torch.Tensor, context_bnc: torch.Tensor, context_mask_bn: torch.Tensor) -> torch.Tensor:
        h = query_btc
        for layer in self.attn_layers:
            attn_out = layer(h, context_bnc, context_mask_bn)
            h = h + attn_out
        return h


__all__ = ["CrossAttention", "CrossAttentionLayers", "SelfAttention", "SelfAttentionLayers"]
