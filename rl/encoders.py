"""Torch ports of V-Max's observation encoders, as SB3 ``BaseFeaturesExtractor``es.

V-Max only runs on its own ScenarioMax-shaped observation; these reimplement its
encoders natively in torch against the block-structured observation
``rl.waymax_env._compute_observation`` produces (layout shared via
``rl.obs_layout``), so they work directly on raw WOMD -- no ScenarioMax
conversion.

Two encoders, both operating on the same blocks:

``lq`` -- **the one V-Max actually used.** Every run in the live
``/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/`` tree, including the ~97%
``repro_sac_v2``, set ``network.encoder.type: lq``. (Careful: the sibling key
``algorithm.network.encoder.type`` reads ``none`` in the same file and is *not*
what the code loads -- see ``train_utils.py:262``.) Perceiver-style: all blocks'
tokens are concatenated into **one** sequence and a single set of learned latents
cross-attends into it, alternating cross- and self-attention for ``depth``
layers with optionally tied weights. Port of
``V-Max/vmax/agents/networks/encoders/lq.py``.

``wayformer`` -- attends **per block** and concatenates the resulting latents
("late fusion"). Port of ``.../encoders/wayformer.py``. Kept for comparison; no
V-Max run in the live tree used it.

Shared conventions taken from ``.../encoders/attention_utils.py``:
  * ``AttentionLayer`` projects q from the query tensor and k/v from the context
    to ``heads * head_features``, then projects back to the **query's** width.
  * ``ReZero`` is a scalar residual gate initialised to zero.
  * ``FeedForward`` is Linear(d*mult) -> GELU -> Linear(d).
  * Masks are *validity* masks (True = keep), the opposite of torch's
    ``key_padding_mask``.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rl.obs_layout import DEFAULT_OBS_BLOCKS, ObsBlock, block_offsets, obs_layout_size


def _mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """V-Max ``build_mlp_embedding``: Dense/act per hidden size, then Dense(out)."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class _ReZero(nn.Module):
    """Learned scalar residual gate, initialised to 0 (Flax ``ReZero``)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class _FeedForward(nn.Module):
    """Flax ``FeedForward``: Dense(d*mult) -> GELU -> dropout -> Dense(d)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AttentionLayer(nn.Module):
    """Flax ``AttentionLayer``: q from ``x``, k/v from ``context``.

    Query and context may have different widths (in LQ the latents are
    ``dk * ff_mult`` wide while the context tokens are ``dk``), so the output
    projection maps back to the *query* width.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        head_features: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner = heads * head_features
        self.heads = heads
        self.scale = head_features ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: [B, Nq, Dq]. ``context``: [B, Nk, Dc]. ``mask_k``: [B, Nk] True=keep."""
        ctx = x if context is None else context
        b, nq, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(b, nq, h, -1)
        k = self.to_k(ctx).view(b, ctx.shape[1], h, -1)
        v = self.to_v(ctx).view(b, ctx.shape[1], h, -1)

        sim = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        if mask_k is not None:
            sim = sim.masked_fill(~mask_k[:, None, :, None], torch.finfo(sim.dtype).min)
        attn = sim.softmax(dim=-2)  # over keys

        out = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, nq, -1)
        return self.dropout(self.to_out(out))


class _LQAttention(nn.Module):
    """Flax ``LQAttention``: latents alternately cross- then self-attend, ``depth`` times.

    The latent bank is ``context_dim * ff_mult`` wide (matching the Flax
    ``latents`` param shape ``(num_latents, dim * ff_mult)``). With
    ``tie_layer_weights`` the attention/FF modules are shared across depths while
    each depth keeps its own ReZero gates -- and, as in the source, a single
    ReZero instance is shared between the attention and the FF within one half of
    a layer.
    """

    def __init__(
        self,
        context_dim: int,
        depth: int = 4,
        num_latents: int = 16,
        latent_num_heads: int = 2,
        latent_head_features: int = 16,
        cross_num_heads: int = 2,
        cross_head_features: int = 16,
        ff_mult: int = 2,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        tie_layer_weights: bool = True,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.tied = bool(tie_layer_weights)
        latent_dim = context_dim * ff_mult
        self.latent_dim = latent_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        def cross() -> _AttentionLayer:
            return _AttentionLayer(latent_dim, context_dim, cross_num_heads,
                                   cross_head_features, attn_dropout)

        def selfa() -> _AttentionLayer:
            return _AttentionLayer(latent_dim, latent_dim, latent_num_heads,
                                   latent_head_features, attn_dropout)

        n = 1 if self.tied else self.depth
        self.cross_attn = nn.ModuleList([cross() for _ in range(n)])
        self.self_attn = nn.ModuleList([selfa() for _ in range(n)])
        self.cross_ff = nn.ModuleList([_FeedForward(latent_dim, ff_mult, ff_dropout) for _ in range(n)])
        self.self_ff = nn.ModuleList([_FeedForward(latent_dim, ff_mult, ff_dropout) for _ in range(n)])

        # ReZero gates are always per-depth, even when weights are tied.
        self.rz_cross = nn.ModuleList([_ReZero() for _ in range(self.depth)])
        self.rz_self = nn.ModuleList([_ReZero() for _ in range(self.depth)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``x``: [B, N, context_dim] tokens. ``mask``: [B, N] True = valid."""
        latent = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for i in range(self.depth):
            j = 0 if self.tied else i
            latent = latent + self.rz_cross[i](self.cross_attn[j](latent, x, mask_k=mask))
            latent = latent + self.rz_cross[i](self.cross_ff[j](latent))
            latent = latent + self.rz_self[i](self.self_attn[j](latent))
            latent = latent + self.rz_self[i](self.self_ff[j](latent))
        return latent


class _BlockEmbedding(nn.Module):
    """Per-block token embedding + learned positional embedding.

    Un-flattens one block to [B, tokens, feat] (+ validity), embeds to ``dk``,
    and adds a learned positional embedding over the block's token slots.
    """

    def __init__(self, block: ObsBlock, dk: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.block = block
        self.embed = _mlp(block.feat_dim, hidden, dk)
        self.pos = nn.Parameter(torch.randn(block.num_tokens, dk) * 0.02)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.block
        rows = chunk.reshape(-1, b.num_tokens, b.stride)
        if b.has_valid:
            feats, valid = rows[..., :-1], rows[..., -1] > 0.5
        else:
            feats = rows
            valid = torch.ones(rows.shape[:2], dtype=torch.bool, device=rows.device)
        tokens = self.embed(feats * valid.unsqueeze(-1)) + self.pos.unsqueeze(0)
        return tokens, valid


class _BlockTokenizer(nn.Module):
    """Shared front-end: flat observation -> per-block tokens and validity masks."""

    def __init__(self, blocks: Sequence[ObsBlock], dk: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.blocks = list(blocks)
        self._offsets = block_offsets(tuple(blocks))
        self.embeds = nn.ModuleList([_BlockEmbedding(b, dk, hidden) for b in self.blocks])

    def forward(self, obs: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for off, b, emb in zip(self._offsets, self.blocks, self.embeds):
            out.append(emb(obs[:, off : off + b.size]))
        return out


def _check_width(observation_space: gym.Space, blocks: Sequence[ObsBlock]) -> None:
    expected = obs_layout_size(tuple(blocks))
    actual = int(observation_space.shape[0])
    if actual != expected:
        raise ValueError(
            f"observation_space width {actual} does not match obs_blocks layout "
            f"width {expected}; pass the same block layout the env was built with."
        )


class LQFeaturesExtractor(BaseFeaturesExtractor):
    """V-Max's ``lq`` encoder: one latent bank cross-attending all blocks at once.

    Defaults are ``repro_sac_v2``'s ``network.encoder`` block verbatim.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        obs_blocks: Sequence[ObsBlock] = DEFAULT_OBS_BLOCKS,
        dk: int = 64,
        embedding_layer_sizes: Sequence[int] = (256, 256),
        encoder_depth: int = 4,
        num_latents: int = 16,
        latent_num_heads: int = 2,
        latent_head_features: int = 16,
        cross_num_heads: int = 2,
        cross_head_features: int = 16,
        ff_mult: int = 2,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        tie_layer_weights: bool = True,
        features_dim: int | None = None,
    ) -> None:
        # V-Max feeds the mean latent (dk * ff_mult wide) straight to the policy
        # head, so that is the natural output width.
        out_dim = int(features_dim) if features_dim else dk * ff_mult
        _check_width(observation_space, obs_blocks)
        super().__init__(observation_space, out_dim)

        self.tokenizer = _BlockTokenizer(obs_blocks, dk, embedding_layer_sizes)
        self.attention = _LQAttention(
            context_dim=dk,
            depth=encoder_depth,
            num_latents=num_latents,
            latent_num_heads=latent_num_heads,
            latent_head_features=latent_head_features,
            cross_num_heads=cross_num_heads,
            cross_head_features=cross_head_features,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            tie_layer_weights=tie_layer_weights,
        )
        latent_dim = dk * ff_mult
        self.out_proj = nn.Identity() if out_dim == latent_dim else nn.Linear(latent_dim, out_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        per_block = self.tokenizer(observations)
        tokens = torch.cat([t for t, _ in per_block], dim=1)
        mask = torch.cat([m for _, m in per_block], dim=1)

        # Softmax over an all-masked key row is NaN. A scene with no traffic
        # lights and no road edges in the box can make every token invalid, so
        # fall back to attending over everything in that case.
        all_invalid = ~mask.any(dim=1, keepdim=True)
        mask = mask | all_invalid

        latent = self.attention(tokens, mask)
        return self.out_proj(latent.mean(dim=1))


class WayformerFeaturesExtractor(BaseFeaturesExtractor):
    """V-Max's ``wayformer`` encoder: per-block latent attention, late fusion.

    Kept for comparison. No run in the live ``vmax_repro`` tree used this; they
    all used :class:`LQFeaturesExtractor`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        obs_blocks: Sequence[ObsBlock] = DEFAULT_OBS_BLOCKS,
        dk: int = 64,
        embedding_layer_sizes: Sequence[int] = (256, 256),
        attention_depth: int = 2,
        num_latents: int = 16,
        latent_num_heads: int = 4,
        latent_head_features: int = 16,
        ff_mult: int = 2,
        dropout: float = 0.0,
        features_dim: int = 256,
    ) -> None:
        _check_width(observation_space, obs_blocks)
        super().__init__(observation_space, features_dim)

        self.tokenizer = _BlockTokenizer(obs_blocks, dk, embedding_layer_sizes)
        self.per_block = nn.ModuleList(
            [
                _LQAttention(
                    context_dim=dk,
                    depth=attention_depth,
                    num_latents=num_latents,
                    latent_num_heads=latent_num_heads,
                    latent_head_features=latent_head_features,
                    cross_num_heads=latent_num_heads,
                    cross_head_features=latent_head_features,
                    ff_mult=ff_mult,
                    attn_dropout=dropout,
                    ff_dropout=dropout,
                    tie_layer_weights=False,
                )
                for _ in obs_blocks
            ]
        )
        self.out_proj = nn.Sequential(nn.Linear(dk * ff_mult, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        latents = []
        for (tokens, valid), attn in zip(self.tokenizer(observations), self.per_block):
            mask = valid | ~valid.any(dim=1, keepdim=True)
            latents.append(attn(tokens, mask))
        return self.out_proj(torch.cat(latents, dim=1).mean(dim=1))


# --------------------------------------------------------------------------- #
# CLI / policy_kwargs plumbing (shared by train_sac.py, train_ppo.py,
# train_bc_sac.py so the encoder is configured identically everywhere)
# --------------------------------------------------------------------------- #
ENCODER_CHOICES = ("mlp", "lq", "wayformer")

# repro_sac_v2's algorithm.network.policy/value layer_sizes.
DEFAULT_NET_ARCH = (256, 64, 32)


def add_encoder_args(parser) -> None:
    """Registers the encoder flags on an ``argparse`` parser."""
    g = parser.add_argument_group("encoder")
    g.add_argument("--encoder", type=str, default="lq", choices=list(ENCODER_CHOICES),
                   help="Policy feature extractor. 'lq' is V-Max's latent-query "
                        "encoder (what repro_sac_v2 used) and the default; 'mlp' is "
                        "SB3's flat MLP; 'wayformer' is the per-block variant.")
    g.add_argument("--encoder-dk", type=int, default=64,
                   help="Token embedding width.")
    g.add_argument("--encoder-depth", type=int, default=4,
                   help="Attention depth (V-Max encoder_depth=4 for lq).")
    g.add_argument("--encoder-latents", type=int, default=16,
                   help="Number of learned latent vectors.")
    g.add_argument("--encoder-heads", type=int, default=2,
                   help="Attention heads (latent and cross).")
    g.add_argument("--encoder-head-dim", type=int, default=16,
                   help="Per-head width.")
    g.add_argument("--encoder-ff-mult", type=int, default=2,
                   help="Feedforward multiplier; also sets the latent width (dk * ff_mult).")
    g.add_argument("--encoder-no-tie-weights", action="store_true",
                   help="Give each attention depth its own weights (V-Max ties them).")
    g.add_argument("--encoder-features-dim", type=int, default=0,
                   help="Output width. 0 (default) keeps the encoder's natural "
                        "latent width, as V-Max does.")
    g.add_argument("--net-arch", type=str, default=None,
                   help="Comma-separated policy/value head sizes. Default "
                        f"{','.join(str(x) for x in DEFAULT_NET_ARCH)} (repro_sac_v2).")


def build_policy_kwargs(args) -> dict:
    """Builds SB3 ``policy_kwargs`` from parsed ``add_encoder_args`` flags."""
    net_arch = (
        [int(x) for x in args.net_arch.split(",") if x.strip()]
        if getattr(args, "net_arch", None)
        else list(DEFAULT_NET_ARCH)
    )
    policy_kwargs: dict = {"net_arch": net_arch}
    encoder = getattr(args, "encoder", "lq")
    if encoder == "mlp":
        return policy_kwargs

    common = {
        "dk": int(args.encoder_dk),
        "num_latents": int(args.encoder_latents),
        "ff_mult": int(args.encoder_ff_mult),
    }
    if encoder == "lq":
        policy_kwargs["features_extractor_class"] = LQFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            **common,
            "encoder_depth": int(args.encoder_depth),
            "latent_num_heads": int(args.encoder_heads),
            "latent_head_features": int(args.encoder_head_dim),
            "cross_num_heads": int(args.encoder_heads),
            "cross_head_features": int(args.encoder_head_dim),
            "tie_layer_weights": not args.encoder_no_tie_weights,
            "features_dim": int(args.encoder_features_dim) or None,
        }
    else:  # wayformer
        policy_kwargs["features_extractor_class"] = WayformerFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            **common,
            "attention_depth": int(args.encoder_depth),
            "latent_num_heads": int(args.encoder_heads),
            "latent_head_features": int(args.encoder_head_dim),
            "features_dim": int(args.encoder_features_dim) or 256,
        }
    return policy_kwargs
