from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLP(nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output sizes")
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layer: int = 4) -> None:
        super().__init__()
        if n_layer < 2:
            raise ValueError("n_layer must be at least 2")

        self.input_mlp = MLP([input_dim, hidden_dim, hidden_dim])
        self.mlp_layers = nn.ModuleList([
            MLP([hidden_dim, hidden_dim // 2]) for _ in range(n_layer - 1)
        ])
        self.mlp_out = MLP([hidden_dim, hidden_dim])

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input x should be 3D tensor, got {tuple(x.shape)}")

        batch_size, num_points, _ = x.shape
        if valid is None:
            valid = torch.ones((batch_size, num_points), dtype=torch.bool, device=x.device)

        has_any = valid.any(dim=1)
        neg_inf = torch.full_like(x[..., 0], -1e9)

        h = self.input_mlp(x)
        for mlp in self.mlp_layers:
            feature_encoded = mlp(h)
            feature_pooled = torch.where(valid[..., None], feature_encoded, neg_inf[..., None]).amax(dim=1, keepdim=True)
            feature_pooled = torch.where(has_any[:, None, None], feature_pooled, torch.zeros_like(feature_pooled))
            h = torch.cat([feature_encoded, feature_pooled.expand(batch_size, num_points, -1)], dim=-1)

        h = torch.where(valid[..., None], h, neg_inf[..., None]).amax(dim=1)
        h = torch.where(has_any[:, None], h, torch.zeros_like(h))
        return self.mlp_out(h)


class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, n_layer: int = 2, n_heads: int | None = 4, dropout: float = 0.0, hidden=512) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.n_layer = int(n_layer)
        self.n_heads = int(n_heads)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layer):
            layer = nn.Module()
            # multi-head attention: embed_dim == feature_dim
            layer.attn = nn.MultiheadAttention(self.feature_dim, self.n_heads, dropout=dropout, batch_first=False)
            layer.ln1 = nn.LayerNorm(self.feature_dim)
            # feed-forward
            layer.ffn = nn.Sequential(
                nn.Linear(self.feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.feature_dim),
                nn.Dropout(dropout),
            )
            layer.ln2 = nn.LayerNorm(self.feature_dim)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply stacked self-attention layers.

        x: [B, T, D]
        valid: [B, T] boolean mask where True indicates valid tokens. If None,
            all tokens are treated as valid.
        """
        if x.ndim != 3:
            raise ValueError(f"Input x must be 3D [B, T, D], got {tuple(x.shape)}")

        batch_size, token_len, feat = x.shape
        if feat != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: module expects {self.feature_dim}, got {feat}")

        if valid is None:
            key_padding_mask = None
        else:
            if valid.shape != (batch_size, token_len):
                raise ValueError("valid mask must have shape [B, T]")
            # MultiheadAttention expects key_padding_mask with True for positions to ignore
            key_padding_mask = ~valid.bool()

        # transpose to [T, B, D] for nn.MultiheadAttention (unless batch_first True)
        # we constructed attn with batch_first=False, so transpose
        h = x.transpose(0, 1).contiguous()

        for layer in self.layers:
            # Self-attention: query=key=value=h
            attn_out, _ = layer.attn(h, h, h, key_padding_mask=key_padding_mask)
            h = layer.ln1(h + attn_out)
            ffn_out = layer.ffn(h.transpose(0, 1)).transpose(0, 1)
            h = layer.ln2(h + ffn_out)

        return h.transpose(0, 1).contiguous()


class SceneQwenVLA(nn.Module):
    def __init__(
        self,
        qwen_name="Qwen/Qwen3-0.6B",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 9,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_name,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        hidden = self.llm.config.hidden_size

        self.other_pos_embedding = nn.Parameter(torch.zeros(1, 128, hidden))  # max 128 other agents
        self.other_pos_embedding.data.uniform_(-0.01, 0.01)
        self.ego_tokenizer = MLP([ego_dim, hidden, hidden, hidden])
        self.goal_tokenizer = MLP([goal_dim, hidden, hidden, hidden])
        self.other_tokenizer = MLP([other_dim, hidden, hidden, hidden])
        self.tl_tokenizer = MLP([tl_dim, hidden, hidden, hidden])
        self.map_tokenizer = PointNet(map_dim, hidden)
        self.attention = SelfAttention(hidden, n_layer=4, n_heads=4)

    def _tokenize_scene_features(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        ego_state = input_features["ego_state"]
        goal_xy = input_features["goal_xy"]
        remaining_timesteps = input_features["remaining_timesteps"]
        other_states = input_features["other_states"]
        other_valid = input_features["other_valid"]
        map_features = input_features["map_features"]
        map_valid = input_features["map_valid"]
        traffic_light_features = input_features["traffic_light_features"]
        traffic_light_valid = input_features["traffic_light_valid"]

        if remaining_timesteps.ndim == 1:
            remaining_timesteps = remaining_timesteps[:, None]

        goal_input = torch.cat([goal_xy, remaining_timesteps], dim=-1)

        batch_size, num_others, other_dim = other_states.shape
        _, num_map_segments, num_points_per_segment, map_dim = map_features.shape
        _, num_tl, tl_dim = traffic_light_features.shape

        ego_token = self.ego_tokenizer(ego_state).unsqueeze(1)
        goal_token = self.goal_tokenizer(goal_input).unsqueeze(1)
        other_token = self.other_tokenizer(other_states.reshape(-1, other_dim)).reshape(batch_size, num_others, -1)
        other_token = other_token + self.other_pos_embedding[:, :num_others, :]
        map_token = self.map_tokenizer(
            map_features.reshape(-1, num_points_per_segment, map_dim),
            map_valid.reshape(-1, num_points_per_segment),
        ).reshape(batch_size, num_map_segments, -1)
        tl_token = self.tl_tokenizer(traffic_light_features.reshape(-1, tl_dim)).reshape(batch_size, num_tl, -1)
        x = torch.cat([ego_token, goal_token, other_token, map_token, tl_token], dim=1)
        x_valid = torch.cat(
            [
                torch.ones((batch_size, 2), dtype=torch.bool, device=ego_state.device),  # ego and goal always valid
                other_valid,
                map_valid.any(dim=2),  # segment valid if any point is valid
                traffic_light_valid,
            ], dim=1
        )
        x = self.attention(x, x_valid)
        return x

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,      # [B, P]
        answer_ids: torch.Tensor | None = None,  # [B, A], optional
    ):
        scene_tokens = self._tokenize_scene_features(input_features)
        device = scene_tokens.device

        text_emb = self.llm.get_input_embeddings()(prompt_ids)

        scene_tokens = scene_tokens.to(text_emb.dtype)

        if answer_ids is not None:
            answer_emb = self.llm.get_input_embeddings()(answer_ids)
            inputs_embeds = torch.cat(
                [text_emb, scene_tokens, answer_emb],
                dim=1,
            )

            labels = torch.cat(
                [
                    torch.full(prompt_ids.shape, -100, device=device),
                    torch.full(scene_tokens.shape[:2], -100, device=device),
                    answer_ids,
                ],
                dim=1,
            )
        else:
            inputs_embeds = torch.cat(
                [text_emb, scene_tokens],
                dim=1,
            )
            labels = None

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=device,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        loss = outputs.loss if labels is not None else None

        return {
            "loss": loss,
        }