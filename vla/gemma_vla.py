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


class SceneGemmaVLA(nn.Module):
    def __init__(
        self,
        gemma_name="google/gemma-4-E2B-it",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 9,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            gemma_name,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            gemma_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        special_tokens = {"additional_special_tokens": ["<SCENE>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.scene_token_id = self.tokenizer.convert_tokens_to_ids("<SCENE>")

        if hasattr(self.llm.config, "text_config"):
            hidden = self.llm.config.text_config.hidden_size
        else:
            hidden = self.llm.config.hidden_size
        # print(hidden)

        self.other_pos_embedding = nn.Parameter(torch.zeros(1, 128, hidden))  # max 128 other agents
        self.other_pos_embedding.data.uniform_(-0.01, 0.01)
        self.ego_tokenizer = MLP([ego_dim, hidden, hidden, hidden])
        self.goal_tokenizer = MLP([goal_dim, hidden, hidden, hidden])
        self.other_tokenizer = MLP([other_dim, hidden, hidden, hidden])
        self.tl_tokenizer = MLP([tl_dim, hidden, hidden, hidden])
        self.map_tokenizer = PointNet(map_dim, hidden)
        self.attention = SelfAttention(hidden, n_layer=4, n_heads=4)

        self._scene_tokens_for_hook = None
        self._scene_start_for_hook = None
        self._scene_len_for_hook = None

        self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
            self._replace_scene_embeddings_hook
        )

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

        return self.attention(x, x_valid)

    def _replace_scene_embeddings_hook(self, module, inputs, output):
        """
        Gemma4 does not allow arbitrary inputs_embeds.
        So we pass input_ids with <SCENE> placeholders, then replace those
        embeddings with our learned scene embeddings inside the embedding layer.
        """
        if self._scene_tokens_for_hook is None:
            return output

        start = self._scene_start_for_hook
        length = self._scene_len_for_hook

        # During generation, later decoding steps may only embed 1 new token.
        # In that case, do not replace anything.
        if output.ndim != 3 or output.shape[1] < start + length:
            return output

        scene_tokens = self._scene_tokens_for_hook.to(
            device=output.device,
            dtype=output.dtype,
        )

        return torch.cat(
            [
                output[:, :start, :],
                scene_tokens,
                output[:, start + length :, :],
            ],
            dim=1,
        )

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
    ):
        scene_tokens = self._tokenize_scene_features(input_features)
        device = scene_tokens.device

        batch_size = prompt_ids.shape[0]
        num_scene_tokens = scene_tokens.shape[1]

        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=self.scene_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )

        input_ids_for_gemma = torch.cat(
            [prompt_ids, scene_dummy_ids, answer_ids],
            dim=1,
        )

        labels = torch.cat(
            [
                torch.full(prompt_ids.shape, -100, dtype=answer_ids.dtype, device=device),
                torch.full(scene_dummy_ids.shape, -100, dtype=answer_ids.dtype, device=device),
                answer_ids,
            ],
            dim=1,
        )

        if self.tokenizer.pad_token_id is not None:
            labels = labels.masked_fill(
                input_ids_for_gemma == self.tokenizer.pad_token_id,
                -100,
            )
    

        if self.tokenizer.pad_token_id is not None:
            attention_mask = (input_ids_for_gemma != self.tokenizer.pad_token_id).long()
            scene_start = prompt_ids.shape[1]
            scene_end = scene_start + num_scene_tokens
            attention_mask[:, scene_start:scene_end] = 1
        else:
            attention_mask = torch.ones_like(input_ids_for_gemma, dtype=torch.long)

        self._scene_tokens_for_hook = scene_tokens
        self._scene_start_for_hook = prompt_ids.shape[1]
        self._scene_len_for_hook = num_scene_tokens

        try:
            outputs = self.llm(
                input_ids=input_ids_for_gemma,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
            )
        finally:
            self._scene_tokens_for_hook = None
            self._scene_start_for_hook = None
            self._scene_len_for_hook = None


        loss = outputs.loss

        return {
            "loss": loss,
        }