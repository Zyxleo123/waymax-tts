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


class SceneQwenVLA(nn.Module):
    def __init__(
        self,
        qwen_name="Qwen/Qwen3-0.6B",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 7,
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

        self.ego_tokenizer = MLP([ego_dim, hidden, hidden, hidden])
        self.goal_tokenizer = MLP([goal_dim, hidden, hidden, hidden])
        self.other_tokenizer = PointNet(other_dim, hidden)
        self.map_tokenizer = PointNet(map_dim, hidden)
        self.tl_tokenizer = PointNet(tl_dim, hidden)

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

        ego_token = self.ego_tokenizer(ego_state).unsqueeze(1)
        goal_token = self.goal_tokenizer(goal_input).unsqueeze(1)
        other_token = self.other_tokenizer(other_states, other_valid).unsqueeze(1)
        map_token = self.map_tokenizer(map_features, map_valid).unsqueeze(1)
        tl_token = self.tl_tokenizer(traffic_light_features, traffic_light_valid).unsqueeze(1)

        return torch.cat([ego_token, goal_token, other_token, map_token, tl_token], dim=1)


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