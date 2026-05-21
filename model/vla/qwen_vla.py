from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.torch_modules.mlp import MLP
from model.torch_modules.point_net import PointNet
from model.torch_modules.attention import CrossAttentionLayers
from model.torch_modules.scene_tokenizer import SceneTokenizer


class VecSceneQwenVLA(nn.Module):
    def __init__(
        self,
        qwen_name="Qwen/Qwen3-0.6B",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 9,
        hidden_dim: int = 512,
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

        token_dim = self.llm.config.hidden_size

        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            other_dim=other_dim,
            map_attr_dim=map_dim,
            tl_attr_dim=tl_dim,
            hidden_dim=hidden_dim,
            cond_dim=token_dim,
            token_num=32,
        )

    def _tokenize_scene_features(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.scene_tokenizer(input_features)

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
                [scene_tokens, text_emb, answer_emb],
                dim=1,
            )

            labels = torch.cat(
                [
                    torch.full(scene_tokens.shape[:2], -100, device=device),
                    torch.full(prompt_ids.shape, -100, device=device),
                    answer_ids,
                ],
                dim=1,
            )
        else:
            inputs_embeds = torch.cat(
                [scene_tokens, text_emb],
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

    def generate_predictions(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,  # [B, P]
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        device = prompt_ids.device
        scene_tokens = self._tokenize_scene_features(input_features)
        text_emb = self.llm.get_input_embeddings()(prompt_ids)
        scene_tokens = scene_tokens.to(text_emb.dtype)

        inputs_embeds = torch.cat([scene_tokens, text_emb], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        input_len = inputs_embeds.shape[1]
        if generated_ids.shape[1] > input_len:
            new_token_ids = generated_ids[:, input_len:]
        else:
            new_token_ids = generated_ids
        decoded = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return decoded
