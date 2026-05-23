from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from model.torch_modules.mlp import MLP
from model.torch_modules.point_net import PointNet
from model.torch_modules.attention import CrossAttentionLayers
from model.torch_modules.scene_tokenizer import SceneTokenizer


class VecSceneGemmaVLA(nn.Module):
    def __init__(
        self,
        gemma_name="google/gemma-4-E2B-it",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 9,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     gemma_name,
        #     trust_remote_code=True,
        # )
        self.processor = AutoProcessor.from_pretrained(
            gemma_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.llm = AutoModelForCausalLM.from_pretrained(
            gemma_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # special_tokens = {"additional_special_tokens": ["<SCENE>"]}
        # num_added = self.tokenizer.add_special_tokens(special_tokens)
        # if num_added > 0:
        #     self.llm.resize_token_embeddings(len(self.tokenizer))
        # self.scene_token_id = self.tokenizer.convert_tokens_to_ids("<SCENE>")

        if hasattr(self.llm.config, "text_config"):
            token_dim = self.llm.config.text_config.hidden_size
        else:
            token_dim = self.llm.config.hidden_size
        # print(token_dim)
        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            other_dim=other_dim,
            map_attr_dim=map_dim,
            tl_attr_dim=tl_dim,
            hidden_dim=hidden_dim,
            cond_dim=token_dim,
            token_num=32,
        )

        self._scene_tokens_for_hook = None
        self._scene_start_for_hook = None
        self._scene_len_for_hook = None

        self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
            self._replace_scene_embeddings_hook
        )

    def _tokenize_scene_features(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.scene_tokenizer(input_features, deterministic=True)

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
                scene_tokens,
                output[:, length:, :],
            ],
            dim=1,
        )
    
    def _generation_eos_token_ids(self) -> int | list[int]:
        """EOS for generation: default eos_token_id, optionally plus newline."""
        eos_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        for tid in self.tokenizer.encode("\n", add_special_tokens=False):
            tid = int(tid)
            if tid not in eos_ids:
                eos_ids.append(tid)
        if not eos_ids:
            raise ValueError("Tokenizer has no eos_token_id for generation.")
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ):
        scene_tokens = self._tokenize_scene_features(input_features)
        device = scene_tokens.device

        batch_size = prompt_ids.shape[0]
        num_scene_tokens = scene_tokens.shape[1]

        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=self.tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )
        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=prompt_mask.dtype,
            device=device,
        )
        if answer_ids is None:
            input_ids = torch.cat([scene_dummy_ids, prompt_ids], dim=1)
            attention_mask = torch.cat([scene_mask, prompt_mask], dim=1)
            labels = None
        else:
            input_ids = torch.cat(
                [scene_dummy_ids, prompt_ids, answer_ids],
                dim=1,
            )
            attention_mask = torch.cat(
                [scene_mask, prompt_mask, answer_mask],
                dim=1,
            )
            if answer_mask is not None:
                answer_labels = answer_ids.masked_fill(answer_mask == 0, -100)
            else:
                answer_labels = answer_ids.masked_fill(
                    answer_ids == self.tokenizer.pad_token_id,
                    -100,
                )

            labels = torch.cat(
                [
                    torch.full(scene_dummy_ids.shape, -100, dtype=answer_ids.dtype, device=device),
                    torch.full(prompt_ids.shape, -100, dtype=answer_ids.dtype, device=device),
                    answer_labels,
                ],
                dim=1,
            )
        self._scene_tokens_for_hook = scene_tokens
        self._scene_start_for_hook = 0
        self._scene_len_for_hook = num_scene_tokens

        try:
            outputs = self.llm(
                input_ids=input_ids,
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

    def generate_predictions(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,  # [B, P]
        prompt_mask: torch.Tensor,  # [B, P]
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        device = prompt_ids.device
        scene_tokens = self._tokenize_scene_features(input_features)
        
        batch_size = prompt_ids.shape[0]
        num_scene_tokens = scene_tokens.shape[1]

        # dummy_id = getattr(self, ", None)
        # if dummy_id is None:
        #     raise ValueError("Scene token ID is not set. Make sure the tokenizer has the <SCENE> token and the model is initialized properly.")
        
        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=self.tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
            device=device,
        )
        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=prompt_mask.dtype,
            device=device,
        )
        input_ids = torch.cat(
            [scene_dummy_ids, prompt_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [scene_mask, prompt_mask],
            dim=1,
        )
        self._scene_tokens_for_hook = scene_tokens
        self._scene_start_for_hook = 0
        self._scene_len_for_hook = num_scene_tokens

        try:
            generated_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self._generation_eos_token_ids(),
            )
        finally:
            self._scene_tokens_for_hook = None
            self._scene_start_for_hook = None
            self._scene_len_for_hook = None

        input_len = input_ids.shape[1]
        new_token_ids = generated_ids[:, input_len:]
        decoded = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return [text.strip() for text in decoded]