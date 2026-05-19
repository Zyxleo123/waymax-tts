from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM

    GEMMA_MODEL_CLS = AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForImageTextToText

    GEMMA_MODEL_CLS = AutoModelForImageTextToText


class BEVGemmaQA(nn.Module):
    """
    BEV image QA model.

    Pipeline:
        BEV image
            -> Gemma4 vision_tower + embed_vision
            -> image soft tokens in Gemma text hidden space
            -> self-attention over image tokens
            -> MLP projection into Gemma hidden space
            -> dummy-token embedding hook
            -> Gemma LLM
            -> answer text
    """

    def __init__(
        self,
        gemma_name: str = "google/gemma-4-E2B-it",
        freeze_gemma: bool = True,
        freeze_vision: bool = True,
        num_attn_heads: int = 4,
    ) -> None:
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            gemma_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.llm = GEMMA_MODEL_CLS.from_pretrained(
            gemma_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.hidden_size = self.llm.get_input_embeddings().weight.shape[1]
        self.freeze_gemma_flag = bool(freeze_gemma)
        self.freeze_vision_flag = bool(freeze_vision)

        print(f"[BEVGemmaQA] hidden_size={self.hidden_size}")
        print(f"[BEVGemmaQA] vision_hidden_size={self.llm.config.vision_config.hidden_size}")
        print(f"[BEVGemmaQA] vision_soft_tokens_per_image={self.llm.config.vision_soft_tokens_per_image}")

        if freeze_gemma:
            for param in self.llm.parameters():
                param.requires_grad = False

        if freeze_vision:
            for param in self.llm.model.vision_tower.parameters():
                param.requires_grad = False
            if hasattr(self.llm.model, "embed_vision"):
                for param in self.llm.model.embed_vision.parameters():
                    param.requires_grad = False

        self.bev_self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=int(num_attn_heads),
            dropout=0.0,
            batch_first=True,
        )

        self.bev_self_attn_ln = nn.LayerNorm(self.hidden_size)

        self.bev_proj_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.bev_proj_mlp_ln = nn.LayerNorm(self.hidden_size)

        self._scene_tokens_for_hook: torch.Tensor | None = None
        self._scene_start_for_hook: int | None = None
        self._scene_len_for_hook: int | None = None

        self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
            self._replace_scene_embeddings_hook
        )

    def _replace_scene_embeddings_hook(self, module, inputs, output):
        """
        Avoid Gemma4 direct inputs_embeds path.

        We pass input_ids normally using dummy scene tokens. Then this hook replaces
        the dummy-token embeddings with our processed BEV scene tokens.
        """
        if self._scene_tokens_for_hook is None:
            return output

        start = self._scene_start_for_hook
        length = self._scene_len_for_hook

        if start is None or length is None:
            return output

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

    def _encode_bev_images(self, images: list[Any], device: torch.device) -> torch.Tensor:
        """
        Encodes BEV images using Gemma4's own vision encoder

        Returns:
            image_tokens: [B, N_img, hidden_size]
        """
        image_inputs = self.processor.image_processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = image_inputs["pixel_values"].to(
            device=device,
            dtype=next(self.llm.model.vision_tower.parameters()).dtype,
        )

        image_position_ids = image_inputs.get("image_position_ids", None)
        if image_position_ids is None:
            image_position_ids = image_inputs.get("pixel_position_ids", None)

        if image_position_ids is None:
            raise RuntimeError(
                f"Could not find image position ids. image_processor returned keys: {list(image_inputs.keys())}"
            )

        image_position_ids = image_position_ids.to(device=device)

        grad_enabled = not self.freeze_vision_flag

        with torch.set_grad_enabled(grad_enabled):
            try:
                vision_outputs = self.llm.model.get_image_features(
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                    return_dict=True,
                )
            except TypeError:
                vision_outputs = self.llm.model.get_image_features(
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                )

        if isinstance(vision_outputs, torch.Tensor):
            image_tokens = vision_outputs
        elif hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            image_tokens = vision_outputs.pooler_output
        elif hasattr(vision_outputs, "last_hidden_state"):
            image_tokens = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, tuple):
            image_tokens = vision_outputs[0]
        else:
            raise RuntimeError(f"Unknown image feature output type: {type(vision_outputs)}")

        if image_tokens.ndim == 2:
            batch_size = len(images)
            hidden = image_tokens.shape[-1]

            if image_tokens.shape[0] % batch_size != 0:
                raise RuntimeError(
                    f"Cannot reshape image tokens {tuple(image_tokens.shape)} with batch_size={batch_size}"
                )

            num_img_tokens = image_tokens.shape[0] // batch_size
            image_tokens = image_tokens.reshape(batch_size, num_img_tokens, hidden)

        if image_tokens.ndim != 3:
            raise RuntimeError(f"Expected image tokens [B, N, H], got {tuple(image_tokens.shape)}")

        if image_tokens.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Image hidden dim {image_tokens.shape[-1]} != LLM hidden dim {self.hidden_size}"
            )

        return image_tokens

    def _project_bev_tokens(self, image_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.bev_self_attn(
            query=image_tokens,
            key=image_tokens,
            value=image_tokens,
        )

        h = self.bev_self_attn_ln(image_tokens + attended)
        projected = self.bev_proj_mlp(h)
        return self.bev_proj_mlp_ln(h + projected)

    def _tokenize_text(
        self,
        questions: list[str],
        answers: list[str] | None,
        device: torch.device,
        max_prompt_length: int,
        max_answer_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        prompt_batch = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=True,
        )

        prompt_ids = prompt_batch["input_ids"].to(device)
        prompt_mask = prompt_batch["attention_mask"].to(device)

        if answers is None:
            return prompt_ids, prompt_mask, None, None

        answer_batch = self.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_answer_length,
            add_special_tokens=False,
        )

        answer_ids = answer_batch["input_ids"].to(device)
        answer_mask = answer_batch["attention_mask"].to(device)

        return prompt_ids, prompt_mask, answer_ids, answer_mask

    def forward(
        self,
        images: list[Any],
        questions: list[str],
        answers: list[str] | None = None,
        max_prompt_length: int = 128,
        max_answer_length: int = 8,
    ) -> dict[str, torch.Tensor | None]:
        device = next(self.parameters()).device

        prompt_ids, prompt_mask, answer_ids, answer_mask = self._tokenize_text(
            questions=questions,
            answers=answers,
            device=device,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_answer_length,
        )

        image_tokens = self._encode_bev_images(images, device=device)
        scene_tokens = self._project_bev_tokens(image_tokens)

        batch_size = prompt_ids.shape[0]
        num_scene_tokens = scene_tokens.shape[1]

        dummy_id = self.tokenizer.pad_token_id
        if dummy_id is None:
            dummy_id = self.tokenizer.eos_token_id

        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=dummy_id,
            dtype=prompt_ids.dtype,
            device=device,
        )

        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=prompt_mask.dtype,
            device=device,
        )

        if answer_ids is not None:
            input_ids = torch.cat([prompt_ids, scene_dummy_ids, answer_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, scene_mask, answer_mask], dim=1)

            labels = torch.cat(
                [
                    torch.full(prompt_ids.shape, -100, dtype=prompt_ids.dtype, device=device),
                    torch.full(scene_dummy_ids.shape, -100, dtype=prompt_ids.dtype, device=device),
                    answer_ids,
                ],
                dim=1,
            )

            labels[:, -answer_ids.shape[1] :] = labels[:, -answer_ids.shape[1] :].masked_fill(
                answer_mask == 0,
                -100,
            )
        else:
            input_ids = torch.cat([prompt_ids, scene_dummy_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, scene_mask], dim=1)
            labels = None

        self._scene_tokens_for_hook = scene_tokens
        self._scene_start_for_hook = prompt_ids.shape[1]
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

        return {
            "loss": outputs.loss if labels is not None else None,
        }

    @torch.no_grad()
    def generate_answers(
        self,
        images: list[Any],
        questions: list[str],
        max_prompt_length: int = 128,
        max_new_tokens: int = 8,
    ) -> list[str]:
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        prompt_ids, prompt_mask, _, _ = self._tokenize_text(
            questions=questions,
            answers=None,
            device=device,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_new_tokens,
        )

        image_tokens = self._encode_bev_images(images, device=device)
        scene_tokens = self._project_bev_tokens(image_tokens)

        batch_size = prompt_ids.shape[0]
        num_scene_tokens = scene_tokens.shape[1]

        dummy_id = self.tokenizer.pad_token_id
        if dummy_id is None:
            dummy_id = self.tokenizer.eos_token_id

        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=dummy_id,
            dtype=prompt_ids.dtype,
            device=device,
        )

        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=prompt_mask.dtype,
            device=device,
        )

        input_ids = torch.cat([prompt_ids, scene_dummy_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, scene_mask], dim=1)

        self._scene_tokens_for_hook = scene_tokens
        self._scene_start_for_hook = prompt_ids.shape[1]
        self._scene_len_for_hook = num_scene_tokens

        try:
            generated_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
        finally:
            self._scene_tokens_for_hook = None
            self._scene_start_for_hook = None
            self._scene_len_for_hook = None

        input_len = input_ids.shape[1]
        new_token_ids = generated_ids[:, input_len:]
        decoded = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        if was_training:
            self.train()

        return [text.strip() for text in decoded]
