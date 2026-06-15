from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from model.vla.modules.mlp import MLP
from model.vla.modules.point_net import PointNet
from model.vla.modules.attention import CrossAttentionLayers, SelfAttentionLayers
from model.vla.modules.scene_tokenizer import SceneTokenizer


class VecSceneGemmaVLA(nn.Module):
    def __init__(
        self,
        gemma_name="google/gemma-4-E2B-it",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 11,
        num_scene_tokens: int = 32,
        hidden_dim: int = 512,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.num_scene_tokens = num_scene_tokens

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

        self._has_lora = False
        if use_lora:
            self._attach_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

        if hasattr(self.llm.config, "text_config"):
            token_dim = self.llm.config.text_config.hidden_size
        else:
            token_dim = self.llm.config.hidden_size
        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            goal_dim=goal_dim,
            other_dim=other_dim,
            map_attr_dim=map_dim,
            tl_attr_dim=tl_dim,
            hidden_dim=hidden_dim,
            cond_dim=token_dim,
            num_tokens=num_scene_tokens + 1,
        )
        self.scene_token_self_attention = SelfAttentionLayers(
            dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=4,
        )
        self.subgoal_prediction_head = MLP([token_dim, hidden_dim, 2])

        self._scene_tokens_for_hook = None
        self._scene_start_for_hook = None
        self._scene_len_for_hook = None

        self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
            self._replace_scene_embeddings_hook
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "VecSceneGemmaVLA":
        run_dir, resolved_checkpoint = cls._resolve_run_dir_and_checkpoint(Path(checkpoint_path))
        run_config = cls._load_training_config(run_dir)
        model_kwargs = cls._build_init_kwargs_from_config(run_config)

        model = cls(**model_kwargs)

        checkpoint = torch.load(resolved_checkpoint, map_location=map_location)
        state_dict = cls._strip_module_prefix(checkpoint.get("model_state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"run_dir: {run_dir}")
        print(f"checkpoint: {resolved_checkpoint}")
        print(f"missing keys: {len(missing)}")
        print(f"unexpected keys: {len(unexpected)}")
        if missing:
            print("  missing:", missing[:10])
        if unexpected:
            print("  unexpected:", unexpected[:10])

        if device is not None or dtype is not None:
            if device is None:
                device = next(model.parameters()).device
            if dtype is None:
                dtype = next(model.parameters()).dtype
            model = model.to(device=device, dtype=dtype)

        model.eval()
        return model

    @staticmethod
    def _resolve_run_dir_and_checkpoint(path: Path) -> tuple[Path, Path]:
        path = path.expanduser().resolve()
        if path.is_file():
            if path.parent.name == "checkpoints":
                return path.parent.parent, path
            return path.parent, path
        if path.is_dir():
            if path.name == "checkpoints":
                checkpoint_files = sorted(path.glob("step_*.pt"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoints found in {path}")
                return path.parent, checkpoint_files[-1]
            checkpoint_dir = path / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = sorted(checkpoint_dir.glob("step_*.pt"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
                return path, checkpoint_files[-1]
        raise FileNotFoundError(f"Could not resolve run directory from: {path}")

    @staticmethod
    def _load_training_config(run_dir: Path) -> dict[str, Any]:
        config_path = run_dir / "training_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"training_config.json not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _build_init_kwargs_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
        def pick(name: str, default: Any) -> Any:
            return config.get(name, default)

        lora_target_modules = pick("lora_target_modules", None)
        if isinstance(lora_target_modules, str) and lora_target_modules.strip():
            lora_target_modules = tuple(part.strip() for part in lora_target_modules.split(",") if part.strip())
        elif isinstance(lora_target_modules, (list, tuple)):
            lora_target_modules = tuple(str(part) for part in lora_target_modules)
        else:
            lora_target_modules = None

        return {
            "gemma_name": pick("gemma_name", "google/gemma-4-E2B-it"),
            "ego_dim": int(pick("ego_dim", 5)),
            "goal_dim": int(pick("goal_dim", 3)),
            "other_dim": int(pick("other_dim", 15)),
            "map_dim": int(pick("map_dim", 25)),
            "tl_dim": int(pick("tl_dim", 11)),
            "num_scene_tokens": int(pick("num_scene_tokens", 32)),
            "hidden_dim": int(pick("hidden_dim", 512)),
            "use_lora": bool(pick("use_lora", False)),
            "lora_r": int(pick("lora_r", 16)),
            "lora_alpha": int(pick("lora_alpha", 32)),
            "lora_dropout": float(pick("lora_dropout", 0.05)),
            "lora_target_modules": lora_target_modules,
        }

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
        if not any(key.startswith("module.") for key in state_dict):
            return state_dict
        return {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    def _load_peft_components(self):
        try:
            peft = importlib.import_module("peft")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "peft is required for LoRA training. Install it with `pip install peft`."
            ) from exc
        return peft.LoraConfig, peft.TaskType, peft.get_peft_model

    def _attach_lora(
        self,
        *,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: tuple[str, ...] | None,
    ) -> None:
        LoraConfig, TaskType, get_peft_model = self._load_peft_components()

        if target_modules is None:
            target_modules = (
                "q_proj.linear",
                "k_proj.linear",
                "v_proj.linear",
                "o_proj.linear",
                "gate_proj.linear",
                "up_proj.linear",
                "down_proj.linear",
            )
        else:
            normalized_modules: list[str] = []
            for module_name in target_modules:
                normalized_modules.append(module_name)
                if "." not in module_name:
                    normalized_modules.append(f"{module_name}.linear")
            target_modules = tuple(dict.fromkeys(normalized_modules))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(target_modules),
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self._has_lora = True

    def _tokenize_scene_features(self, input_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scene_tokens = self.scene_tokenizer(input_features, deterministic=True)
        scene_tokens = self.scene_token_self_attention(scene_tokens)
        scene_tokens = scene_tokens[:, : self.num_scene_tokens, :]
        subgoal_token = scene_tokens[:, -1, :]
        return scene_tokens, subgoal_token

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
        if self._has_lora:
            for name, param in self.llm.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False
        else:
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
        predict_subgoal: bool = True,
    ):
        scene_tokens, subgoal_token = self._tokenize_scene_features(input_features)
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

        if predict_subgoal:
            subgoal_pred = self.subgoal_prediction_head(subgoal_token)
            subgoal_xy = input_features["subgoal_xy"]
            subgoal_loss = nn.functional.mse_loss(subgoal_pred, subgoal_xy, reduction="mean")
            loss = outputs.loss + subgoal_loss
        else:
            loss = outputs.loss

        return {
            "loss": loss,
            "language_loss": outputs.loss,
            "subgoal_loss": subgoal_loss if predict_subgoal else 0,
        }

    def generate_predictions(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,  # [B, P]
        prompt_mask: torch.Tensor,  # [B, P]
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        device = prompt_ids.device
        scene_tokens, subgoal_token = self._tokenize_scene_features(input_features)

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

        subgoal_pred = self.subgoal_prediction_head(subgoal_token)
        return {
            "answer": [text.strip() for text in decoded],
            "subgoal_pred": subgoal_pred.detach().cpu().numpy(),
        }