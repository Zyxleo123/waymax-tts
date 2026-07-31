from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor
from model.vla.modules.mlp import MLP
from model.vla.modules.attention import CrossAttentionLayers, SelfAttentionLayers
from model.vla.modules.scene_tokenizer import SceneTokenizer
from model.vla.modules.point_net import PointNet


class TemporalGemmaVLA(nn.Module):
    def __init__(
        self,
        gemma_name="google/gemma-4-E2B-it",
        ego_dim: int = 5,
        goal_dim: int = 3,
        other_dim: int = 15,
        map_dim: int = 25,
        tl_dim: int = 11,
        num_scene_tokens_per_step: int = 8,
        num_scene_tokens: int = 32,
        num_history_steps: int = 5,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_attn_layers: int = 4,
        predict_target_lane: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.ego_dim = int(ego_dim)
        self.goal_dim = int(goal_dim)
        self.other_dim = int(other_dim)
        self.map_dim = int(map_dim)
        self.tl_dim = int(tl_dim)
        self.num_scene_tokens_per_step = int(num_scene_tokens_per_step)
        self.num_scene_tokens = int(num_scene_tokens)
        self.num_history_steps = int(num_history_steps)
        self.hidden_dim = int(hidden_dim)
        self.predict_target_lane = bool(predict_target_lane)

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
            num_tokens=num_scene_tokens_per_step,
            num_heads=num_heads,
            num_attn_layers=num_attn_layers,
        )
        self.timestep_embedding = nn.Embedding(
            num_history_steps,
            token_dim,
        )
        self.temporal_cross_attn = CrossAttentionLayers(
            query_dim=token_dim,
            context_dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_attn_layers,
        )
        self.temporal_self_attn = SelfAttentionLayers(
            dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_attn_layers,
        )
        self.scene_token_queries = nn.Parameter(torch.zeros(1, num_scene_tokens + 1, token_dim))

        if self.predict_target_lane:
            self.lane_tokenizer = PointNet(map_dim, token_dim)
            self.target_lane_cross_attn = CrossAttentionLayers(
                query_dim=token_dim,
                context_dim=token_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_attn_layers,
            )
            self.target_lane_head = nn.Linear(token_dim, 1)

        self.subgoal_prediction = MLP([token_dim, hidden_dim, 2])

        self._scene_tokens_for_hook = None
        self._scene_start_for_hook = None
        self._scene_len_for_hook = None
        self._last_target_lane_pred_ids = None

        self._embedding_hook_handle = self.llm.get_input_embeddings().register_forward_hook(
            self._replace_scene_embeddings_hook
        )

    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_torch(self, value: Any, *, device: torch.device | None = None) -> Any:
        if device is None:
            device = self._module_device()
        if torch.is_tensor(value):
            return value.to(device=device) if device is not None else value
        if isinstance(value, (int, float, bool)):
            return torch.as_tensor(value, device=device)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, device=device)
        if hasattr(value, "__array__"):
            return torch.as_tensor(np.asarray(value), device=device)
        return value

    def _to_torch_feature_map(
        self,
        input_features: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        if device is None:
            device = self._module_device()
        return {
            key: self._to_torch(value, device=device)
            for key, value in input_features.items()
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TemporalGemmaVLA":
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
            "num_scene_tokens_per_step": int(pick("num_scene_tokens_per_step", 8)),
            "num_scene_tokens": int(pick("num_scene_tokens", 32)),
            "hidden_dim": int(pick("hidden_dim", 512)),
            "num_heads": int(pick("num_heads", 4)),
            "num_attn_layers": int(pick("num_attn_layers", 4)),
            "use_lora": bool(pick("use_lora", False)),
            "lora_r": int(pick("lora_r", 16)),
            "lora_alpha": int(pick("lora_alpha", 32)),
            "lora_dropout": float(pick("lora_dropout", 0.05)),
            "lora_target_modules": lora_target_modules,
            "predict_target_lane": bool(pick("predict_target_lane", False)),
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

    def _build_history_valid_mask(
        self,
        input_features: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        num_history_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        history_valid = self._to_torch(input_features.get("history_valid"), device=device)
        if history_valid is None:
            return torch.ones((batch_size, num_history_steps), dtype=torch.bool, device=device)
        history_valid = history_valid.to(device=device)
        if history_valid.dtype != torch.bool:
            history_valid = history_valid > 0
        if history_valid.ndim != 2:
            raise ValueError(
                f"history_valid must have shape [B, T], got {tuple(history_valid.shape)}"
            )
        if history_valid.shape[0] != batch_size or history_valid.shape[1] != num_history_steps:
            raise ValueError(
                f"history_valid shape mismatch: expected [{batch_size}, {num_history_steps}], "
                f"got {tuple(history_valid.shape)}"
            )
        return history_valid

    def _flatten_target_lane_ids(
        self,
        target_lane_ids: Any,
        *,
        batch_size: int,
        num_history_steps: int,
        used_temporal_batch: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        if target_lane_ids is None:
            return None

        if isinstance(target_lane_ids, torch.Tensor):
            lane_ids = target_lane_ids.to(device=device)
        else:
            lane_ids = torch.as_tensor(target_lane_ids, device=device)

        if lane_ids.numel() == 0:
            return None

        lane_ids = lane_ids.to(dtype=torch.long)
        if used_temporal_batch:
            if lane_ids.ndim == 2 and lane_ids.shape == (batch_size, num_history_steps):
                return lane_ids.reshape(-1)
            if lane_ids.ndim == 1 and lane_ids.shape[0] == batch_size * num_history_steps:
                return lane_ids
            if lane_ids.ndim == 1 and lane_ids.shape[0] == batch_size:
                return lane_ids.repeat_interleave(num_history_steps)
            raise ValueError(
                "target_lane_ids must have shape [B, T], [B*T], or [B] when using temporal batching, "
                f"got {tuple(lane_ids.shape)} with B={batch_size}, T={num_history_steps}."
            )

        if lane_ids.ndim == 2 and lane_ids.shape == (batch_size, num_history_steps):
            return lane_ids[:, -1]
        if lane_ids.ndim == 2 and lane_ids.shape[0] == batch_size and lane_ids.shape[1] == 1:
            return lane_ids[:, 0]
        if lane_ids.ndim == 1 and lane_ids.shape[0] == batch_size:
            return lane_ids
        raise ValueError(
            "target_lane_ids must have shape [B, T], [B, 1], or [B] when not using temporal batching, "
            f"got {tuple(lane_ids.shape)} with B={batch_size}, T={num_history_steps}."
        )

    def _tokenize_scene_features(self, input_features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ego_state = input_features["ego_state"]: [B, T, ego_dim]
        goal_xy = input_features["goal_xy"]: [B, T, goal_dim]
        remaining_timesteps = input_features["remaining_timesteps"]: [B, T, 1]
        other_states = input_features["other_states"]: [B, T, max_other_agents, other_dim]
        other_valid = input_features["other_valid"]: [B, T, max_other_agents]
        map_features = input_features["map_features"]: [B, T, num_map_segments, num_points_per_segment, map_dim]
        map_valid = input_features["map_valid"]: [B, T, num_map_segments]
        traffic_light_features = input_features["traffic_light_features"]: [B, T, num_traffic_lights, traffic_light_dim]
        traffic_light_valid = input_features["traffic_light_valid"]: [B, T, num_traffic_lights]
        """
        input_features = self._to_torch_feature_map(input_features)
        batch_size, num_history_steps = input_features["ego_state"].shape[:2]
        flatten_features = {
            "ego_state": input_features["ego_state"].reshape(batch_size * num_history_steps, input_features["ego_state"].shape[-1]),
            "goal_xy": input_features["goal_xy"].reshape(batch_size * num_history_steps, input_features["goal_xy"].shape[-1]),
            "remaining_timesteps": input_features["remaining_timesteps"].reshape(batch_size * num_history_steps, input_features["remaining_timesteps"].shape[-1]),
            "other_states": input_features["other_states"].reshape(batch_size * num_history_steps, *input_features["other_states"].shape[2:]),
            "other_valid": input_features["other_valid"].reshape(batch_size * num_history_steps, *input_features["other_valid"].shape[2:]),
            "map_features": input_features["map_features"].reshape(batch_size * num_history_steps, *input_features["map_features"].shape[2:]),
            "map_valid": input_features["map_valid"].reshape(batch_size * num_history_steps, *input_features["map_valid"].shape[2:]),
            "traffic_light_features": input_features["traffic_light_features"].reshape(batch_size * num_history_steps, *input_features["traffic_light_features"].shape[2:]),
            "traffic_light_valid": input_features["traffic_light_valid"].reshape(batch_size * num_history_steps, *input_features["traffic_light_valid"].shape[2:]),
        }
        per_step_tokens = self.scene_tokenizer(flatten_features, deterministic=True).reshape(
            batch_size,
            num_history_steps,
            -1,
            self.scene_tokenizer.token_dim,
        )
        timestep_emb = self.timestep_embedding(
            torch.arange(num_history_steps, device=per_step_tokens.device)
        ).unsqueeze(0).unsqueeze(2)
        per_step_tokens = per_step_tokens + timestep_emb

        history_valid = self._build_history_valid_mask(
            input_features,
            batch_size=batch_size,
            num_history_steps=num_history_steps,
            device=per_step_tokens.device,
        )
        context_tokens = per_step_tokens.reshape(batch_size, num_history_steps * self.num_scene_tokens_per_step, -1)
        context_valid = history_valid.repeat_interleave(self.num_scene_tokens_per_step, dim=1)

        timestep_scene_tokens: list[torch.Tensor] = []
        timestep_subgoal_tokens: list[torch.Tensor] = []
        for step_idx in range(num_history_steps):
            end_token_idx = (step_idx + 1) * self.num_scene_tokens_per_step
            step_scene_tokens = self.temporal_cross_attn(
                query_btc=self.scene_token_queries.expand(batch_size, -1, -1),
                context_bnc=context_tokens[:, :end_token_idx, :],
                context_mask_bn=context_valid[:, :end_token_idx],
            )
            step_scene_tokens = self.temporal_self_attn(step_scene_tokens)
            timestep_subgoal_tokens.append(step_scene_tokens[:, -1, :])
            timestep_scene_tokens.append(step_scene_tokens[:, :self.num_scene_tokens, :])

        scene_tokens = torch.stack(timestep_scene_tokens, dim=1)
        subgoal_tokens = torch.stack(timestep_subgoal_tokens, dim=1)
        return scene_tokens, subgoal_tokens

    def _reshape_temporal_text_inputs(
        self,
        *,
        scene_tokens: torch.Tensor,
        subgoal_tokens: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor | None,
        answer_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        bool,
    ]:
        if prompt_ids.ndim not in (2, 3):
            raise ValueError(f"prompt_ids must have rank 2 or 3, got {prompt_ids.ndim}")
        if prompt_mask.ndim != prompt_ids.ndim:
            raise ValueError("prompt_mask must have the same rank as prompt_ids")
        if answer_ids is not None and answer_ids.ndim != prompt_ids.ndim:
            raise ValueError("answer_ids must have the same rank as prompt_ids")
        if answer_mask is not None and answer_mask.ndim != prompt_ids.ndim:
            raise ValueError("answer_mask must have the same rank as prompt_ids")

        batch_size, num_history_steps, num_scene_tokens = scene_tokens.shape[:3]

        if prompt_ids.ndim == 3:
            if prompt_ids.shape[:2] != (batch_size, num_history_steps):
                raise ValueError(
                    "When prompt_ids is [B, T, P], its [B, T] shape must match scene tokens: "
                    f"expected [{batch_size}, {num_history_steps}], got {tuple(prompt_ids.shape[:2])}"
                )

            flat_scene_tokens = scene_tokens.reshape(batch_size * num_history_steps, num_scene_tokens, -1)
            flat_subgoal_tokens = subgoal_tokens.reshape(batch_size * num_history_steps, -1)
            flat_prompt_ids = prompt_ids.reshape(batch_size * num_history_steps, prompt_ids.shape[-1])
            flat_prompt_mask = prompt_mask.reshape(batch_size * num_history_steps, prompt_mask.shape[-1])
            flat_answer_ids = None
            flat_answer_mask = None
            if answer_ids is not None:
                flat_answer_ids = answer_ids.reshape(batch_size * num_history_steps, answer_ids.shape[-1])
            if answer_mask is not None:
                flat_answer_mask = answer_mask.reshape(batch_size * num_history_steps, answer_mask.shape[-1])
            return (
                flat_scene_tokens,
                flat_subgoal_tokens,
                flat_prompt_ids,
                flat_prompt_mask,
                flat_answer_ids,
                flat_answer_mask,
                True,
            )

        text_batch = prompt_ids.shape[0]
        if text_batch == batch_size * num_history_steps:
            flat_scene_tokens = scene_tokens.reshape(batch_size * num_history_steps, num_scene_tokens, -1)
            flat_subgoal_tokens = subgoal_tokens.reshape(batch_size * num_history_steps, -1)
            return (
                flat_scene_tokens,
                flat_subgoal_tokens,
                prompt_ids,
                prompt_mask,
                answer_ids,
                answer_mask,
                True,
            )

        if text_batch == batch_size:
            return (
                scene_tokens[:, -1, :, :],
                subgoal_tokens[:, -1, :],
                prompt_ids,
                prompt_mask,
                answer_ids,
                answer_mask,
                False,
            )

        raise ValueError(
            "prompt_ids first dimension must be B, B*T, or [B, T, P]. "
            f"Got prompt batch={text_batch}, B={batch_size}, T={num_history_steps}."
        )

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
        target_lane_ids: Any | None = None,
        compute_subgoal_loss: bool = False,
    ):
        device = self._module_device()
        input_features = self._to_torch_feature_map(input_features, device=device)
        prompt_ids = self._to_torch(prompt_ids, device=device)
        prompt_mask = self._to_torch(prompt_mask, device=device)
        if answer_ids is not None:
            answer_ids = self._to_torch(answer_ids, device=device)
        if answer_mask is not None:
            answer_mask = self._to_torch(answer_mask, device=device)
        scene_tokens, subgoal_tokens = self._tokenize_scene_features(input_features)
        

        (
            flat_scene_tokens,
            flat_subgoal_tokens,
            flat_prompt_ids,
            flat_prompt_mask,
            flat_answer_ids,
            flat_answer_mask,
            used_temporal_batch,
        ) = self._reshape_temporal_text_inputs(
            scene_tokens=scene_tokens,
            subgoal_tokens=subgoal_tokens,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            answer_ids=answer_ids,
            answer_mask=answer_mask,
        )
        
        device = flat_scene_tokens.device
        batch_size = flat_prompt_ids.shape[0]
        num_scene_tokens = flat_scene_tokens.shape[1]
        lane_loss = torch.zeros((), dtype=flat_scene_tokens.dtype, device=device)
        target_lane_accuracy = torch.zeros((), dtype=flat_scene_tokens.dtype, device=device)

        if self.predict_target_lane:
            B, H, N = input_features["lane_features"].shape[:3]
            lane_features = input_features["lane_features"].reshape(
                B * H * N, *input_features["lane_features"].shape[3:]
            )
            lane_valid = input_features["lane_valid"].reshape(
                B * H * N, *input_features["lane_valid"].shape[3:]
            )
            lane_ids = input_features["lane_ids"].reshape(
                B * H, *input_features["lane_ids"].shape[2:]
            )

            lane_tokens = self.lane_tokenizer(lane_features, lane_valid)  # [B*T*n_seg, token_dim]
            lane_context = self.target_lane_cross_attn(
                query_btc=lane_tokens.reshape(B * H, N, -1),
                context_bnc=flat_scene_tokens,
                context_mask_bn=torch.ones(
                    (B * H, num_scene_tokens),
                    dtype=torch.bool,
                    device=device,
                )
            )
            lane_logits = self.target_lane_head(lane_context).squeeze(-1)  # [B*T, n_seg]

            target_lane_ids_flat = self._flatten_target_lane_ids(
                target_lane_ids,
                batch_size=B,
                num_history_steps=H,
                used_temporal_batch=used_temporal_batch,
                device=device,
            )
            if target_lane_ids_flat is not None:
                if target_lane_ids_flat.ndim != 1:
                    target_lane_ids_flat = target_lane_ids_flat.reshape(-1)
                if int(target_lane_ids_flat.shape[0]) != int(lane_ids.shape[0]):
                    raise ValueError(
                        f"target_lane_ids length mismatch: expected {lane_ids.shape[0]}, got {target_lane_ids_flat.shape[0]}"
                    )

                lane_valid_flat = lane_valid.reshape(B * H, N, -1).any(dim=-1)
                target_match = lane_ids == target_lane_ids_flat[:, None]
                has_match = target_match.any(dim=-1) & lane_valid_flat.any(dim=-1)
                if bool(has_match.any()):
                    lane_logits_masked = lane_logits.masked_fill(~lane_valid_flat, -1e9)
                    lane_targets = target_match.long().argmax(dim=-1)
                    lane_loss = F.cross_entropy(
                        lane_logits_masked[has_match],
                        lane_targets[has_match],
                    )
                    lane_pred = lane_logits_masked.argmax(dim=-1)
                    target_lane_accuracy = (lane_pred[has_match] == lane_targets[has_match]).float().mean()
                else:
                    lane_loss = torch.zeros((), dtype=flat_scene_tokens.dtype, device=device)
                    target_lane_accuracy = torch.zeros((), dtype=flat_scene_tokens.dtype, device=device)


        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=self.tokenizer.pad_token_id,
            dtype=flat_prompt_ids.dtype,
            device=device,
        )
        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=flat_prompt_mask.dtype,
            device=device,
        )
        if flat_answer_ids is None:
            input_ids = torch.cat([scene_dummy_ids, flat_prompt_ids], dim=1)
            attention_mask = torch.cat([scene_mask, flat_prompt_mask], dim=1)
            labels = None
        else:
            input_ids = torch.cat(
                [scene_dummy_ids, flat_prompt_ids, flat_answer_ids],
                dim=1,
            )
            if flat_answer_mask is None:
                flat_answer_mask = (flat_answer_ids != self.tokenizer.pad_token_id).to(flat_prompt_mask.dtype)
            attention_mask = torch.cat(
                [scene_mask, flat_prompt_mask, flat_answer_mask],
                dim=1,
            )

            answer_labels = flat_answer_ids.masked_fill(flat_answer_mask == 0, -100)

            labels = torch.cat(
                [
                    torch.full(scene_dummy_ids.shape, -100, dtype=flat_answer_ids.dtype, device=device),
                    torch.full(flat_prompt_ids.shape, -100, dtype=flat_answer_ids.dtype, device=device),
                    answer_labels,
                ],
                dim=1,
            )
        self._scene_tokens_for_hook = flat_scene_tokens
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

        language_loss = outputs.loss
        if compute_subgoal_loss:
            subgoal_xy = self._to_torch(input_features["subgoal_xy"], device=device)
            if subgoal_xy.ndim == 3:
                if used_temporal_batch:
                    subgoal_xy = subgoal_xy.reshape(-1, subgoal_xy.shape[-1])
                else:
                    subgoal_xy = subgoal_xy[:, -1, :]
            subgoal_preds = self.subgoal_prediction(flat_subgoal_tokens)
            subgoal_loss = nn.functional.mse_loss(subgoal_preds, subgoal_xy, reduction="mean")
            loss = language_loss + subgoal_loss + lane_loss
        else:
            subgoal_loss = 0
            loss = language_loss + lane_loss

        return {
            "loss": loss,
            "subgoal_loss": subgoal_loss.detach().cpu() if isinstance(subgoal_loss, torch.Tensor) else subgoal_loss,
            "language_loss": language_loss.detach().cpu() if isinstance(language_loss, torch.Tensor) else language_loss,
            "target_lane_loss": lane_loss.detach().cpu(),
            "target_lane_accuracy": target_lane_accuracy.detach().cpu(),
        }

    def generate_predictions(
        self,
        input_features: Mapping[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_new_tokens: int = 8,
    ) -> tuple[list[str] | list[list[str]], Any]:
        device = self._module_device()
        input_features = self._to_torch_feature_map(input_features, device=device)
        prompt_ids = self._to_torch(prompt_ids, device=device)
        prompt_mask = self._to_torch(prompt_mask, device=device)
        scene_tokens, subgoal_tokens = self._tokenize_scene_features(input_features)
        (
            flat_scene_tokens,
            flat_subgoal_tokens,
            flat_prompt_ids,
            flat_prompt_mask,
            _,
            _,
            used_temporal_batch,
        ) = self._reshape_temporal_text_inputs(
            scene_tokens=scene_tokens,
            subgoal_tokens=subgoal_tokens,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            answer_ids=None,
            answer_mask=None,
        )
        device = flat_prompt_ids.device

        batch_size = flat_prompt_ids.shape[0]
        num_scene_tokens = flat_scene_tokens.shape[1]

        scene_dummy_ids = torch.full(
            (batch_size, num_scene_tokens),
            fill_value=self.tokenizer.pad_token_id,
            dtype=flat_prompt_ids.dtype,
            device=device,
        )
        scene_mask = torch.ones(
            (batch_size, num_scene_tokens),
            dtype=flat_prompt_mask.dtype,
            device=device,
        )
        input_ids = torch.cat(
            [scene_dummy_ids, flat_prompt_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [scene_mask, flat_prompt_mask],
            dim=1,
        )
        self._scene_tokens_for_hook = flat_scene_tokens
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
        output_texts = [text.strip() for text in decoded]

        subgoal_preds = self.subgoal_prediction(flat_subgoal_tokens)

        if prompt_ids.ndim == 3:
            batch_size, num_history_steps = scene_tokens.shape[:2]
            output_texts = [
                output_texts[i * num_history_steps : (i + 1) * num_history_steps]
                for i in range(batch_size)
            ]
            subgoal_preds = subgoal_preds.reshape(batch_size, num_history_steps, -1)

        target_lane_pred_ids = None
        if self.predict_target_lane:
            B, H, N = input_features["lane_features"].shape[:3]
            lane_features = input_features["lane_features"].reshape(
                B * H * N, *input_features["lane_features"].shape[3:]
            )
            lane_valid = input_features["lane_valid"].reshape(
                B * H * N, *input_features["lane_valid"].shape[3:]
            )
            lane_ids = input_features["lane_ids"].reshape(
                B * H, *input_features["lane_ids"].shape[2:]
            )

            lane_tokens = self.lane_tokenizer(lane_features, lane_valid)
            lane_context = self.target_lane_cross_attn(
                query_btc=lane_tokens.reshape(B * H, N, -1),
                context_bnc=flat_scene_tokens,
                context_mask_bn=torch.ones(
                    (B * H, num_scene_tokens),
                    dtype=torch.bool,
                    device=device,
                ),
            )
            lane_logits = self.target_lane_head(lane_context).squeeze(-1)

            lane_valid_flat = lane_valid.reshape(B * H, N, -1).any(dim=-1)
            lane_logits_masked = lane_logits.masked_fill(~lane_valid_flat, -1e9)
            lane_pred_idx = lane_logits_masked.argmax(dim=-1)

            target_lane_pred_ids = torch.gather(
                lane_ids,
                dim=1,
                index=lane_pred_idx.unsqueeze(-1),
            ).squeeze(-1)

            no_valid_lane = ~lane_valid_flat.any(dim=-1)
            if bool(no_valid_lane.any()):
                target_lane_pred_ids = target_lane_pred_ids.clone()
                target_lane_pred_ids[no_valid_lane] = -1

            if prompt_ids.ndim == 3:
                target_lane_pred_ids = target_lane_pred_ids.reshape(B, H)
            elif not used_temporal_batch:
                target_lane_pred_ids = target_lane_pred_ids.reshape(B, H)[:, -1]

        return {
            'answer': output_texts,
            'subgoal_preds': subgoal_preds.detach().cpu().numpy(),
            'target_lane_ids': target_lane_pred_ids.detach().cpu().numpy() if target_lane_pred_ids is not None else None
        }
