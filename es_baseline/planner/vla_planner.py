from __future__ import annotations

from typing import Any, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch

# from data.inst_dataloader import INST_SUBGOAL_PROMPT
from data.utils import VLA_PROMPT
from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessBatch, PreprocessConfig
from model.diffusion.diffusion_policy import DiffusionPolicy
from model.vla.gemma_vla import VecSceneGemmaVLA
from model.vla.embedding_gemma_encoder import EmbeddingGemmaEncoder
from planner.abstract_planner import AbstractPlanner, PlannerResult


@dataclass
class VLAPlannerResult(PlannerResult):
    instruction_texts: Sequence[str]
    subgoal_preds: Sequence[Any] | None = None



class VLAPlanner(AbstractPlanner):
    def __init__(
        self,
        policy: DiffusionPolicy,
        gemma_vla: VecSceneGemmaVLA,
        instruction_encoder: EmbeddingGemmaEncoder,
        preprocess_cfg: PreprocessConfig,
        population_size: int = 1,
        num_worlds: int = 1,
        *,
        torch_device: str | torch.device | None = None,
        jax_device: Any | None = None,
        prompt_text: str = VLA_PROMPT,
        max_new_tokens: int = 64,
        dummy_instruction: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.policy = policy
        self.gemma_vla = gemma_vla
        self.instruction_encoder = instruction_encoder
        self.preprocess_cfg = preprocess_cfg
        self.num_worlds = int(num_worlds)
        self.population_size = int(population_size)
        self.prompt_text = prompt_text
        self.max_new_tokens = int(max_new_tokens)
        self.inst_feature_dim = int(self.policy.instruction_encoder.input_dim)
        self.dummy_instruction = dummy_instruction

        if torch_device is None:
            if torch.cuda.is_available():
                torch_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
            else:
                torch_device = torch.device("cpu")
        self.torch_device = torch.device(torch_device)
        self.gemma_vla = self.gemma_vla.to(self.torch_device)
        self.gemma_vla.eval()

        if jax_device is None:
            gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
            jax_device = gpu_devices[0] if gpu_devices else jax.devices()[0]
        self.jax_device = jax_device

        self._compute_condition_jit = jax.jit(self.policy.compute_condition)
        self._sample_population_jit = jax.jit(
            lambda *, cond_bf, inst_cond_bf, inst_cond_mask_bf, rng: self._sample_population(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=inst_cond_mask_bf,
                rng=rng,
            ),
        )

    def plan_trajectory(
        self,
        sim_state,
        goal,
        *,
        rng: jax.Array,
        timestep: int = 0,
        mask_goal: bool = False,
        instruction_texts: list[str] | None = None,
        use_subgoal: bool = False,
        **kwargs: Any,
    ) -> VLAPlannerResult:
        rng, key_pre, key_sample = jax.random.split(rng, 3)
        pre_batch, _ = preprocess_simulator_state(
            sim_state,
            key_pre,
            self.preprocess_cfg,
            anchor_step_override=timestep,
            goal_step_override=90,
            goal_xy_override=goal,
        )
        features = pre_batch.features
        prompt_ids, prompt_mask = self._build_prompt_batch(
            batch_size=int(pre_batch.features["ego_state"].shape[0])
        )
        torch_features = self._move_features_to_torch(pre_batch.features, device=self.torch_device)

        if self.dummy_instruction:
            instruction_texts = ["Go to the left to turn left."] * len(prompt_ids)
        elif instruction_texts is None:
            with torch.inference_mode():
                amp_enabled = self.torch_device.type == "cuda"
                with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    predictions = self.gemma_vla.generate_predictions(
                        input_features=torch_features,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        max_new_tokens=self.max_new_tokens,
                    )
        instruction_texts = predictions["answer"]
        subgoal_preds = predictions.get("subgoal_pred", None)
        instruction_features, instruction_mask = self.encode_instructions(
            instruction_texts=instruction_texts
        )

        features = dict(pre_batch.features)
        features["inst_features"] = instruction_features
        features["inst_valid"] = instruction_mask
        if mask_goal:
            features["goal_xy"] = jnp.zeros_like(features["goal_xy"])
            features["remaining_timesteps"] = jnp.zeros_like(features["remaining_timesteps"])
        if use_subgoal and subgoal_preds is not None:
            features["subgoal_xy"] = jnp.asarray(subgoal_preds, dtype=jnp.float32) / 100.0
            features["subgoal_valid"] = jnp.ones((self.num_worlds,), dtype=jnp.float32)
        else:
            features["subgoal_xy"] = jnp.zeros((self.num_worlds, 2), dtype=jnp.float32)
            features["subgoal_valid"] = jnp.zeros((self.num_worlds,), dtype=jnp.float32)

        with jax.default_device(self.jax_device):
            cond_bf, inst_cond_bf = self._compute_condition_jit(features)
            current_norm_bktd = self._sample_population_jit(
                cond_bf=cond_bf,
                inst_cond_bf=inst_cond_bf,
                inst_cond_mask_bf=instruction_mask,
                rng=key_sample,
            )

        current_world_bkt5, world_t_seconds_bt, world_t_valid_bt = self._postprocess_population(
            current_norm_bktd,
            pre_batch,
        )
        best_indices = jnp.zeros((cond_bf.shape[0], 1), dtype=jnp.int32)
        best_norm_bt1d = self._gather_population(current_norm_bktd, best_indices)
        best_world_bt15 = self._gather_population(current_world_bkt5, best_indices)

        return VLAPlannerResult(
            start_t_b=pre_batch.aux["anchor_step"].astype(jnp.int32),
            trajectory_norm_btd=jnp.squeeze(best_norm_bt1d, axis=1),
            trajectory_world_bt5=jnp.squeeze(best_world_bt15, axis=1),
            world_t_seconds_bt=world_t_seconds_bt,
            world_t_valid_bt=world_t_valid_bt,
            aux=pre_batch.aux,
            instruction_texts=instruction_texts,
            subgoal_preds=subgoal_preds.tolist() if subgoal_preds is not None else None,
        )

    def _build_prompt_batch(
        self,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [self.prompt_text] * batch_size
        tokenized = self.gemma_vla.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenized["input_ids"].to(self.torch_device)
        prompt_mask = tokenized["attention_mask"].to(self.torch_device)
        return prompt_ids, prompt_mask

    def _move_features_to_torch(
        self,
        features: Mapping[str, jax.Array],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for key, value in features.items():
            array = np.asarray(value)
            tensor = torch.as_tensor(array, device=device)
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype=torch.float32)
            moved[key] = tensor
        return moved

    def encode_instructions(
        self,
        *,
        instruction_texts: Sequence[str],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = len(instruction_texts)
        inst_features = self.instruction_encoder.encode(instruction_texts, batch_size=batch_size)
        inst_mask = np.ones((batch_size,), dtype=bool)
        return jnp.asarray(inst_features, dtype=jnp.float32), jnp.asarray(inst_mask)
                    

    def _sample_population(
        self,
        *,
        cond_bf: jax.Array,
        inst_cond_bf: jax.Array,
        inst_cond_mask_bf: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        batch_size = int(cond_bf.shape[0])
        cond_bkf = jnp.repeat(cond_bf, repeats=self.population_size, axis=0)
        inst_cond_bf = jnp.repeat(inst_cond_bf, repeats=self.population_size, axis=0)
        inst_cond_mask_bf = jnp.repeat(inst_cond_mask_bf, repeats=self.population_size, axis=0)
        samples_btd = self.policy.sample_from_condition(cond_bkf, inst_cond_bf, inst_cond_mask_bf, rng=rng)
        horizon = int(samples_btd.shape[1])
        target_dim = int(samples_btd.shape[2])
        return samples_btd.reshape(batch_size, self.population_size, horizon, target_dim)

    def _postprocess_population(
        self,
        proposals_bktd: jax.Array,
        pre_batch: PreprocessBatch,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size, population_size, horizon, target_dim = proposals_bktd.shape
        flat_proposals = proposals_bktd.reshape(batch_size * population_size, horizon, target_dim)
        repeated_aux = self._repeat_dict_leading_axis(pre_batch.aux, repeats=population_size)
        post = postprocess_predictions(flat_proposals, repeated_aux, self.preprocess_cfg)

        world_flat = post["trajectory_world_world_dt"]
        world_steps = int(world_flat.shape[1])
        world_dim = int(world_flat.shape[2])
        trajectories_world_bkt5 = world_flat.reshape(
            batch_size,
            population_size,
            world_steps,
            world_dim,
        )
        world_t_seconds_bt = jnp.asarray(pre_batch.aux["world_t_seconds"], dtype=jnp.float32)
        world_t_valid_bt = jnp.asarray(pre_batch.aux["world_t_valid"])
        return trajectories_world_bkt5, world_t_seconds_bt, world_t_valid_bt

    @staticmethod
    def _gather_population(population_bk: jax.Array, indices_bm: jax.Array) -> jax.Array:
        gather_indices = jnp.asarray(indices_bm, dtype=jnp.int32)
        trailing_shape = population_bk.shape[2:]
        reshape_shape = gather_indices.shape + (1,) * len(trailing_shape)
        broadcast_shape = gather_indices.shape + trailing_shape
        gather_indices = jnp.broadcast_to(
            gather_indices.reshape(reshape_shape),
            broadcast_shape,
        )
        return jnp.take_along_axis(population_bk, gather_indices, axis=1)

    @staticmethod
    def _repeat_dict_leading_axis(
        values: dict[str, jax.Array],
        *,
        repeats: int,
    ) -> dict[str, jax.Array]:
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x), repeats=repeats, axis=0),
            values,
        )


__all__ = ["VLAPlanner"]