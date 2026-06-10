from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from model.diffusion.diffusion_policy import DiffusionPolicy
from train_diffusion.utils.checkpoints import restore_checkpoint
from train_diffusion.utils.utils import coerce_tree_like
from data.types import PreprocessConfig


@dataclass
class InferenceBundle:
    graphdef: Any
    nonparam_state: Any
    params_state: Any
    preprocess_cfg: PreprocessConfig
    model_config: dict[str, Any]
    rng_key: jax.Array
    checkpoint_path: str


@dataclass
class PredictionBatch:
    start_t_b: jax.Array
    trajectories_world_bkt5: jax.Array
    world_t_seconds_bk: jax.Array
    world_t_valid_bk: jax.Array
    aux: dict[str, jax.Array]

_REQUIRED_META_KEYS = (
    "target_dim",
    "hidden_dim",
    "cond_dim",
    "map_attr_dim",
    "tl_attr_dim",
    "predict_horizon",
    "predict_type",
    "model_dt",
    "max_range",
    "ego_range",
    "max_velocity",
    "max_width",
    "max_tl_points",
    "num_map_type_classes",
)

def _load_checkpoint_metadata(
    checkpoint_path: str, metadata_path: str | None
) -> dict[str, Any]:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    meta = Path(metadata_path) if metadata_path is not None else ckpt / "metadata.json"
    if not meta.exists():
        raise ValueError(f"Metadata file not found: {meta}")

    with meta.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "pretrained_ckpt" in metadata:
        metadata_pretrained = _load_checkpoint_metadata(metadata["pretrained_ckpt"], None)
        metadata.update(metadata_pretrained)

    missing = [k for k in _REQUIRED_META_KEYS if k not in metadata]
    if missing:
        raise KeyError(f"Missing required metadata keys: {missing}")

    return metadata


def _build_preprocess_cfg(metadata: dict[str, Any]) -> PreprocessConfig:
    return PreprocessConfig(
        model_dt=float(metadata["model_dt"]),
        world_dt_fallback=0.1,
        max_range=float(metadata["max_range"]),
        ego_range=float(metadata["ego_range"]),
        max_velocity=float(metadata["max_velocity"]),
        max_width=float(metadata["max_width"]),
        max_tl_points=int(metadata["max_tl_points"]),
        num_map_type_classes=int(metadata["num_map_type_classes"]),
        predict_horizon=int(metadata["predict_horizon"]),
        map_unknown_type_index=20,
        max_segments=int(metadata["max_num_segments"]),
        max_points_per_segment=int(metadata["max_points_per_segment"]),
        inst_dim=int(metadata["inst_dim"]),
    )


def _build_model_from_metadata(metadata: dict[str, Any], seed: int) -> DiffusionPolicy:
    return DiffusionPolicy(
        target_dim=int(metadata["target_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        cond_dim=int(metadata["cond_dim"]),
        map_attr_dim=int(metadata["map_attr_dim"]),
        tl_attr_dim=int(metadata["tl_attr_dim"]),
        inst_attr_dim=int(metadata["inst_dim"]),
        predict_horizon=int(metadata["predict_horizon"]),
        predict_type=str(metadata["predict_type"]),
        rngs=nnx.Rngs(int(seed)),
    )

def load_model_for_inference(
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    metadata_path: str | None = None,
    seed: int = 0,
    skip_checkpoint_load: bool = False,
) -> InferenceBundle:
    metadata = _load_checkpoint_metadata(checkpoint_path, metadata_path)
    preprocess_cfg = _build_preprocess_cfg(metadata)
    model = _build_model_from_metadata(metadata, seed)

    graphdef, template_params_state, template_nonparam_state = nnx.split(
        model, nnx.Param, ...
    )
    params_state = template_params_state
    nonparam_state = template_nonparam_state

    if not skip_checkpoint_load:
        restored = restore_checkpoint(checkpoint_path)
        if "params_state" not in restored or "ema_params" not in restored:
            raise KeyError("Checkpoint is missing 'params_state' or 'ema_params'.")

        raw_params_state = restored["ema_params"] if use_ema else restored["params_state"]
        params_state = coerce_tree_like(template_params_state, raw_params_state)
        nonparam_state = coerce_tree_like(template_nonparam_state, restored.get("nonparam_state", template_nonparam_state))

    model_cfg = {
        "target_dim": int(metadata["target_dim"]),
        "predict_horizon": int(metadata["predict_horizon"]),
        "predict_type": str(metadata["predict_type"]),
    }

    return InferenceBundle(
        graphdef=graphdef,
        nonparam_state=nonparam_state,
        params_state=params_state,
        preprocess_cfg=preprocess_cfg,
        model_config=model_cfg,
        rng_key=jax.random.PRNGKey(int(seed)),
        checkpoint_path=checkpoint_path,
    )

