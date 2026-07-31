from __future__ import annotations

import dataclasses
import datetime
import json
import os
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model.diffusion.diffusion_policy import DiffusionPolicy
from data.types import PreprocessConfig


def configure_jax_compilation_cache(cache_dir: str) -> None:
    cache_dir = (cache_dir or "").strip()
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


def build_training_components(args):
    preprocess_cfg = PreprocessConfig(
        model_dt=args.model_dt,
        world_dt_fallback=0.1,
        max_range=args.max_range,
        ego_range=args.ego_range,
        max_velocity=args.max_velocity,
        max_width=args.max_width,
        max_tl_points=args.max_tl_points,
        num_map_type_classes=args.num_map_type_classes,
        predict_horizon=args.predict_horizon,
        max_segments=args.max_num_segments,
        max_points_per_segment=args.max_points_per_segment,
        num_object_types=args.num_object_types,
        inst_dim=args.inst_dim,
    )
    model = DiffusionPolicy(
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        map_attr_dim=args.map_attr_dim,
        tl_attr_dim=args.tl_attr_dim,
        predict_horizon=args.predict_horizon,
        predict_type=args.predict_type,
        rngs=nnx.Rngs(args.seed),
    )
    total_steps = max(1, args.epochs * args.steps_per_epoch)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=max(1, args.warmup_steps),
        decay_steps=max(args.warmup_steps + 1, total_steps),
        end_value=args.lr * 0.1,
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    return preprocess_cfg, model, tx, lr_schedule


def build_data_parallel_mesh_if_needed(args):
    if not args.data_parallel:
        return None
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if len(gpu_devices) <= 1:
        raise ValueError("--data_parallel was requested but <=1 GPU device is available.")
    if int(args.batch_size) % len(gpu_devices) != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by GPU count ({len(gpu_devices)}) for data parallel."
        )
    return Mesh(gpu_devices, ("data",))


def ema_update(ema_params, params, decay: float):
    return jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)


def clip_grads(grads, max_norm: float):
    norm = optax.global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
    return clipped, norm


def replicate_to_mesh(value, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, replicated) if hasattr(x, "shape") else x,
        value,
    )


def coerce_tree_like(template, raw):
    if hasattr(template, "_fields"):
        return type(template)(**{k: coerce_tree_like(getattr(template, k), raw[k]) for k in template._fields})
    if isinstance(template, tuple):
        return type(template)(coerce_tree_like(t, r) for t, r in zip(template, raw, strict=True))
    if isinstance(template, list):
        if isinstance(raw, dict):
            ordered_raw = [raw[str(i)] for i in range(len(template))]
        else:
            ordered_raw = raw
        return type(template)(coerce_tree_like(t, r) for t, r in zip(template, ordered_raw, strict=True))
    if isinstance(template, dict):
        if not isinstance(raw, dict):
            return raw

        coerced = {}
        for key, template_value in template.items():
            if key in raw:
                raw_value = raw[key]
            else:
                str_key = str(key)
                if str_key in raw:
                    raw_value = raw[str_key]
                elif isinstance(key, str) and key.isdigit() and int(key) in raw:
                    raw_value = raw[int(key)]
                else:
                    raise KeyError(
                        f"Missing key {key!r} in raw pytree; available keys: {list(raw.keys())[:10]}"
                    )
            coerced[key] = coerce_tree_like(template_value, raw_value)
        return coerced
    return raw


def make_dataset_config(args, waymax_config, global_batch_size: int):
    num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
    use_distributed = bool(args.data_parallel and num_gpus > 1)
    if use_distributed and global_batch_size % num_gpus != 0:
        raise ValueError(
            f"batch_size ({global_batch_size}) must be divisible by GPU count ({num_gpus}) for distributed loading."
        )
    batch_dims = (global_batch_size // num_gpus,) if use_distributed else (global_batch_size,)
    cfg_fields = {f.name for f in dataclasses.fields(waymax_config.WOD_1_3_1_TRAINING)}

    replace_kwargs = {
        "path": args.tfrecord_path,
        "max_num_objects": args.max_num_objects,
        "batch_dims": batch_dims,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
    }

    def _set_if_supported(field: str, value, *, when: bool = True):
        if when and field in cfg_fields:
            replace_kwargs[field] = value

    _set_if_supported("distributed", use_distributed)
    _set_if_supported(
        "tf_data_service_address",
        args.tf_data_service_address,
        when=bool(args.tf_data_service_address),
    )
    _set_if_supported("num_shards", int(args.dataset_num_shards))

    max_num_rg_points = int(args.dataset_max_num_rg_points)
    if "max_num_rg_points" in cfg_fields:
        if max_num_rg_points < 30000:
            raise ValueError(
                "dataset_max_num_rg_points < 30000 is invalid for WOMD TFExample parsing; "
                "use 30000 for this dataset."
            )
        replace_kwargs["max_num_rg_points"] = max_num_rg_points

    _set_if_supported("include_sdc_paths", bool(args.dataset_include_sdc_paths))
    return dataclasses.replace(waymax_config.WOD_1_3_1_TRAINING, **replace_kwargs)


def run_data_parallel_warmup_if_needed(model: DiffusionPolicy, args) -> None:
    if not args.data_parallel:
        return

    num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if num_gpus <= 1:
        return

    bsz = int(args.batch_size)
    if bsz % num_gpus != 0:
        raise ValueError(f"batch_size ({bsz}) must be divisible by GPU count ({num_gpus}).")
    per_dev = bsz // num_gpus
    n_other = args.max_num_objects
    horizon = args.predict_horizon
    features = {
        "ego_state": jnp.zeros((num_gpus, per_dev, 5), dtype=jnp.float32),
        "goal_xy": jnp.zeros((num_gpus, per_dev, 2), dtype=jnp.float32),
        "ego_trajectory": jnp.zeros((num_gpus, per_dev, horizon, args.target_dim), dtype=jnp.float32),
        "other_states": jnp.zeros((num_gpus, per_dev, n_other, 7), dtype=jnp.float32),
        "other_valid": jnp.ones((num_gpus, per_dev, n_other), dtype=jnp.bool_),
        "map_features": jnp.zeros((num_gpus, per_dev, args.max_map_points, args.map_attr_dim), dtype=jnp.float32),
        "map_valid": jnp.ones((num_gpus, per_dev, args.max_map_points), dtype=jnp.bool_),
        "traffic_light_features": jnp.zeros(
            (num_gpus, per_dev, args.max_tl_points, args.tl_attr_dim), dtype=jnp.float32
        ),
        "traffic_light_valid": jnp.ones((num_gpus, per_dev, args.max_tl_points), dtype=jnp.bool_),
    }
    with Mesh(gpu_devices, ("data",)):
        spec_b = P("data", None)
        spec_bt = P("data", None, None)
        spec_bnt = P("data", None, None, None)
        sharded = {
            "ego_state": jax.lax.with_sharding_constraint(features["ego_state"], spec_b),
            "goal_xy": jax.lax.with_sharding_constraint(features["goal_xy"], spec_b),
            "ego_trajectory": jax.lax.with_sharding_constraint(features["ego_trajectory"], spec_bt),
            "other_states": jax.lax.with_sharding_constraint(features["other_states"], spec_bnt),
            "other_valid": jax.lax.with_sharding_constraint(features["other_valid"], spec_bt),
            "map_features": jax.lax.with_sharding_constraint(features["map_features"], spec_bnt),
            "map_valid": jax.lax.with_sharding_constraint(features["map_valid"], spec_bt),
            "traffic_light_features": jax.lax.with_sharding_constraint(features["traffic_light_features"], spec_bnt),
            "traffic_light_valid": jax.lax.with_sharding_constraint(features["traffic_light_valid"], spec_bt),
        }
        sharded = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]), sharded)
        cond = model.compute_condition(sharded)
        target = sharded["ego_trajectory"][:, : model.predict_horizon, :]
        _ = model.loss_from_condition(target, cond, rng=jax.random.PRNGKey(args.seed), data_parallel=False)


def iter_with_prefetch(base_iter, prefetch_size: int):
    if prefetch_size <= 0:
        yield from base_iter
        return
    q = deque()
    for _ in range(prefetch_size):
        q.append(next(base_iter))
    while True:
        item = q.popleft()
        q.append(next(base_iter))
        yield item


def canonicalize_state_for_mesh(sim_state, mesh: Mesh | None):
    if mesh is None:
        return sim_state

    def canon_leaf(x):
        if not hasattr(x, "ndim"):
            return x
        spec = P() if x.ndim == 0 else P("data", *([None] * (x.ndim - 1)))
        return jax.device_put(x, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map(canon_leaf, sim_state)


def state_in_axes_from_template(sim_state):
    def _leaf_axis(x):
        if not hasattr(x, "ndim"):
            raise TypeError(f"SimulatorState contains non-array leaf type={type(x)}")
        if x.ndim == 0:
            raise TypeError("SimulatorState scalar leaves are not supported for pmap data axis.")
        return 0

    return jax.tree_util.tree_map(_leaf_axis, sim_state)


def sim_state_signature(sim_state):
    leaves, treedef = jax.tree_util.tree_flatten(sim_state)
    sig = []
    for i, x in enumerate(leaves):
        if not hasattr(x, "shape") or not hasattr(x, "dtype"):
            raise TypeError(f"SimulatorState leaf[{i}] is non-array type={type(x)}")
        sig.append((tuple(x.shape), str(x.dtype)))
    return treedef, tuple(sig)


def assert_matching_sim_state_signature(sim_state, ref_treedef, ref_sig, where: str):
    treedef, sig = sim_state_signature(sim_state)
    if treedef != ref_treedef:
        raise ValueError(f"{where}: SimulatorState tree structure changed; this can trigger recompilation.")
    if sig != ref_sig:
        for i, (a, b) in enumerate(zip(sig, ref_sig, strict=True)):
            if a != b:
                raise ValueError(
                    f"{where}: SimulatorState leaf[{i}] signature changed from {b} to {a}; "
                    "this can trigger recompilation."
                )
        raise ValueError(f"{where}: SimulatorState signature length changed; this can trigger recompilation.")


def init_wandb_run(args, config_dict):
    wandb_name = f"{args.wandb_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.abspath(os.path.join(args.save_dir, wandb_name))
    os.makedirs(run_dir, exist_ok=True)

    os.environ["WANDB_MODE"] = args.wandb_mode
    run = wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        dir=run_dir,
        entity=args.wandb_entity,
        config=config_dict,
    )
    return run, run_dir


def wandb_log_train_step(step: int, epoch: int, metrics, ema_loss, lr: float):
    loss_val = float(jnp.asarray(jax.device_get(metrics["loss"])).mean())
    grad_norm = float(jnp.asarray(jax.device_get(metrics["grad_norm"])).mean())

    if ema_loss is None:
        ema_loss = loss_val
    else:
        ema_loss = 0.99 * ema_loss + 0.01 * loss_val

    payload = {
        "step": step,
        "epoch": epoch,
        "train/loss": loss_val,
        "train/loss_ema": ema_loss,
        "train/grad_norm": grad_norm,
        "train/lr": lr,
    }
    def _log_value(value):
        arr = jnp.asarray(jax.device_get(value))
        return float(arr.mean()) if arr.ndim > 0 else float(arr)

    for key, value in metrics.items():
        if key in {"loss", "grad_norm"}:
            continue
        try:
            payload[f"train/{key}"] = _log_value(value)
        except Exception:
            continue

    wandb.log(payload, step=step)
    return ema_loss, payload


def log_jsonl(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def update_tqdm(pbar, step: int, pbar_every: int, metrics, ema_loss):
    if step % max(1, pbar_every) != 0:
        return ema_loss
    loss_val = float(jnp.asarray(jax.device_get(metrics["loss"])).mean())
    grad_norm = float(jnp.asarray(jax.device_get(metrics["grad_norm"])).mean())
    ade_m = float(metrics.get("traj_ade_m", 0.0))
    fde_m = float(metrics.get("traj_fde_m", 0.0))
    if ema_loss is None:
        ema_loss = loss_val
    else:
        ema_loss = 0.99 * ema_loss + 0.01 * loss_val
    pbar.set_postfix({"loss": f"{ema_loss:.4f}", "gn": f"{grad_norm:.2f}"})
    return ema_loss
