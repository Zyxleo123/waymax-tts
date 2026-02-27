from __future__ import annotations

import dataclasses
import datetime
import gzip
import glob
import json
import os
from typing import Any

# Work around NCCL cuMem allocator failures seen on this host for multi-GPU collectives.
os.environ["NCCL_CUMEM_ENABLE"] = "0"

import jax
import jax.numpy as jnp
import jax.profiler
import optax
import wandb
from flax import nnx
from jax.sharding import Mesh
from tqdm import tqdm

from model.diffusion_policy import DiffusionPolicy
from train.checkpoints import restore_checkpoint, save_checkpoint
from train.config import config_to_dict, parse_args
from train.preprocess import preprocess_simulator_state
from train.types import PreprocessConfig


def _build_preprocess_config(args) -> PreprocessConfig:
    return PreprocessConfig(
        model_dt=args.model_dt,
        world_dt_fallback=0.1,
        max_range=args.max_range,
        ego_range=args.ego_range,
        max_velocity=args.max_velocity,
        max_width=args.max_width,
        max_map_points=args.max_map_points,
        max_tl_points=args.max_tl_points,
        num_map_type_classes=args.num_map_type_classes,
        predict_horizon=args.predict_horizon,
    )


def _build_optimizer(args):
    total_steps = max(1, args.epochs * args.steps_per_epoch)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=max(1, args.warmup_steps),
        decay_steps=max(args.warmup_steps + 1, total_steps),
        end_value=args.lr * 0.1,
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    return tx, schedule


def _build_data_parallel_mesh_if_needed(args):
    if not args.data_parallel:
        return None
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if len(gpu_devices) <= 1:
        raise ValueError("--data_parallel was requested but <=1 GPU device is available.")
    return Mesh(gpu_devices, ("data",))


def _resolve_global_batch_size(args) -> int:
    if not args.data_parallel:
        return int(args.batch_size)
    num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
    if num_gpus <= 1:
        raise ValueError("--data_parallel was requested but <=1 GPU device is available.")
    # In data-parallel mode, `batch_size` is per-device batch size.
    return int(args.batch_size) * num_gpus


def _ema_update(ema_params, params, decay: float):
    return jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)


def _clip_grads(grads, max_norm: float):
    norm = optax.global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
    return clipped, norm


def _build_model(args, seed: int) -> DiffusionPolicy:
    return DiffusionPolicy(
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        map_attr_dim=args.map_attr_dim,
        tl_attr_dim=args.tl_attr_dim,
        predict_horizon=args.predict_horizon,
        predict_type=args.predict_type,
        rngs=nnx.Rngs(seed),
    )


def _is_optarray_leaf(x):
    t = type(x)
    return hasattr(x, "value") and t.__module__.startswith("flax.nnx.training.optimizer")


def _unwrap_opt_state(tree):
    return jax.tree_util.tree_map(
        lambda x: x.value if _is_optarray_leaf(x) else x,
        tree,
        is_leaf=_is_optarray_leaf,
    )


def _rewrap_opt_state(template, raw_tree):
    return jax.tree_util.tree_map(
        lambda t, v: type(t)(v) if _is_optarray_leaf(t) else v,
        template,
        raw_tree,
        is_leaf=_is_optarray_leaf,
    )


def _make_dataset_config(args, waymax_config, global_batch_size: int):
    cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=args.tfrecord_path,
        max_num_objects=args.max_num_objects,
        batch_dims=(global_batch_size,),
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    return cfg


def _run_data_parallel_warmup_if_needed(model: DiffusionPolicy, args) -> None:
    if not args.data_parallel:
        return
    if not model.supports_data_parallel():
        return

    num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
    if num_gpus <= 1:
        return

    bsz = _resolve_global_batch_size(args)
    n_other = args.max_num_objects
    horizon = args.predict_horizon
    features = {
        "ego_state": jnp.zeros((bsz, 5), dtype=jnp.float32),
        "ego_trajectory": jnp.zeros((bsz, horizon, args.target_dim), dtype=jnp.float32),
        "other_states": jnp.zeros((bsz, n_other, 7), dtype=jnp.float32),
        "other_valid": jnp.ones((bsz, n_other), dtype=jnp.bool_),
        "map_features": jnp.zeros((bsz, args.max_map_points, args.map_attr_dim), dtype=jnp.float32),
        "map_valid": jnp.ones((bsz, args.max_map_points), dtype=jnp.bool_),
        "traffic_light_features": jnp.zeros((bsz, args.max_tl_points, args.tl_attr_dim), dtype=jnp.float32),
        "traffic_light_valid": jnp.ones((bsz, args.max_tl_points), dtype=jnp.bool_),
    }
    _ = model.loss(features, rng=jax.random.PRNGKey(args.seed), data_parallel=True)


def make_train_step(preprocess_cfg: PreprocessConfig, grad_clip_norm: float, data_parallel: bool, mesh: Mesh | None):
    def _train_step_impl(model, optimizer, ema_params, rng_key, sim_state):
        rng_key, key_pre, key_loss = jax.random.split(rng_key, 3)
        pre_batch, _ = preprocess_simulator_state(sim_state, key_pre, preprocess_cfg)
        features = pre_batch.features

        def _loss_fn(m):
            return m.loss(features, rng=key_loss, data_parallel=data_parallel)

        loss, grads = nnx.value_and_grad(_loss_fn)(model)
        clipped_grads, grad_norm = _clip_grads(grads, grad_clip_norm)
        optimizer.update(clipped_grads)

        params = nnx.state(model, nnx.Param)
        ema_params_next = _ema_update(ema_params, params, preprocess_cfg.ema_decay)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "ego_traj_max": jnp.max(features["ego_trajectory"]),
            "ego_traj_min": jnp.min(features["ego_trajectory"]),
        }
        return ema_params_next, rng_key, metrics
    if data_parallel:
        # Avoid jit-of-sharded-jit interactions in multi-GPU mode.
        return _train_step_impl
    return nnx.jit(_train_step_impl)


def _count_d2h_events_in_trace(profile_root: str) -> int:
    perfetto_files = glob.glob(os.path.join(profile_root, "plugins", "profile", "*", "perfetto_trace.json.gz"))
    if not perfetto_files:
        return 0

    d2h = 0
    for path in perfetto_files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("traceEvents", []):
            name = ev.get("name")
            if isinstance(name, str) and "MemcpyD2H" in name:
                d2h += 1
    return d2h


def run_transfer_profile_check(
    train_step_fn,
    model,
    optimizer,
    ema_params,
    rng_key,
    scenarios,
    steps: int,
    max_d2h_events: int,
    profile_dir: str,
):
    if steps <= 0:
        return ema_params, rng_key

    os.makedirs(profile_dir, exist_ok=True)
    run_dir = os.path.join(profile_dir, datetime.datetime.now().strftime("transfer_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    with jax.profiler.trace(run_dir, create_perfetto_trace=True):
        for _ in range(steps):
            sim_state = next(scenarios)
            ema_params, rng_key, _ = train_step_fn(model, optimizer, ema_params, rng_key, sim_state)

    d2h_events = _count_d2h_events_in_trace(run_dir)
    if d2h_events > max_d2h_events:
        raise RuntimeError(
            f"Transfer profile check failed: MemcpyD2H events={d2h_events} exceeds "
            f"allowed={max_d2h_events}. Trace dir: {run_dir}"
        )
    print(
        f"[transfer-check] OK: MemcpyD2H events={d2h_events} "
        f"(allowed<={max_d2h_events}) trace={run_dir}"
    )
    return ema_params, rng_key


def _is_nccl_runtime_error(exc: Exception) -> bool:
    msg = str(exc)
    return "NCCL" in msg or "nccl" in msg


def _log_jsonl(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    args = parse_args()
    preprocess_cfg = _build_preprocess_config(args)

    wandb_name = f"{args.wandb_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.save_dir, wandb_name)
    os.makedirs(run_dir, exist_ok=True)

    os.environ["WANDB_MODE"] = args.wandb_mode
    run = wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        entity=args.wandb_entity,
        config=config_to_dict(args),
    )

    model = _build_model(args, args.seed)
    _run_data_parallel_warmup_if_needed(model, args)
    global_batch_size = _resolve_global_batch_size(args)

    # Import Waymax dataloader lazily after data-parallel collectives are initialized.
    from waymax import config as waymax_config
    from waymax import dataloader

    tx, lr_schedule = _build_optimizer(args)
    optimizer = nnx.Optimizer(model, tx)
    data_parallel_enabled = args.data_parallel
    data_mesh = _build_data_parallel_mesh_if_needed(args) if data_parallel_enabled else None
    if data_parallel_enabled and not model.supports_data_parallel():
        raise ValueError("Model does not support data_parallel on this host/device configuration.")
    train_step_fn = make_train_step(preprocess_cfg, args.grad_clip_norm, data_parallel_enabled, data_mesh)

    ema_params = jax.tree_util.tree_map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    rng_key = jax.random.PRNGKey(args.seed)

    start_epoch = 1
    global_step = 0

    if args.resume_path:
        restored = restore_checkpoint(args.resume_path)
        nnx.update(model, restored["model_state"])
        optimizer.opt_state = _rewrap_opt_state(optimizer.opt_state, restored["opt_state"])
        optimizer.step = _rewrap_opt_state(optimizer.step, restored["opt_step"])
        ema_params = restored["ema_params"]
        rng_key = restored["rng_key"]
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])

    ds_cfg = _make_dataset_config(args, waymax_config, global_batch_size)
    scenarios = dataloader.simulator_state_generator(ds_cfg)

    try:
        ema_params, rng_key = run_transfer_profile_check(
            train_step_fn=train_step_fn,
            model=model,
            optimizer=optimizer,
            ema_params=ema_params,
            rng_key=rng_key,
            scenarios=scenarios,
            steps=args.transfer_profile_steps,
            max_d2h_events=args.transfer_profile_max_d2h_events,
            profile_dir=args.transfer_profile_dir,
        )
    except Exception as exc:
        if data_parallel_enabled and args.data_parallel_fallback and _is_nccl_runtime_error(exc):
            print(f"[data-parallel] NCCL failure during transfer profiling; falling back to single-device. Error: {exc}")
            data_parallel_enabled = False
            data_mesh = None
            train_step_fn = make_train_step(preprocess_cfg, args.grad_clip_norm, data_parallel_enabled, data_mesh)
        else:
            raise

    for epoch in range(start_epoch, args.epochs + 1):
        ema_loss = None
        pbar = tqdm(total=args.steps_per_epoch, desc=f"Epoch {epoch:04d}")

        for _ in range(args.steps_per_epoch):
            sim_state = next(scenarios)
            try:
                ema_params, rng_key, metrics = train_step_fn(model, optimizer, ema_params, rng_key, sim_state)
            except Exception as exc:
                if data_parallel_enabled and args.data_parallel_fallback and _is_nccl_runtime_error(exc):
                    print(f"[data-parallel] NCCL failure during train step; falling back to single-device. Error: {exc}")
                    data_parallel_enabled = False
                    data_mesh = None
                    train_step_fn = make_train_step(preprocess_cfg, args.grad_clip_norm, data_parallel_enabled, data_mesh)
                    ema_params, rng_key, metrics = train_step_fn(model, optimizer, ema_params, rng_key, sim_state)
                else:
                    raise

            loss_val = float(metrics["loss"])
            grad_norm = float(metrics["grad_norm"])
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.99 * ema_loss + 0.01 * loss_val

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{ema_loss:.4f}", "gn": f"{grad_norm:.2f}"})

            if global_step % args.log_every == 0:
                lr = float(lr_schedule(global_step - 1))
                payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "train/loss": loss_val,
                    "train/loss_ema": ema_loss,
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                    "data/ego_traj_max": float(metrics["ego_traj_max"]),
                    "data/ego_traj_min": float(metrics["ego_traj_min"]),
                }
                wandb.log(payload, step=global_step)
                _log_jsonl(args.log_jsonl_path, payload)

        pbar.close()
        wandb.log({"epoch": epoch}, step=global_step)

        train_state = {
            "epoch": jnp.array(epoch, dtype=jnp.int32),
            "global_step": jnp.array(global_step, dtype=jnp.int32),
            "model_state": nnx.state(model),
            "opt_state": _unwrap_opt_state(optimizer.opt_state),
            "opt_step": _unwrap_opt_state(optimizer.step),
            "ema_params": ema_params,
            "rng_key": rng_key,
        }
        save_checkpoint(run_dir, train_state, config_to_dict(args), tag="latest")
        if epoch % args.save_every == 0:
            save_checkpoint(run_dir, train_state, config_to_dict(args), tag=f"epoch_{epoch:04d}")

    run.finish()


if __name__ == "__main__":
    main()
