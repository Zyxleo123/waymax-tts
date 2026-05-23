from __future__ import annotations

import itertools
import os
import gc

# Reduce TensorFlow/XLA startup noise and keep TF from competing for GPU memory.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_LOGGING_LEVEL", "ERROR")
# Keep this for Waymax compatibility on hosts where TF can still see GPUs.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P
from tqdm import tqdm

# Keep JAX/XLA logging to errors unless explicitly overridden by the caller.
jax.config.update("jax_logging_level", "ERROR")

from train.utils.checkpoints import restore_checkpoint, save_checkpoint
from data.preprocess import preprocess_simulator_state
from data.types import PreprocessConfig
from train.utils.utils import (
    assert_matching_sim_state_signature,
    build_data_parallel_mesh_if_needed,
    canonicalize_state_for_mesh,
    clip_grads,
    coerce_tree_like,
    configure_jax_compilation_cache,
    ema_update,
    init_wandb_run,
    iter_with_prefetch,
    log_jsonl,
    make_dataset_config,
    replicate_to_mesh,
    sim_state_signature,
    state_in_axes_from_template,
    update_tqdm,
    wandb_log_train_step,
)

from model.diffusion.diffusion_policy import DiffusionPolicy
from train.configs.diffusion_pretrain_config import DiffusionPretrainConfig, config_to_dict, parse_args


def build_training_components(args: DiffusionPretrainConfig):
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
    )
    model = DiffusionPolicy(
        rngs=nnx.Rngs(args.seed),
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        other_dim=args.other_dim,
        map_attr_dim=args.map_attr_dim,
        tl_attr_dim=args.tl_attr_dim,
        inst_attr_dim=args.inst_dim,
        predict_horizon=args.predict_horizon,
        predict_type=args.predict_type,
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


def _add_dummy_instruction_features(feats: dict[str, jax.Array], args: DiffusionPretrainConfig) -> dict[str, jax.Array]:
    batch_size = feats["ego_state"].shape[0]
    augmented = dict(feats)
    augmented["inst_features"] = jnp.zeros((batch_size, args.inst_dim), dtype=jnp.float32)
    augmented["inst_valid"] = jnp.zeros((batch_size,), dtype=jnp.bool_)
    return augmented


def make_train_step(
    pretrain_cfg: DiffusionPretrainConfig,
    preprocess_cfg: PreprocessConfig,
    grad_clip_norm: float,
    goal_mask_prob: float,
    data_parallel: bool,
    mesh: Mesh | None,
    ema_update_every: int,
    graphdef,
    nonparam_state,
    tx,
    state_in_axes=None,
):
    use_data_parallel = bool(data_parallel and mesh is not None)
    ego_range = jnp.asarray(preprocess_cfg.ego_range, dtype=jnp.float32)
    max_velocity = jnp.asarray(preprocess_cfg.max_velocity, dtype=jnp.float32)
    goal_mask_prob = jnp.asarray(jnp.clip(goal_mask_prob, 0.0, 1.0), dtype=jnp.float32)

    def _maybe_mask_goal_xy(feats: dict[str, jax.Array], rng: jax.Array) -> dict[str, jax.Array]:
        if "goal_xy" not in feats:
            return feats
        keep_goal = jax.random.bernoulli(
            rng,
            p=jnp.asarray(1.0, dtype=jnp.float32) - goal_mask_prob,
            shape=(feats["goal_xy"].shape[0], 1),
        )
        masked = dict(feats)
        masked["goal_xy"] = jnp.where(keep_goal, feats["goal_xy"], jnp.zeros_like(feats["goal_xy"]))
        masked["remaining_timesteps"] = jnp.where(keep_goal, feats["remaining_timesteps"], jnp.zeros_like(feats["remaining_timesteps"]))
        return masked

    def _wrap_to_pi(angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _trajectory_metrics(pred_btd, target_btd):
        pred_xy = pred_btd[..., :2]
        target_xy = target_btd[..., :2]
        xy_err_m = jnp.linalg.norm(pred_xy - target_xy, axis=-1) * ego_range

        metrics = {
            "traj_ade_m": jnp.mean(xy_err_m),
            "traj_fde_m": jnp.mean(xy_err_m[:, -1]),
            "traj_final_longitudinal_err_m": jnp.mean(jnp.abs(pred_btd[:, -1, 0] - target_btd[:, -1, 0]) * ego_range),
            "traj_final_lateral_err_m": jnp.mean(jnp.abs(pred_btd[:, -1, 1] - target_btd[:, -1, 1]) * ego_range),
        }

        if target_btd.shape[-1] >= 4:
            pred_speed = jnp.linalg.norm(pred_btd[..., 2:4], axis=-1) * max_velocity
            target_speed = jnp.linalg.norm(target_btd[..., 2:4], axis=-1) * max_velocity
            metrics["speed_mae_mps"] = jnp.mean(jnp.abs(pred_speed - target_speed))
        else:
            metrics["speed_mae_mps"] = jnp.asarray(0.0, dtype=jnp.float32)

        if target_btd.shape[-1] >= 5:
            yaw_err = _wrap_to_pi((pred_btd[..., 4] - target_btd[..., 4]) * jnp.pi)
            metrics["yaw_mae_deg"] = jnp.mean(jnp.abs(yaw_err)) * (180.0 / jnp.pi)
        else:
            metrics["yaw_mae_deg"] = jnp.asarray(0.0, dtype=jnp.float32)

        return metrics

    def merge_model(p):
        return nnx.merge(graphdef, p, nonparam_state)

    def single_device_loss_and_metrics(p, sim_state, key_pre, key_loss, key_sample):
        m = merge_model(p)
        key_pre, key_goal_mask = jax.random.split(key_pre)
        pre_batch, _ = preprocess_simulator_state(sim_state, key_pre, preprocess_cfg)
        feats = _maybe_mask_goal_xy(pre_batch.features, key_goal_mask)
        feats = _add_dummy_instruction_features(feats, pretrain_cfg)
        features = m._as_features(feats)
        cond, inst_cond = m.compute_condition(features)
        target = features.ego_trajectory[:, : m.predict_horizon, :]
        inst_cond_mask = jnp.asarray(features.inst_valid, dtype=bool)
        loss = m.loss_from_condition(target, cond, inst_cond, inst_cond_mask, rng=key_loss)
        # pred = m._sample_from_condition_impl(cond, key_sample)
        # return loss, _trajectory_metrics(pred, target)
        return loss, {}

    if use_data_parallel:
        num_devices = mesh.devices.size

        def local_loss_with_params(p, state_local, key_pre_local, key_loss_local, key_sample_local):
            m = merge_model(p)
            key_pre_local, key_goal_mask_local = jax.random.split(key_pre_local)
            pre_local, _ = preprocess_simulator_state(state_local, key_pre_local, preprocess_cfg)
            feats_local = _maybe_mask_goal_xy(pre_local.features, key_goal_mask_local)
            feats_local = _add_dummy_instruction_features(feats_local)
            features_local = m._as_features(feats_local)
            cond_local, inst_cond_local = m.compute_condition(features_local)
            target_local = features_local.ego_trajectory[:, : m.predict_horizon, :]
            inst_cond_mask_local = jnp.asarray(features_local.inst_valid, dtype=bool)
            loss_local = m.loss_from_condition(target_local, cond_local, inst_cond_local, inst_cond_mask_local, rng=key_loss_local)
            # pred_local = m._sample_from_condition_impl(cond_local, key_sample_local)
            # return loss_local, _trajectory_metrics(pred_local, target_local)
            return loss_local, {}
        
        pmapped_local_loss = jax.pmap(
            local_loss_with_params,
            axis_name="data",
            in_axes=(None, state_in_axes, 0, 0, 0),
            out_axes=(0, 0),
        )

        def compute_loss_and_metrics(p, sim_state, key_pre, key_loss, key_sample):
            key_pre_devices = jax.random.split(key_pre, num_devices)
            key_loss_devices = jax.random.split(key_loss, num_devices)
            key_sample_devices = jax.random.split(key_sample, num_devices)
            key_pre_devices = jax.lax.with_sharding_constraint(key_pre_devices, P("data", None))
            key_loss_devices = jax.lax.with_sharding_constraint(key_loss_devices, P("data", None))
            key_sample_devices = jax.lax.with_sharding_constraint(key_sample_devices, P("data", None))
            losses, metric_tree = pmapped_local_loss(p, sim_state, key_pre_devices, key_loss_devices, key_sample_devices)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, metric_tree)

    else:

        def compute_loss_and_metrics(p, sim_state, key_pre, key_loss, key_sample):
            return single_device_loss_and_metrics(p, sim_state, key_pre, key_loss, key_sample)

    def train_step_impl(params_state, opt_state, ema_params, rng_key, global_step, sim_state):
        rng_key, key_pre, key_loss, key_sample = jax.random.split(rng_key, 4)

        def loss_fn(p):
            return compute_loss_and_metrics(p, sim_state, key_pre, key_loss, key_sample)

        (loss, train_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_state)
        clipped_grads, grad_norm = clip_grads(grads, grad_clip_norm)
        updates, opt_state_next = tx.update(clipped_grads, opt_state, params_state)
        params_next = optax.apply_updates(params_state, updates)

        ema_params_candidate = ema_update(ema_params, params_next, preprocess_cfg.ema_decay)
        if ema_update_every <= 1:
            ema_params_next = ema_params_candidate
        else:
            step_next = global_step + jnp.int32(1)
            do_ema = (step_next % jnp.int32(ema_update_every)) == 0
            ema_params_next = jax.tree_util.tree_map(
                lambda old, new: jnp.where(do_ema, new, old),
                ema_params,
                ema_params_candidate,
            )

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            **train_metrics,
        }
        return params_next, opt_state_next, ema_params_next, rng_key, global_step + jnp.int32(1), metrics

    if use_data_parallel:
        return train_step_impl
    return jax.jit(train_step_impl)


def run_train_step(
    train_step_fn,
    params_state,
    opt_state,
    ema_params,
    rng_key,
    global_step,
    sim_state,
    *,
    data_parallel: bool,
    data_mesh: Mesh | None,
    sim_state_treedef=None,
    sim_state_sig=None,
    where: str | None = None,
):
    sim_state = canonicalize_state_for_mesh(sim_state, data_mesh if data_parallel else None)
    if sim_state_treedef is not None and sim_state_sig is not None:
        assert_matching_sim_state_signature(sim_state, sim_state_treedef, sim_state_sig, where=where or "train_step")

    if data_parallel and data_mesh is not None:
        with data_mesh:
            return train_step_fn(params_state, opt_state, ema_params, rng_key, global_step, sim_state)
    return train_step_fn(params_state, opt_state, ema_params, rng_key, global_step, sim_state)


def main():
    args = parse_args()
    configure_jax_compilation_cache(args.jax_compilation_cache_dir)
    preprocess_cfg, model, tx, lr_schedule = build_training_components(args)

    run, run_dir = init_wandb_run(args, config_to_dict(args))
    global_batch_size = int(args.batch_size)

    # Waymax uses TensorFlow input pipelines; keep TF on CPU and quiet non-actionable logs.
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    try:
        tf.autograph.set_verbosity(0)
    except Exception:
        pass
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    # Import Waymax dataloader lazily after data-parallel collectives are initialized.
    from waymax import config as waymax_config
    from waymax import dataloader

    # Split model into trainable params and static/non-param state for pure jitted train step.
    graphdef, params_state, nonparam_state = nnx.split(model, nnx.Param, ...)
    opt_state = tx.init(params_state)
    ema_params = jax.tree_util.tree_map(lambda x: x.copy(), params_state)
    rng_key = jax.random.PRNGKey(args.seed)

    start_epoch = 1
    global_step = jnp.array(0, dtype=jnp.int32)
    global_step_host: int = 0

    if args.resume_path:
        (
            params_state,
            nonparam_state,
            opt_state,
            ema_params,
            rng_key,
            start_epoch,
            global_step,
            global_step_host,
        ) = restore_checkpoint(
            args.resume_path,
            params_state=params_state,
            nonparam_state=nonparam_state,
            tx=tx,
            coerce_tree_like_fn=coerce_tree_like,
        )

    data_parallel_enabled = bool(args.data_parallel)
    if data_parallel_enabled and not model.supports_data_parallel():
        raise ValueError("Model does not support data_parallel on this host/device configuration.")
    data_mesh = build_data_parallel_mesh_if_needed(args) if data_parallel_enabled else None

    if data_parallel_enabled and data_mesh is not None:
        with data_mesh:
            params_state = replicate_to_mesh(params_state, data_mesh)
            nonparam_state = replicate_to_mesh(nonparam_state, data_mesh)
            opt_state = replicate_to_mesh(opt_state, data_mesh)
            ema_params = replicate_to_mesh(ema_params, data_mesh)
            rng_key = replicate_to_mesh(rng_key, data_mesh)
            global_step = replicate_to_mesh(global_step, data_mesh)

    ds_cfg = make_dataset_config(args, waymax_config, global_batch_size)

    def make_scenarios_iter():
        return iter_with_prefetch(dataloader.simulator_state_generator(ds_cfg), args.prefetch_size)

    scenarios = make_scenarios_iter()

    # Data/mesh setup from a single canonical batch.
    first_sim_state = next(scenarios)
    first_sim_state = canonicalize_state_for_mesh(first_sim_state, data_mesh if data_parallel_enabled else None)
    state_in_axes = state_in_axes_from_template(first_sim_state) if data_parallel_enabled else None
    sim_state_treedef, sim_state_sig = sim_state_signature(first_sim_state)
    scenarios = itertools.chain([first_sim_state], scenarios)

    train_step_fn = make_train_step(
        args,
        preprocess_cfg,
        args.grad_clip_norm,
        args.goal_mask_prob,
        data_parallel_enabled,
        data_mesh,
        args.ema_update_every,
        graphdef,
        nonparam_state,
        tx,
        state_in_axes,
    )

    # Compile and autotune kernels once before entering the timed training loop.
    # We intentionally discard outputs to keep optimizer/model state unchanged.
    _, _, _, _, _, warmup_metrics = run_train_step(
        train_step_fn,
        params_state,
        opt_state,
        ema_params,
        rng_key,
        global_step,
        first_sim_state,
        data_parallel=data_parallel_enabled,
        data_mesh=data_mesh,
        sim_state_treedef=sim_state_treedef,
        sim_state_sig=sim_state_sig,
        where="warmup",
    )
    _ = jax.block_until_ready(warmup_metrics["loss"])

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch > start_epoch and args.recreate_dataloader_each_epoch:
            # Recreate the TF/Waymax iterator periodically to avoid long-run host-memory growth.
            scenarios = make_scenarios_iter()

        ema_loss = None
        pbar = tqdm(total=args.steps_per_epoch, desc=f"Epoch {epoch:04d}")

        for _ in range(args.steps_per_epoch):
            sim_state = next(scenarios)
            params_state, opt_state, ema_params, rng_key, global_step, metrics = run_train_step(
                train_step_fn,
                params_state,
                opt_state,
                ema_params,
                rng_key,
                global_step,
                sim_state,
                data_parallel=data_parallel_enabled,
                data_mesh=data_mesh,
                sim_state_treedef=sim_state_treedef,
                sim_state_sig=sim_state_sig,
                where=f"epoch={epoch} step={global_step_host + 1}",
            )

            global_step_host += 1
            pbar.update(1)
            ema_loss = update_tqdm(pbar, global_step_host, args.pbar_every, metrics, ema_loss)

            if global_step_host % args.log_every == 0:
                lr = float(lr_schedule(global_step_host - 1))
                ema_loss, payload = wandb_log_train_step(global_step_host, epoch, metrics, ema_loss, lr)
                log_jsonl(args.log_jsonl_path, payload)

        pbar.close()

        save_checkpoint(
            run_dir,
            config_to_dict(args),
            epoch=epoch,
            global_step=global_step,
            params_state=params_state,
            nonparam_state=nonparam_state,
            opt_state=opt_state,
            ema_params=ema_params,
            rng_key=rng_key,
            save_every=args.save_every,
        )

        if args.epoch_gc_every > 0 and (epoch % args.epoch_gc_every == 0):
            # Periodic host GC helps return Python-held objects between long epochs.
            gc.collect()

    run.finish()


if __name__ == "__main__":
    main()