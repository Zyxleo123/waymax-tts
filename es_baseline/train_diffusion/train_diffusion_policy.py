from __future__ import annotations
from collections.abc import Mapping
import json
import os
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from data.cache_loader import build_dataloader
from model.diffusion.diffusion_policy import DiffusionPolicy
try:
	from .configs.diffusion_train_config import DiffusionTrainConfig, config_to_dict, parse_args
except ImportError:
	from train_diffusion.configs.diffusion_train_config import DiffusionTrainConfig, config_to_dict, parse_args
from train_diffusion.utils.checkpoints import restore_checkpoint, save_checkpoint
from train_diffusion.utils.infer import _build_model_from_metadata, _load_checkpoint_metadata
from train_diffusion.utils.utils import (
	clip_grads,
	coerce_tree_like,
	configure_jax_compilation_cache,
	ema_update,
	iter_with_prefetch,
	init_wandb_run,
	wandb_log_train_step,
	log_jsonl,
	update_tqdm,
)


def _jsonable(value: Any) -> Any:
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	if isinstance(value, dict):
		return {k: _jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	return str(value)


def _scalarize_for_logging(metrics: dict[str, Any]) -> dict[str, float]:
	logged: dict[str, float] = {}
	for key, value in metrics.items():
		arr = jnp.asarray(jax.device_get(value))
		logged[key] = float(arr.mean()) if arr.ndim > 0 else float(arr)
	return logged

def _state_to_pure_dict(state):
    """Convert flax.nnx.State to a pure dict. Leave dicts unchanged."""
    if isinstance(state, dict):
        return state

    if hasattr(state, "to_pure_dict"):
        return state.to_pure_dict()

    # flax.nnx.State
    return nnx.to_pure_dict(state)


def _pure_dict_to_state(tree):
    """Convert pure dict to flax.nnx.State. Leave State unchanged."""
    if isinstance(tree, nnx.State):
        return tree

    if hasattr(nnx.State, "from_pure_dict"):
        return nnx.State.from_pure_dict(tree)

    return nnx.State(tree)


def _path_to_string(path) -> str:
	parts: list[str] = []
	for item in path:
		if hasattr(item, "key"):
			parts.append(str(item.key))
		elif hasattr(item, "name"):
			parts.append(str(item.name))
		elif hasattr(item, "idx"):
			parts.append(str(item.idx))
		else:
			parts.append(str(item))
	return "/".join(parts)


def _is_trainable_path(path) -> bool:
	path_str = _path_to_string(path)
	return ("instruction_encoder" in path_str) or ("film_blocks" in path_str) or ("subgoal_encoder" in path_str)


def _zero_film_block_params(params_tree):
    def _maybe_zero(path, leaf):
        return jnp.zeros_like(leaf) if "film_blocks" in _path_to_string(path) else leaf

    return jax.tree_util.tree_map_with_path(_maybe_zero, params_tree)


def _load_pretrained_state(
	checkpoint_path: str,
	model: DiffusionPolicy,
	*,
	use_ema: bool,
) -> tuple[Any, Any]:
	metadata = _load_checkpoint_metadata(checkpoint_path, None)
	graphdef, template_params_state, template_nonparam_state = nnx.split(model, nnx.Param, ...)
	restored = restore_checkpoint(checkpoint_path)

	if "params_state" not in restored:
		raise KeyError("Checkpoint is missing 'params_state'.")
	if use_ema and "ema_params" in restored:
		raw_params_state = restored["ema_params"]
	else:
		raw_params_state = restored["params_state"]

	params_state = coerce_tree_like(template_params_state, raw_params_state)
	nonparam_state = coerce_tree_like(template_nonparam_state, restored.get("nonparam_state", template_nonparam_state))
	return graphdef, params_state, nonparam_state


def _build_model_from_checkpoint(checkpoint_path: str, seed: int) -> DiffusionPolicy:
	metadata = _load_checkpoint_metadata(checkpoint_path, None)
	return _build_model_from_metadata(metadata, seed)


def _make_optimizer(args: DiffusionTrainConfig, params_tree) -> tuple[optax.GradientTransformation, Any, Any]:
	# params_tree should be a pure dict pytree, not flax.nnx.State.
	trainable_mask = jax.tree_util.tree_map_with_path(
		lambda path, leaf: _is_trainable_path(path),
		params_tree,
	)
	param_labels = jax.tree_util.tree_map(
		lambda is_trainable: "train" if is_trainable else "freeze",
		trainable_mask,
	)

	total_steps = max(1, int(args.epochs) * int(args.steps_per_epoch))
	lr_schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,
		peak_value=float(args.lr),
		warmup_steps=max(1, int(args.warmup_steps)),
		decay_steps=max(int(args.warmup_steps) + 1, total_steps),
		end_value=float(args.lr) * 0.1,
	)

	train_tx = optax.adamw(
		learning_rate=lr_schedule,
		weight_decay=float(args.weight_decay),
	)
	tx = optax.multi_transform(
		{
			"train": train_tx,
			"freeze": optax.set_to_zero(),
		},
		param_labels,
	)
	return tx, lr_schedule, trainable_mask


def _make_train_step(
	graphdef,
	nonparam_state,
	tx,
	trainable_mask,
	grad_clip_norm: float,
	ema_decay: float,
	goal_mask_prob: float,
):
	goal_mask_prob = jnp.asarray(jnp.clip(goal_mask_prob, 0.0, 1.0), dtype=jnp.float32)

	def _mask_goal_features(feats: dict[str, jax.Array], rng: jax.Array) -> dict[str, jax.Array]:
		if "goal_xy" not in feats and "subgoal_valid" not in feats:
			return feats

		rng_goal, rng_subgoal = jax.random.split(rng)
		keep_goal = None
		keep_subgoal = None
		if "goal_xy" in feats:
			keep_goal = jax.random.bernoulli(
				rng_goal,
				p=jnp.asarray(1.0, dtype=jnp.float32) - goal_mask_prob,
				shape=(feats["goal_xy"].shape[0], 1),
			)
		if "subgoal_valid" in feats:
			keep_subgoal = jax.random.bernoulli(
				rng_subgoal,
				p=jnp.asarray(1.0, dtype=jnp.float32) - goal_mask_prob,
				shape=feats["subgoal_valid"].shape,
			)

		masked = dict(feats)
		if keep_goal is not None:
			masked["goal_xy"] = jnp.where(keep_goal, feats["goal_xy"], jnp.zeros_like(feats["goal_xy"]))
			masked["remaining_timesteps"] = jnp.where(
				keep_goal,
				feats["remaining_timesteps"],
				jnp.zeros_like(feats["remaining_timesteps"]),
			)
		if keep_subgoal is not None:
			masked["subgoal_valid"] = jnp.where(
				keep_subgoal,
				feats["subgoal_valid"],
				jnp.zeros_like(feats["subgoal_valid"]),
			)
			masked["subgoal_xy"] = jnp.where(
				keep_subgoal[..., None],
				feats["subgoal_xy"],
				jnp.zeros_like(feats["subgoal_xy"]),
			)
		return masked

	def merge_model(params_tree):
		return nnx.merge(
			graphdef,
			_pure_dict_to_state(params_tree),
			_pure_dict_to_state(nonparam_state),
		)

	def train_step(params_state, opt_state, ema_params, rng_key, batch_features):
		rng_key, loss_key, mask_key = jax.random.split(rng_key, 3)
		batch_features = _mask_goal_features(batch_features, mask_key)

		def loss_fn(p):
			model = merge_model(p)
			loss = model.loss(batch_features, rng=loss_key)
			inst_valid = jnp.asarray(batch_features["inst_valid"], dtype=jnp.float32)
			metrics = {
				"loss": loss,
				"inst_valid_rate": jnp.mean(inst_valid),
			}
			return loss, metrics

		(loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_state)
		clipped_grads, grad_norm = clip_grads(grads, grad_clip_norm)
		updates, opt_state = tx.update(clipped_grads, opt_state, params_state)
		params_state = optax.apply_updates(params_state, updates)
		ema_params = ema_update(ema_params, params_state, ema_decay)
		metrics = {**metrics, "grad_norm": grad_norm}
		return params_state, opt_state, ema_params, rng_key, metrics

	return jax.jit(train_step)

def main() -> None:
	args = parse_args()
	if args.resume_path is None and args.pretrained_ckpt is None:
		raise ValueError("Provide either --resume_path or --pretrained_ckpt.")

	configure_jax_compilation_cache(args.jax_compilation_cache_dir)

	checkpoint_path = args.resume_path or args.pretrained_ckpt
	assert checkpoint_path is not None
	model = _build_model_from_checkpoint(checkpoint_path, args.seed)
	graphdef, params_state, nonparam_state = nnx.split(model, nnx.Param, ...)
	params_state = _state_to_pure_dict(params_state)
	nonparam_state = _state_to_pure_dict(nonparam_state)

	if args.resume_path:
		tx, lr_schedule, trainable_mask = _make_optimizer(args, params_state)
		params_state, nonparam_state, opt_state, ema_params, rng_key, start_epoch, global_step, global_step_host = restore_checkpoint(
			args.resume_path,
			params_state=params_state,
			nonparam_state=nonparam_state,
			tx=tx,
			coerce_tree_like_fn=coerce_tree_like,
		)
	else:
		restored_graphdef, params_state, nonparam_state = _load_pretrained_state(
            args.pretrained_ckpt,
            model,
            use_ema=bool(args.use_ema),
        )
		params_state = _state_to_pure_dict(params_state)
		nonparam_state = _state_to_pure_dict(nonparam_state)
		# params_state = _zero_film_block_params(params_state)
		graphdef = restored_graphdef
		tx, lr_schedule, trainable_mask = _make_optimizer(args, params_state)
		opt_state = tx.init(params_state)
		ema_params = jax.tree_util.tree_map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            params_state,
        )
		rng_key = jax.random.PRNGKey(int(args.seed))
		start_epoch = 1
		global_step = jnp.array(0, dtype=jnp.int32)
		global_step_host = 0

	loader = build_dataloader(
		args.cache_dir,
		preprocess_cfg=args.preprocess_cfg,
		anchor_steps=(0, 10, 20, 30, 40),
		backend='jax',
		include_inst_features=True,
		batch_size=int(args.cache_batch_size),
		shuffle_seed=int(args.shuffle_seed),
		num_workers=int(args.num_workers),
		pin_memory=bool(args.pin_memory),
		drop_last=True
	)
	train_step = _make_train_step(
		graphdef,
		nonparam_state,
		tx,
		trainable_mask,
		float(args.grad_clip_norm),
		float(args.ema_decay),
		float(args.goal_mask_prob),
	)

	config_dict = config_to_dict(args)
	run, run_dir = init_wandb_run(args, config_dict)
	iterator = iter_with_prefetch(iter(loader), 0)
	ema_loss = None

	for epoch in range(int(start_epoch), int(args.epochs) + 1):
		pbar = tqdm(total=int(args.steps_per_epoch), desc=f"Epoch {epoch:04d}")
		for step_in_epoch in range(int(args.steps_per_epoch)):
			try:
				batch = next(iterator)
			except StopIteration:
				iterator = iter_with_prefetch(iter(loader), 0)
				batch = next(iterator)


			params_state, opt_state, ema_params, rng_key, metrics = train_step(
				params_state,
				opt_state,
				ema_params,
				rng_key,
				batch.features,
			)
			# print(f"Step {global_step_host}: loss={metrics['loss']:.6g}, inst_valid_rate={metrics['inst_valid_rate']:.6g}, grad_norm={metrics['grad_norm']:.6g}")
			global_step = global_step + jnp.int32(1)
			global_step_host += 1
			pbar.update(1)
			ema_loss = update_tqdm(pbar, global_step_host, args.pbar_every, metrics, ema_loss)

			if global_step_host % int(args.log_every) == 0:
				metric_values = _scalarize_for_logging(metrics)
				metric_values["epoch"] = int(epoch)
				metric_values["step"] = int(global_step_host)
				metric_values["lr"] = float(lr_schedule(max(global_step_host - 1, 0)))
				# print(json.dumps(metric_values, sort_keys=True))
				ema_loss, payload = wandb_log_train_step(global_step_host, int(epoch), metric_values, ema_loss, float(lr_schedule(max(global_step_host - 1, 0))))
				log_jsonl(os.path.join(run_dir, "train_logs.jsonl"), payload)

		pbar.close()

		if epoch % int(args.save_every) == 0 or epoch == int(args.epochs):
			save_checkpoint(
				run_dir,
				config_dict,
				epoch=epoch,
				global_step=global_step,
				params_state=params_state,
				nonparam_state=nonparam_state,
				opt_state=opt_state,
				ema_params=ema_params,
				rng_key=rng_key,
				save_every=args.save_every,
			)


if __name__ == "__main__":
	main()
