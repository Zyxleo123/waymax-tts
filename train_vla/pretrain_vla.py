from __future__ import annotations

import dataclasses
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.vla.gemma_vla import VecSceneGemmaVLA
from train_vla.configs.vla_pretrain_config import VLAPretrainConfig, parse_args
from train_vla.utils.utils import (
	resolve_device,
	resolve_dtype,
	build_vla_model,
	save_checkpoint,
	save_training_config,
	evaluate_answer_accuracy,
	flatten_batch,
	tokenize_text_batch,
	move_features_to_device,
)
from data.cache_loader import build_dataloader
from data.utils import split_cache_paths, resolve_cache_paths
from datetime import datetime


def run_training(cfg: VLAPretrainConfig) -> None:
	device = resolve_device()
	dtype = resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	run_name = cfg.wandb_run_name + f"_{cfg.model_type}"
	if cfg.tag:
		run_name += f"_{cfg.tag}"
	run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

	output_dir = Path(cfg.output_dir) / run_name
	output_dir.mkdir(parents=True, exist_ok=True)
	save_training_config(cfg, output_dir)

	# Split cache paths into train and validation sets
	all_cache_paths = resolve_cache_paths(cfg.cache_dir, cfg.file_indices)
	train_cache_paths, val_cache_paths = split_cache_paths(all_cache_paths, cfg.validation_fraction)

	train_loader = build_dataloader(
		cfg.cache_dir,
		preprocess_cfg=cfg.preprocess_cfg,
		annotation_dir=cfg.annotation_dir,
		anchor_steps=(0, 10, 20, 30, 40, 50, 60, 70, 80),
		language_label="qa",
		backend="torch",
		include_inst_features=False,
		file_indices=None,  # use cache_paths instead
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		cache_paths=train_cache_paths,
	)

	val_loader = build_dataloader(
		cfg.cache_dir,
		preprocess_cfg=cfg.preprocess_cfg,
		annotation_dir=cfg.annotation_dir,
		anchor_steps=(0, 10, 20, 30, 40, 50, 60, 70, 80),
		language_label="qa",
		backend="torch",
		include_inst_features=False,
		file_indices=None,  # use cache_paths instead
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		cache_paths=val_cache_paths,
	) if val_cache_paths else None

	model = build_vla_model(cfg)
	if cfg.use_gradient_checkpointing and hasattr(model.llm, "gradient_checkpointing_enable"):
		model.llm.gradient_checkpointing_enable()
	if cfg.freeze_llm:
		model.freeze_llm()

	model = model.to(device=device, dtype=dtype)
	model.train()

	wandb_run = None
	if cfg.wandb_project:
		try:
			import wandb
		except ImportError as exc:
			raise ImportError("wandb is required when wandb_project is set") from exc
		wandb_run = wandb.init(
			project=cfg.wandb_project,
			entity=cfg.wandb_entity,
			name=run_name,
			mode=cfg.wandb_mode,
			config=dataclasses.asdict(cfg),
		)
		wandb_run.define_metric("train/step")
		wandb_run.define_metric("train/*", step_metric="train/step")
		wandb_run.define_metric("eval/*", step_metric="train/step")

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	
	# Warmup + optional decay scheduler
	scheduler = None
	if cfg.warmup_steps > 0:
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=cfg.warmup_steps)
		
		# If max_steps is set, add cosine annealing after warmup
		if cfg.max_steps is not None:
			decay_steps = cfg.max_steps - cfg.warmup_steps
			if decay_steps > 0:
				decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
				scheduler = torch.optim.lr_scheduler.SequentialLR(
					optimizer, 
					[warmup_scheduler, decay_scheduler], 
					milestones=[cfg.warmup_steps]
				)
			else:
				scheduler = warmup_scheduler
		else:
			scheduler = warmup_scheduler

	if cfg.dtype == "fp16":
		scaler = torch.cuda.amp.GradScaler()
	else:
		scaler = None

	global_step = 0
	optimizer.zero_grad(set_to_none=True)
	for epoch in range(cfg.num_epochs):
		epoch_total = None
		if cfg.max_steps is not None:
			epoch_total = max(0, cfg.max_steps - global_step)
		pbar = tqdm(total=epoch_total, desc=f"epoch {epoch + 1}/{cfg.num_epochs}", unit="step")
		for batch in train_loader:
			features = batch.features
			prompts = batch.prompts
			answers = batch.answers

			prompt_ids, prompt_mask, answer_ids, answer_mask = tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				add_eos=cfg.add_eos,
				device=device,
				max_prompt_length=cfg.max_prompt_length,
				max_answer_length=cfg.max_answer_length,
			)
			features = move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(features, prompt_ids=prompt_ids, prompt_mask=prompt_mask, answer_ids=answer_ids, answer_mask=answer_mask)
				loss = outputs["loss"] / cfg.grad_accum_steps

			if scaler is not None:
				scaler.scale(loss).backward()
			else:
				loss.backward()

			if (global_step + 1) % cfg.grad_accum_steps == 0:
				if cfg.max_grad_norm > 0:
					if scaler is not None:
						scaler.unscale_(optimizer)
						clip_grad_norm_(trainable_params, cfg.max_grad_norm)
					else:
						clip_grad_norm_(trainable_params, cfg.max_grad_norm)

				if scaler is not None:
					scaler.step(optimizer)
					scaler.update()
				else:
					optimizer.step()
				optimizer.zero_grad(set_to_none=True)
				if scheduler is not None:
					scheduler.step()

			if global_step % cfg.log_every == 0:
				current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
				train_loss = float(outputs["loss"].detach().cpu())
				pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.3e}", batch=len(prompts))
				if wandb_run is not None:
					wandb_run.log(
						{
							"train/step": global_step,
							"train/loss": train_loss,
							"train/subgoal_loss": float(outputs.get("subgoal_loss", torch.tensor(0.0)).detach().cpu()),
							"train/language_loss": float(outputs.get("language_loss", torch.tensor(0.0)).detach().cpu()),
							"train/lr": current_lr,
							"train/batch_size": len(prompts),
						},
						step=global_step,
					)

			if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
				save_checkpoint(output_dir, global_step, model, optimizer, scheduler, cfg=cfg)

			if (global_step + 1) % 1000 == 0 or global_step == 0:

				# Evaluate on validation set
				val_metrics = {"accuracy": 0.0, "num_samples": 0.0}
				val_per_q_metrics = {}
				if val_loader is not None:
					val_metrics, val_per_q_metrics = evaluate_answer_accuracy(
						model,
						val_loader,
						device=device,
						dtype=dtype,
						max_samples=cfg.eval_num_samples,
						max_prompt_length=cfg.max_prompt_length,
						max_answer_length=cfg.max_answer_length,
						model_type=cfg.model_type,
					)

					# Evaluate on training set
					train_metrics, train_per_q_metrics = evaluate_answer_accuracy(
						model,
						train_loader,
						device=device,
						dtype=dtype,
						max_samples=cfg.eval_num_samples,
						max_prompt_length=cfg.max_prompt_length,
						max_answer_length=cfg.max_answer_length,
						model_type=cfg.model_type,
					)

					tqdm.write(
						f"epoch={epoch} "
						f"train_accuracy={train_metrics['accuracy']:.4f} train_samples={int(train_metrics['num_samples'])} "
						f"val_accuracy={val_metrics['accuracy']:.4f} val_samples={int(val_metrics['num_samples'])}"
					)
					if wandb_run is not None:
						wandb_log_dict = {
							"train/step": global_step,
							"overall_acc/train_accuracy": train_metrics["accuracy"],
							"overall_acc/val_accuracy": val_metrics["accuracy"],
						}
						# Add per-question metrics
						for q_key, q_metrics in train_per_q_metrics.items():
							wandb_log_dict[f"train_acc/train_qa_{q_key}_accuracy"] = q_metrics["accuracy"]
							# wandb_log_dict[f"train/train_qa_{q_key}_samples"] = q_metrics["num_samples"]
						for q_key, q_metrics in val_per_q_metrics.items():
							wandb_log_dict[f"val_acc/val_qa_{q_key}_accuracy"] = q_metrics["accuracy"]
							# wandb_log_dict[f"val/val_qa_{q_key}_samples"] = q_metrics["num_samples"]
						wandb_run.log(wandb_log_dict, step=global_step)

			global_step += 1
			pbar.update(1)
			if cfg.max_steps is not None and global_step >= cfg.max_steps:
				break
		pbar.close()
		if cfg.max_steps is not None and global_step >= cfg.max_steps:
			break

		

	save_pretrained = getattr(model.llm, "save_pretrained", None)
	if callable(save_pretrained):
		model.llm.save_pretrained(str(output_dir / "llm"))
	model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
	save_checkpoint(output_dir, global_step, model, optimizer, scheduler, cfg=cfg)
	if wandb_run is not None:
		wandb_run.finish()



def main() -> None:
	config = parse_args()
	run_training(config)


if __name__ == "__main__":
	main()