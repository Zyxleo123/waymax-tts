from __future__ import annotations

import dataclasses
from pathlib import Path
from datetime import datetime

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from data.inst_dataloader import build_inst_dataloader, resolve_cache_paths, split_cache_paths
from train_vla.configs.vla_finetuning_config import VLAFinetuningConfig, parse_args
from train_vla.utils.utils import (
	resolve_device,
	resolve_dtype,
	build_vla_model,
	save_checkpoint,
	save_training_config,
	tokenize_text_batch,
	move_features_to_device,
)

keywords = ["stop", "straight", "left", "right", "turn", "chang", "accel", "slow"]
def compare_keyword(preds: list[str], labels: list[str]) -> list[float]:
	accuracies = []
	for i, (pred, label) in enumerate(zip(preds, labels)):
		pred_lower = pred.lower()
		label_lower = label.lower()
		target_keywords = []
		for keyword in keywords:
			if keyword in label_lower:
				target_keywords.append(keyword)
		total_keywords = len(target_keywords)
		included_keywords = 0
		for keyword in target_keywords:
			if keyword in pred_lower:
				included_keywords += 1
		if total_keywords == 0:
			accuracies.append(1.0 if included_keywords == 0 else 0.0)
		else:
			accuracies.append(included_keywords / total_keywords)
	return accuracies

def compare_subgoal(preds: list[str], labels: list[str]) -> tuple[list[float], list[float]]:
	format_accuracies = []
	l2_distances = []
	for i, (pred, label) in enumerate(zip(preds, labels)):
		try:
			pred_x, pred_y = pred.split(",")
			pred_x = float(pred_x.strip())
			pred_y = float(pred_y.strip())
		except ValueError:
			format_accuracies.append(0.0)
			continue
		label_x, label_y = label.split(",")
		label_x = float(label_x.strip())
		label_y = float(label_y.strip())
		format_accuracies.append(1.0)
		l2_distance = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) ** 0.5
		l2_distances.append(l2_distance)
	return l2_distances, format_accuracies

def evaluate_accuracy(
	model, loader, *, device, dtype, max_samples, max_prompt_length, max_answer_length, add_eos, generate_subgoal
):
	keyword_accuracies, subgoal_l2_distances, subgoal_format_accuracies = [], [], []
	was_training = model.training
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			features, prompts, answers = _flatten_instruction_batch(batch)
			remaining = max_samples - len(keyword_accuracies)
			if remaining <= 0:
				break
			if len(prompts) > remaining:
				prompts = prompts[:remaining]
				answers = answers[:remaining]
				features = {k: v[:remaining] for k, v in features.items()}
				
			prompt_ids, prompt_mask, answer_ids, answer_mask = tokenize_text_batch(
                model.tokenizer,
                prompts,
                answers,
                add_eos=add_eos,
                device=device,
                max_prompt_length=max_prompt_length,
                max_answer_length=max_answer_length,
            )
			features = move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				inst_preds, subgoal_preds = model.generate_inst_subgoal_predictions(
                    input_features=features,
                    prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
                    max_new_tokens=max_answer_length,
                )
			inst_answers = [model.split_instruction_subgoal_text(a)[0] for a in answers]
			subgoal_answers = [model.split_instruction_subgoal_text(a)[1] for a in answers]
			keyword_accuracies.extend(compare_keyword(inst_preds, inst_answers))
			if generate_subgoal:
				l2_distances, format_accuracies = compare_subgoal(subgoal_preds, subgoal_answers)
				subgoal_l2_distances.extend(l2_distances)
				subgoal_format_accuracies.extend(format_accuracies)
	keyword_accuracy = sum(keyword_accuracies) / len(keyword_accuracies) if keyword_accuracies else 0.0
	subgoal_l2_distance = sum(subgoal_l2_distances) / len(subgoal_l2_distances) if subgoal_l2_distances else 0.0
	subgoal_format_accuracy = sum(subgoal_format_accuracies) / len(subgoal_format_accuracies) if subgoal_format_accuracies else 0.0

	if was_training:
		model.train()
	return {
		"keyword_accuracy": keyword_accuracy,
        "subgoal_l2_distance": subgoal_l2_distance,
		"subgoal_format_accuracy": subgoal_format_accuracy,
    }
			

def _flatten_instruction_batch(batch) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
	if batch.prompts is None or batch.answers is None:
		raise ValueError("batch.prompts and batch.answers are required for instruction finetuning")
	features = batch.features
	prompts = list(batch.prompts)
	answers = list(batch.answers)

	return features, prompts, answers


def run_training(cfg: VLAFinetuningConfig) -> None:
	device = resolve_device()
	dtype = resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	run_name = cfg.wandb_run_name + f"_{cfg.model_type}"
	if cfg.tag:
		run_name += f"_{cfg.tag}"
	from datetime import datetime
	run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

	output_dir = Path(cfg.output_dir) / run_name
	output_dir.mkdir(parents=True, exist_ok=True)
	save_training_config(cfg, output_dir)

	all_cache_paths = resolve_cache_paths(cfg.cache_dir, cfg.file_indices, cfg.anchor_step)
	train_cache_paths, val_cache_paths = split_cache_paths(all_cache_paths, cfg.validation_fraction)

	train_loader = build_inst_dataloader(
		cfg.cache_dir,
		anchor_step=cfg.anchor_step,
		file_indices=None,
		instruction_dir=cfg.instruction_dir,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		cache_paths=train_cache_paths,
		generate_subgoal=cfg.generate_subgoal,
	)

	val_loader = build_inst_dataloader(
		cfg.cache_dir,
		anchor_step=cfg.anchor_step,
		file_indices=None,
		instruction_dir=cfg.instruction_dir,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
		cache_paths=val_cache_paths,
		generate_subgoal=cfg.generate_subgoal,
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

	scheduler = None
	if cfg.warmup_steps > 0:
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=0.001,
			total_iters=cfg.warmup_steps,
		)
		if cfg.max_steps is not None:
			decay_steps = cfg.max_steps - cfg.warmup_steps
			if decay_steps > 0:
				decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
				scheduler = torch.optim.lr_scheduler.SequentialLR(
					optimizer,
					[warmup_scheduler, decay_scheduler],
					milestones=[cfg.warmup_steps],
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
			features, prompts, answers = _flatten_instruction_batch(batch)
			if not prompts:
				continue

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
				outputs = model(
					features,
					prompt_ids=prompt_ids,
					prompt_mask=prompt_mask,
					answer_ids=answer_ids,
					answer_mask=answer_mask,
				)
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
							"train/lr": current_lr,
							"train/batch_size": len(prompts),
						},
						step=global_step,
					)

			if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
				save_checkpoint(output_dir, global_step, model, optimizer, scheduler, cfg=cfg)
				
			if (global_step + 1) % 1000 == 0 or global_step == 0:
				if val_loader is not None:
					val_metrics = evaluate_accuracy(
                        model=model,
                        loader=val_loader,
                        device=device,
                        dtype=dtype,
                        max_samples=cfg.eval_num_samples,
                        max_prompt_length=cfg.max_prompt_length,
                        max_answer_length=cfg.max_answer_length,
                        add_eos=cfg.add_eos,
						generate_subgoal=cfg.generate_subgoal,
                    )
				train_metrics = evaluate_accuracy(
                    model=model,
                    loader=train_loader,
                    device=device,
                    dtype=dtype,
                    max_samples=cfg.eval_num_samples,
                    max_prompt_length=cfg.max_prompt_length,
                    max_answer_length=cfg.max_answer_length,
                    add_eos=cfg.add_eos,
					generate_subgoal=cfg.generate_subgoal,
                )
				if wandb_run is not None:
					wandb_log_dict = {
						"val_accuracy/keyword_accuracy": val_metrics["keyword_accuracy"],
                        "val_accuracy/subgoal_l2_distance": val_metrics["subgoal_l2_distance"],
						"val_accuracy/subgoal_format_accuracy": val_metrics["subgoal_format_accuracy"],
						"train_accuracy/keyword_accuracy": train_metrics["keyword_accuracy"],
                        "train_accuracy/subgoal_l2_distance": train_metrics["subgoal_l2_distance"],
                        "train_accuracy/subgoal_format_accuracy": train_metrics["subgoal_format_accuracy"],
                    }
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
