from __future__ import annotations

import dataclasses
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from vla.qwen_vla import SceneQwenVLA
from vla.qa_dataloader import build_qa_dataloader


YES_TOKEN = "yes"
NO_TOKEN = "no"


@dataclass(frozen=True)
class TrainQAConfig:
	cache_dir: str
	qa_dir: str
	file_indices: list[int] | None
	output_dir: str
	wandb_project: str | None = "waymax_rs_qa"
	wandb_run_name: str | None = None
	wandb_entity: str | None = None
	wandb_mode: str = "online"
	qwen_name: str = "Qwen/Qwen3-0.6B"
	batch_size: int = 1
	learning_rate: float = 1e-5
	weight_decay: float = 0.01
	num_epochs: int = 1
	max_steps: int | None = None
	grad_accum_steps: int = 1
	max_grad_norm: float = 1.0
	shuffle_seed: int = 0
	shuffle_buffer_size: int = 1024
	dataset_num_shards: int = 1
	include_sdc_paths: bool = False
	max_num_objects: int = 64
	anchor_step_override: int | None = 0
	freeze_llm: bool = False
	use_gradient_checkpointing: bool = True
	dtype: str = "bf16"
	log_every: int = 10
	save_every: int = 1000
	eval_num_samples: int = 32
	max_prompt_length: int = 128
	max_answer_length: int = 4
	num_workers: int = 0
	pin_memory: bool = True


def _resolve_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
	if dtype_name == "bf16":
		return torch.bfloat16
	if dtype_name == "fp16":
		return torch.float16
	raise ValueError(f"Unsupported dtype: {dtype_name}")


def _ensure_tokenizer(model: SceneQwenVLA) -> None:
	if model.tokenizer.pad_token is None:
		model.tokenizer.pad_token = model.tokenizer.eos_token


def _qa_to_text(qa_item: Mapping[str, Any]) -> tuple[str, str]:
	question = str(qa_item["question"])
	answer = str(qa_item["answer"])
	prompt = f"{question}"
	answer_token = YES_TOKEN if answer == "yes" else NO_TOKEN
	return prompt, answer_token


def _parse_file_indices(raw_file_indices: list[str] | None) -> list[int] | None:
	if raw_file_indices is None:
		return None
	parsed: list[int] = []
	for token in raw_file_indices:
		for part in token.split(","):
			part = part.strip()
			if not part:
				continue
			parsed.append(int(part))
	return parsed or None


def _flatten_batch(batch) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
	if batch.qa is None:
		raise ValueError("batch.qa is required for QA training")

	feature_keys = list(batch.features.keys())
	flattened_features: dict[str, list[torch.Tensor]] = {key: [] for key in feature_keys}
	prompts: list[str] = []
	answers: list[str] = []

	for scenario_idx, scenario_qa in enumerate(batch.qa):
		qas = scenario_qa["qas"]
		for qa_item in qas:
			prompt, answer = _qa_to_text(qa_item)
			prompts.append(prompt)
			answers.append(answer)
			for key in feature_keys:
				flattened_features[key].append(batch.features[key][scenario_idx])

	stacked_features = {key: torch.stack(values, dim=0) for key, values in flattened_features.items()}
	return stacked_features, prompts, answers


def _tokenize_text_batch(tokenizer, prompts: list[str], answers: list[str], *, device: torch.device, max_prompt_length: int, max_answer_length: int) -> tuple[torch.Tensor, torch.Tensor]:
	prompt_batch = tokenizer(
		prompts,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=max_prompt_length,
		add_special_tokens=True,
	)
	answer_batch = tokenizer(
		answers,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=max_answer_length,
		add_special_tokens=False,
	)

	return prompt_batch["input_ids"].to(device), answer_batch["input_ids"].to(device)


def _move_features_to_device(features: Mapping[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
	moved: dict[str, torch.Tensor] = {}
	for key, value in features.items():
		if value.dtype.is_floating_point:
			moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
		else:
			moved[key] = value.to(device=device, non_blocking=True)
	return moved


def _parse_generated_yes_no(text: str) -> bool | None:
	lowered = text.lower()
	yes_match = re.search(r"<yes>|\byes\b", lowered)
	no_match = re.search(r"<no>|\bno\b", lowered)
	if yes_match is None and no_match is None:
		return None
	if yes_match is None:
		return False
	if no_match is None:
		return True
	return yes_match.start() <= no_match.start()


def _generate_yes_no_predictions(
	model: SceneQwenVLA,
	features: Mapping[str, torch.Tensor],
	prompt_ids: torch.Tensor,
	*,
	device: torch.device,
	max_new_tokens: int,
) -> list[bool | None]:
	scene_tokens = model._tokenize_scene_features(features)
	text_emb = model.llm.get_input_embeddings()(prompt_ids)
	scene_tokens = scene_tokens.to(text_emb.dtype)
	inputs_embeds = torch.cat([text_emb, scene_tokens], dim=1)
	attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
	generated_ids = model.llm.generate(
		inputs_embeds=inputs_embeds,
		attention_mask=attention_mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		pad_token_id=model.tokenizer.pad_token_id,
		eos_token_id=model.tokenizer.eos_token_id,
	)
	input_len = inputs_embeds.shape[1]
	if generated_ids.shape[1] > input_len:
		new_token_ids = generated_ids[:, input_len:]
	else:
		new_token_ids = generated_ids
	decoded = model.tokenizer.batch_decode(new_token_ids, skip_special_tokens=False)
	return [_parse_generated_yes_no(text) for text in decoded]


def _evaluate_yes_no_accuracy(
	model: SceneQwenVLA,
	loader,
	*,
	device: torch.device,
	dtype: torch.dtype,
	max_samples: int,
	max_prompt_length: int,
	max_answer_length: int,
) -> dict[str, float]:
	if max_samples <= 0:
		return {"accuracy": 0.0, "num_samples": 0.0}

	total = 0
	correct = 0
	was_training = model.training
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			features, prompts, answers = _flatten_batch(batch)
			if not prompts:
				continue

			remaining = max_samples - total
			if remaining <= 0:
				break
			if len(prompts) > remaining:
				prompts = prompts[:remaining]
				answers = answers[:remaining]
				features = {key: value[:remaining] for key, value in features.items()}

			prompt_ids, _ = _tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				device=device,
				max_prompt_length=max_prompt_length,
				max_answer_length=max_answer_length,
			)
			features = _move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				predictions = _generate_yes_no_predictions(
					model,
					features,
					prompt_ids,
					device=device,
					max_new_tokens=max_answer_length,
				)

			target_is_yes = [answer == YES_TOKEN for answer in answers]
			correct += sum(int(pred is not None and pred == target) for pred, target in zip(predictions, target_is_yes))
			total += len(answers)
			if total >= max_samples:
				break
	if was_training:
		model.train()
	accuracy = float(correct / total) if total > 0 else 0.0
	return {"accuracy": accuracy, "num_samples": float(total)}


def save_checkpoint(output_dir: str, step: int, model: SceneQwenVLA, optimizer: torch.optim.Optimizer, scheduler: Any) -> None:
	ckpt_dir = Path(output_dir) / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
	torch.save(
		{
			"step": step,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		},
		ckpt_path,
	)


def run_training(cfg: TrainQAConfig) -> None:
	device = _resolve_device()
	dtype = _resolve_dtype(cfg.dtype)
	torch.backends.cuda.matmul.allow_tf32 = True

	output_dir = Path(cfg.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	loader = build_qa_dataloader(
		cfg.cache_dir,
		file_indices=cfg.file_indices,
		qa_dir=cfg.qa_dir,
		batch_size=cfg.batch_size,
		shuffle_seed=cfg.shuffle_seed,
		num_workers=cfg.num_workers,
		pin_memory=cfg.pin_memory,
	)

	model = SceneQwenVLA(qwen_name=cfg.qwen_name)
	_ensure_tokenizer(model)
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
			name=cfg.wandb_run_name,
			mode=cfg.wandb_mode,
			config=dataclasses.asdict(cfg),
		)
		wandb_run.define_metric("train/step")
		wandb_run.define_metric("train/*", step_metric="train/step")
		wandb_run.define_metric("eval/*", step_metric="train/step")

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	scheduler = None
	if cfg.max_steps is not None:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_steps))

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
		for batch in loader:
			features, prompts, answers = _flatten_batch(batch)
			if not prompts:
				continue

			prompt_ids, answer_ids = _tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				device=device,
				max_prompt_length=cfg.max_prompt_length,
				max_answer_length=cfg.max_answer_length,
			)
			features = _move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				outputs = model(features, prompt_ids, answer_ids=answer_ids)
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
				save_checkpoint(cfg.output_dir, global_step, model, optimizer, scheduler)

			global_step += 1
			pbar.update(1)
			if cfg.max_steps is not None and global_step >= cfg.max_steps:
				break
		pbar.close()
		if cfg.max_steps is not None and global_step >= cfg.max_steps:
			break

		eval_metrics = _evaluate_yes_no_accuracy(
			model,
			loader,
			device=device,
			dtype=dtype,
			max_samples=cfg.eval_num_samples,
			max_prompt_length=cfg.max_prompt_length,
			max_answer_length=cfg.max_answer_length,
		)
		tqdm.write(
			f"epoch={epoch} eval_accuracy={eval_metrics['accuracy']:.4f} "
			f"eval_samples={int(eval_metrics['num_samples'])}"
		)
		if wandb_run is not None:
			wandb_run.log(
				{
					"train/step": global_step,
					"eval/yes_no_accuracy": eval_metrics["accuracy"],
					"eval/num_samples": eval_metrics["num_samples"],
				},
				step=global_step,
			)

	save_pretrained = getattr(model.llm, "save_pretrained", None)
	if callable(save_pretrained):
		model.llm.save_pretrained(str(output_dir / "llm"))
	model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
	save_checkpoint(cfg.output_dir, global_step, model, optimizer, scheduler)
	if wandb_run is not None:
		wandb_run.finish()


def _parse_args() -> TrainQAConfig:
	parser = argparse.ArgumentParser(description="Train SceneQwenVLA on QA data.")
	parser.add_argument("--cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/qa_cache/")
	parser.add_argument("--tfrecord_dir", type=str, default=None, help="Deprecated alias for --cache_dir.")
	parser.add_argument("--qa_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/qa_dataset/")
	parser.add_argument("--file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--output_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/qa_output")
	parser.add_argument("--wandb_project", type=str, default="waymax_rs_qa")
	parser.add_argument("--wandb_run_name", type=str, default=None)
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
	parser.add_argument("--qwen_name", type=str, default="Qwen/Qwen3-0.6B")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--learning_rate", type=float, default=5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--num_epochs", type=int, default=1000)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--grad_accum_steps", type=int, default=1)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--shuffle_buffer_size", type=int, default=1024)
	parser.add_argument("--dataset_num_shards", type=int, default=1)
	parser.add_argument("--include_sdc_paths", action="store_true")
	parser.add_argument("--max_num_objects", type=int, default=64)
	parser.add_argument("--anchor_step_override", type=int, default=0)
	parser.add_argument("--freeze_llm", action="store_true", default=True)
	parser.add_argument("--no_gradient_checkpointing", action="store_true")
	parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16"))
	parser.add_argument("--log_every", type=int, default=10)
	parser.add_argument("--save_every", type=int, default=1000)
	parser.add_argument("--eval_num_samples", type=int, default=32)
	parser.add_argument("--max_prompt_length", type=int, default=128)
	parser.add_argument("--max_answer_length", type=int, default=4)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--no_pin_memory", action="store_true")
	args = parser.parse_args()

	return TrainQAConfig(
		cache_dir=args.cache_dir or args.tfrecord_dir,
		qa_dir=args.qa_dir,
		file_indices=_parse_file_indices(args.file_indices),
		output_dir=args.output_dir,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		wandb_entity=args.wandb_entity,
		wandb_mode=args.wandb_mode,
		qwen_name=args.qwen_name,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		num_epochs=args.num_epochs,
		max_steps=args.max_steps,
		grad_accum_steps=args.grad_accum_steps,
		max_grad_norm=args.max_grad_norm,
		shuffle_seed=args.shuffle_seed,
		shuffle_buffer_size=args.shuffle_buffer_size,
		dataset_num_shards=args.dataset_num_shards,
		include_sdc_paths=args.include_sdc_paths,
		max_num_objects=args.max_num_objects,
		anchor_step_override=args.anchor_step_override,
		freeze_llm=args.freeze_llm,
		use_gradient_checkpointing=not args.no_gradient_checkpointing,
		dtype=args.dtype,
		log_every=args.log_every,
		save_every=args.save_every,
		eval_num_samples=args.eval_num_samples,
		max_prompt_length=args.max_prompt_length,
		max_answer_length=args.max_answer_length,
		num_workers=args.num_workers,
		pin_memory=not args.no_pin_memory,
	)


def main() -> None:
	config = _parse_args()
	run_training(config)


if __name__ == "__main__":
	main()
