from __future__ import annotations

import dataclasses
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.vla.gemma_vla import VecSceneGemmaVLA
from model.vla.qwen_vla import VecSceneQwenVLA
from train_vla.configs.vla_pretrain_config import VLAPretrainConfig, parse_args

def resolve_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_name: str) -> torch.dtype:
	if dtype_name == "bf16":
		return torch.bfloat16
	if dtype_name == "fp16":
		return torch.float16
	raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_tokenizer(model: Any) -> None:
	if model.tokenizer.pad_token is None:
		model.tokenizer.pad_token = model.tokenizer.eos_token


def qa_to_text(qa_item: Mapping[str, Any]) -> tuple[str, str]:
	question = str(qa_item["question"])
	answer = str(qa_item["answer"])
	prompt = f"{question}"
	# answer_token = YES_TOKEN if answer == "yes" else NO_TOKEN
	return prompt, answer



def flatten_batch(batch) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
	if batch.qa is None:
		raise ValueError("batch.qa is required for QA training")

	feature_keys = list(batch.features.keys())
	flattened_features: dict[str, list[torch.Tensor]] = {key: [] for key in feature_keys}
	prompts: list[str] = []
	answers: list[str] = []
	qa_keys: list[str] = []

	for scenario_idx, scenario_qa in enumerate(batch.qa):
		qas = scenario_qa["qas"][:1]
		for qa_item in qas:
			prompt, answer = qa_to_text(qa_item)
			prompts.append(prompt)
			answers.append(answer)
			qa_keys.append(str(qa_item.get("key", "unknown")))
			for key in feature_keys:
				flattened_features[key].append(batch.features[key][scenario_idx])

	stacked_features = {key: torch.stack(values, dim=0) for key, values in flattened_features.items()}
	return stacked_features, prompts, answers, qa_keys


def tokenize_text_batch(tokenizer, prompts: list[str], answers: list[str], *, device: torch.device, max_prompt_length: int, max_answer_length: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def move_features_to_device(features: Mapping[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
	moved: dict[str, torch.Tensor] = {}
	for key, value in features.items():
		if value.dtype.is_floating_point:
			moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
		else:
			moved[key] = value.to(device=device, non_blocking=True)
	return moved


def extract_answer_text(text: str) -> str:
	"""Extract the answer from generated text by stripping whitespace.
	
	This works for any answer type: yes/no, numbers, positions, etc.
	"""
	return text.strip()



def build_qa_model(cfg: VLAPretrainConfig | Mapping[str, Any]) -> Any:
	if isinstance(cfg, VLAPretrainConfig):
		config = dataclasses.asdict(cfg)
	else:
		config = dict(cfg)
	model_type = config.get("model_type", "qwen")
	if model_type == "qwen":
		model = VecSceneQwenVLA(qwen_name=config["qwen_name"])
	elif model_type == "gemma":
		model = VecSceneGemmaVLA(gemma_name=config["gemma_name"])
	else:
		raise ValueError(f"Unknown model_type: {model_type}")
	ensure_tokenizer(model)
	return model


def save_training_config(cfg: VLAPretrainConfig, output_dir: str | Path) -> Path:
	config_path = Path(output_dir) / "training_config.json"
	config_path.parent.mkdir(parents=True, exist_ok=True)
	with config_path.open("w", encoding="utf-8") as f:
		json.dump(dataclasses.asdict(cfg), f, indent=2)
	return config_path


def load_training_config(output_dir: str | Path, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
	config_path = Path(output_dir) / "training_config.json"
	if config_path.exists():
		with config_path.open(encoding="utf-8") as f:
			return json.load(f)
	if checkpoint is not None:
		keys = ("model_type", "qwen_name", "gemma_name", "max_prompt_length", "max_answer_length")
		fallback = {key: checkpoint[key] for key in keys if key in checkpoint}
		if fallback:
			return fallback
	raise FileNotFoundError(
		f"No training_config.json in {output_dir} and checkpoint has no model metadata. "
		"Pass model_type and model name explicitly."
	)


def evaluate_answer_accuracy(
	model: Any,
	loader,
	*,
	device: torch.device,
	dtype: torch.dtype,
	max_samples: int,
	max_prompt_length: int,
	max_answer_length: int,
	model_type: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
	"""Evaluate model accuracy by comparing generated answers with dataset answers.
	
	Returns (overall_metrics, per_question_metrics) where:
	- overall_metrics: {"accuracy": float, "num_samples": float}
	- per_question_metrics: {question_key: {"accuracy": float, "num_samples": float}, ...}
	
	Supports any answer type: yes/no, numbers, positions, etc.
	Matches are case-insensitive and whitespace-normalized.
	"""
	if max_samples <= 0:
		return {"accuracy": 0.0, "num_samples": 0.0}, {}

	total = 0
	correct = 0
	per_question_stats: dict[str, dict[str, int]] = {}  # {q_key: {"correct": int, "total": int}}

	was_training = model.training
	model.eval()
	with torch.inference_mode():
		for batch in loader:
			features, prompts, answers, qa_keys = flatten_batch(batch)
			if not prompts:
				continue

			remaining = max_samples - total
			if remaining <= 0:
				break
			if len(prompts) > remaining:
				prompts = prompts[:remaining]
				answers = answers[:remaining]
				qa_keys = qa_keys[:remaining]
				features = {key: value[:remaining] for key, value in features.items()}

			prompt_ids, _ = tokenize_text_batch(
				model.tokenizer,
				prompts,
				answers,
				device=device,
				max_prompt_length=max_prompt_length,
				max_answer_length=max_answer_length,
			)
			features = move_features_to_device(features, device, dtype)

			amp_enabled = device.type == "cuda"
			with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
				predictions = model.generate_predictions(
					input_features=features,
					prompt_ids=prompt_ids,
					max_new_tokens=max_answer_length
				)

			# Compare predictions with ground truth answers (normalized)
			for pred, target, q_key in zip(predictions, answers, qa_keys):
				pred_norm = pred.lower().strip()
				target_norm = target.lower().strip()
				is_correct = pred_norm == target_norm
				if is_correct:
					correct += 1

				# Update per-question stats
				if q_key not in per_question_stats:
					per_question_stats[q_key] = {"correct": 0, "total": 0}
				per_question_stats[q_key]["total"] += 1
				if is_correct:
					per_question_stats[q_key]["correct"] += 1

			total += len(answers)
			if total >= max_samples:
				break

	if was_training:
		model.train()

	# Calculate overall accuracy
	overall_accuracy = float(correct / total) if total > 0 else 0.0
	overall_metrics = {"accuracy": overall_accuracy, "num_samples": float(total)}

	# Calculate per-question accuracy
	per_question_metrics: dict[str, dict[str, float]] = {}
	for q_key, stats in per_question_stats.items():
		q_accuracy = float(stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
		per_question_metrics[q_key] = {
			"accuracy": q_accuracy,
			"num_samples": float(stats["total"]),
		}

	return overall_metrics, per_question_metrics


def save_checkpoint(
	output_dir: str,
	step: int,
	model: Any,
	optimizer: torch.optim.Optimizer,
	scheduler: Any,
	*,
	cfg: VLAPretrainConfig | None = None,
) -> None:
	ckpt_dir = Path(output_dir) / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
	payload: dict[str, Any] = {
		"step": step,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
	}
	if cfg is not None:
		payload["model_type"] = cfg.model_type
		payload["qwen_name"] = cfg.qwen_name
		payload["gemma_name"] = cfg.gemma_name
		payload["max_prompt_length"] = cfg.max_prompt_length
		payload["max_answer_length"] = cfg.max_answer_length
	torch.save(payload, ckpt_path)