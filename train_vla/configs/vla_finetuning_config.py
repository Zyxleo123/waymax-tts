from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from train_vla.configs.vla_pretrain_config import VLAPretrainConfig
from data.types import PreprocessConfig


SCENE_TOKENIZER_OVERRIDE_FIELDS = (
	"num_scene_tokens",
	"ego_dim",
	"goal_dim",
	"other_dim",
	"map_dim",
	"tl_dim",
	"scene_hidden_dim",
	"max_num_objects",
	"preprocess_cfg",
)


@dataclass(frozen=True)
class VLAFinetuningConfig:
	cache_dir: str
	annotation_dir: str
	file_indices: list[int] | None
	output_dir: str
	wandb_project: str | None = "finetune_vla"
	wandb_run_name: str | None = None
	wandb_entity: str | None = None
	wandb_mode: str = "online"
	model_type: str = "gemma"
	qwen_name: str = "Qwen/Qwen3-0.6B"
	gemma_name: str = "google/gemma-4-E2B-it"
	batch_size: int = 1
	learning_rate: float = 1e-5
	warmup_steps: int = 1000
	weight_decay: float = 0.01
	num_epochs: int = 1
	max_steps: int | None = None
	grad_accum_steps: int = 1
	max_grad_norm: float = 1.0
	shuffle_seed: int = 0
	shuffle_buffer_size: int = 1024
	dataset_num_shards: int = 1
	include_sdc_paths: bool = False
	max_num_objects: int = 128
	anchor_step: int | None = 10
	freeze_llm: bool = True
	use_gradient_checkpointing: bool = True
	dtype: str = "bf16"
	log_every: int = 10
	save_every: int = 1000
	eval_num_samples: int = 32
	validation_fraction: float = 0.2
	max_prompt_length: int = 128
	max_answer_length: int = 4
	num_workers: int = 0
	pin_memory: bool = True
	add_eos: bool = False
	num_scene_tokens: int = 32
	ego_dim: int = 5
	goal_dim: int = 3
	other_dim: int = 15
	map_dim: int = 25
	tl_dim: int = 11
	scene_hidden_dim: int = 512
	pretrained_model_path: str | None = None
	scene_tokenizer_ckpt: str | None = None
	use_lora: bool = True
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.05
	lora_target_modules: str | None = None

	preprocess_cfg: PreprocessConfig = PreprocessConfig()
	tag: str | None = None


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


def _resolve_pretrained_run_dir(pretrained_model_path: str | Path) -> Path:
	path = Path(pretrained_model_path)
	if path.is_file():
		if path.parent.name == "checkpoints":
			return path.parent.parent
		return path.parent
	if path.is_dir():
		if (path / "training_config.json").exists():
			return path
		if path.name == "checkpoints":
			return path.parent
		if (path / "checkpoints").exists() and (path / "training_config.json").exists():
			return path
	raise FileNotFoundError(f"Could not resolve pretrained run directory from: {pretrained_model_path}")


def _load_pretrain_config(pretrained_model_path: str | Path) -> VLAPretrainConfig:
	run_dir = _resolve_pretrained_run_dir(pretrained_model_path)
	config_path = run_dir / "training_config.json"
	if not config_path.exists():
		raise FileNotFoundError(f"Pretrain training_config.json not found: {config_path}")

	with config_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	valid_keys = {field.name for field in fields(VLAPretrainConfig)}
	filtered = {k: v for k, v in data.items() if k in valid_keys}
	return VLAPretrainConfig(**filtered)


def _with_scene_tokenizer_overrides(
	finetune_cfg: VLAFinetuningConfig,
	pretrain_cfg: VLAPretrainConfig,
) -> VLAFinetuningConfig:
	updated = asdict(finetune_cfg)
	pretrain_dict = asdict(pretrain_cfg)
	for key in SCENE_TOKENIZER_OVERRIDE_FIELDS:
		if key in pretrain_dict:
			updated[key] = pretrain_dict[key]
	return VLAFinetuningConfig(**updated)


def parse_args() -> VLAFinetuningConfig:
	parser = argparse.ArgumentParser(description="Finetune Scene VLA with LoRA.")
	parser.add_argument("--cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/sim_state_cache_npz/")
	parser.add_argument("--tfrecord_dir", type=str, default=None, help="Deprecated alias for --cache_dir.")
	parser.add_argument("--annotation_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/annotations/")
	parser.add_argument("--file_indices", type=str, nargs="*", default=None)
	parser.add_argument("--output_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/finetune_vla")
	parser.add_argument("--wandb_project", type=str, default="finetune_vla")
	parser.add_argument("--wandb_run_name", type=str, default="finetune_vla")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
	parser.add_argument("--model_type", type=str, default="gemma", choices=["qwen", "gemma", "old_qwen"])
	parser.add_argument("--qwen_name", type=str, default="Qwen/Qwen3-0.6B")
	parser.add_argument("--gemma_name", type=str, default="google/gemma-4-E2B-it")
	parser.add_argument("--num_scene_tokens", type=int, default=32)
	parser.add_argument("--ego_dim", type=int, default=5)
	parser.add_argument("--goal_dim", type=int, default=3)
	parser.add_argument("--other_dim", type=int, default=15)
	parser.add_argument("--map_dim", type=int, default=25)
	parser.add_argument("--tl_dim", type=int, default=11)
	parser.add_argument("--scene_hidden_dim", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--learning_rate", type=float, default=5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--num_epochs", type=int, default=1000)
	parser.add_argument("--warmup_steps", type=int, default=1000)
	parser.add_argument("--max_steps", type=int, default=None)
	parser.add_argument("--grad_accum_steps", type=int, default=1)
	parser.add_argument("--max_grad_norm", type=float, default=1.0)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--shuffle_buffer_size", type=int, default=1024)
	parser.add_argument("--dataset_num_shards", type=int, default=1)
	parser.add_argument("--include_sdc_paths", action="store_true")
	parser.add_argument("--max_num_objects", type=int, default=128)
	parser.add_argument("--anchor_step", type=int, default=10)
	parser.add_argument("--freeze_llm", action="store_true", default=True)
	parser.add_argument("--no_gradient_checkpointing", action="store_true")
	parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16"))
	parser.add_argument("--log_every", type=int, default=10)
	parser.add_argument("--save_every", type=int, default=5000)
	parser.add_argument("--eval_num_samples", type=int, default=500)
	parser.add_argument("--validation_fraction", type=float, default=0.001)
	parser.add_argument("--max_prompt_length", type=int, default=128)
	parser.add_argument("--max_answer_length", type=int, default=64)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--no_pin_memory", action="store_true")
	parser.add_argument("--add_eos", action="store_true", default=True)
	parser.add_argument("--pretrained_model_path", type=str, default=None)
	parser.add_argument(
		"--scene_tokenizer_ckpt",
		type=str,
		required=True,
		help="Checkpoint file or run/checkpoints directory used to initialize scene_tokenizer.",
	)
	parser.add_argument("--use_lora", action="store_true", default=True)
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument(
		"--lora_target_modules",
		type=str,
		default=None,
		help="Comma-separated LoRA target modules, e.g. q_proj,k_proj,v_proj,o_proj.",
	)
	parser.add_argument("--tag", type=str, default=None, help="Optional tag to add to wandb run name.")

	args = parser.parse_args()
	cfg = VLAFinetuningConfig(
		cache_dir=args.cache_dir or args.tfrecord_dir,
		annotation_dir=args.annotation_dir,
		file_indices=_parse_file_indices(args.file_indices),
		output_dir=args.output_dir,
		wandb_project=args.wandb_project,
		wandb_run_name=args.wandb_run_name,
		wandb_entity=args.wandb_entity,
		wandb_mode=args.wandb_mode,
		model_type=args.model_type,
		qwen_name=args.qwen_name,
		gemma_name=args.gemma_name,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		num_epochs=args.num_epochs,
		max_steps=args.max_steps,
		warmup_steps=args.warmup_steps,
		grad_accum_steps=args.grad_accum_steps,
		max_grad_norm=args.max_grad_norm,
		shuffle_seed=args.shuffle_seed,
		shuffle_buffer_size=args.shuffle_buffer_size,
		dataset_num_shards=args.dataset_num_shards,
		include_sdc_paths=args.include_sdc_paths,
		max_num_objects=args.max_num_objects,
		anchor_step=args.anchor_step,
		freeze_llm=args.freeze_llm,
		use_gradient_checkpointing=not args.no_gradient_checkpointing,
		dtype=args.dtype,
		log_every=args.log_every,
		save_every=args.save_every,
		eval_num_samples=args.eval_num_samples,
		validation_fraction=args.validation_fraction,
		max_prompt_length=args.max_prompt_length,
		max_answer_length=args.max_answer_length,
		num_workers=args.num_workers,
		pin_memory=not args.no_pin_memory,
		add_eos=args.add_eos,
		num_scene_tokens=args.num_scene_tokens,
		ego_dim=args.ego_dim,
		goal_dim=args.goal_dim,
		other_dim=args.other_dim,
		map_dim=args.map_dim,
		tl_dim=args.tl_dim,
		scene_hidden_dim=args.scene_hidden_dim,
		pretrained_model_path=args.pretrained_model_path,
		scene_tokenizer_ckpt=args.scene_tokenizer_ckpt,
		use_lora=args.use_lora,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		lora_target_modules=args.lora_target_modules,
		tag=args.tag,
	)

	if cfg.pretrained_model_path:
		pretrain_cfg = _load_pretrain_config(cfg.pretrained_model_path)
		cfg = _with_scene_tokenizer_overrides(cfg, pretrain_cfg)
		if not cfg.scene_tokenizer_ckpt:
			cfg = VLAFinetuningConfig(**{**asdict(cfg), "scene_tokenizer_ckpt": str(cfg.pretrained_model_path)})

	return cfg

