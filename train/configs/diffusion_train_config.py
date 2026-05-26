from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class DiffusionTrainConfig:
	cache_dir: str
	anchor_step: int | str = 'all'
	file_indices: list[int] | None = None
	cache_batch_size: int = 64
	shuffle_seed: int = 0
	instruction_seed: int = 0
	num_workers: int = 0
	pin_memory: bool = False

	pretrained_checkpoint: str | None = None
	resume_path: str | None = None
	use_ema: bool = True

	save_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/vla/train_diffusion"
	epochs: int = 20
	steps_per_epoch: int = 1000
	save_every: int = 5
	log_every: int = 50
	seed: int = 0

	pbar_every: int = 10
	log_jsonl_path: str = "./train_logs.jsonl"

	lr: float = 1e-4
	weight_decay: float = 1e-4
	warmup_steps: int = 500
	grad_clip_norm: float = 1.0
	ema_decay: float = 0.999

	jax_compilation_cache_dir: str = "./.jax_compilation_cache"

	# wandb options
	wandb_project: str = "waymax_vla"
	wandb_name: str = "train_diffusion"
	wandb_entity: str | None = None
	wandb_mode: str = "online"


def parse_args() -> DiffusionTrainConfig:
	parser = argparse.ArgumentParser()

	parser.add_argument("--cache_dir", type=str, required=True)
	parser.add_argument("--anchor_step", type=str, default='all')
	parser.add_argument("--file_indices", type=int, nargs="*", default=None)
	parser.add_argument("--cache_batch_size", type=int, default=64)
	parser.add_argument("--shuffle_seed", type=int, default=0)
	parser.add_argument("--instruction_seed", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", action="store_true")

	parser.add_argument("--pretrained_checkpoint", type=str, default=None)
	parser.add_argument("--resume_path", type=str, default=None)
	parser.add_argument("--use_ema", action="store_true", default=True)
	parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")

	parser.add_argument("--save_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/train_diffusion")
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--steps_per_epoch", type=int, default=1000)
	parser.add_argument("--save_every", type=int, default=10)
	parser.add_argument("--log_every", type=int, default=50)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--pbar_every", type=int, default=10)

	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--warmup_steps", type=int, default=500)
	parser.add_argument("--grad_clip_norm", type=float, default=1.0)
	parser.add_argument("--ema_decay", type=float, default=0.999)

	parser.add_argument("--jax_compilation_cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/.jax_compilation_cache")

	parser.add_argument("--wandb_project", type=str, default="pretrain_diffusion")
	parser.add_argument("--wandb_name", type=str, default="train_diffusion")
	parser.add_argument("--wandb_entity", type=str, default=None)
	parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
	args = parser.parse_args()
	args.anchor_step = int(args.anchor_step) if args.anchor_step.isdigit() else args.anchor_step
	return DiffusionTrainConfig(**vars(args))


def config_to_dict(config: DiffusionTrainConfig) -> dict:
	return asdict(config)