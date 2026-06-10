from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from data.types import PreprocessConfig


@dataclass
class DiffusionPretrainConfig:
    seed: int = 0
    tfrecord_path: str = "/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord@1000"
    max_num_objects: int = 128
    batch_size: int = 64
    shuffle_seed: int | None = 0
    shuffle_buffer_size: int = 8
    tf_data_service_address: str | None = None
    dataset_num_shards: int = 2
    dataset_max_num_rg_points: int = 30000
    dataset_include_sdc_paths: bool = False

    epochs: int = 500
    steps_per_epoch: int = 1000
    save_every: int = 10
    save_dir: str = "/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain"
    resume_path: str | None = None

    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    grad_clip_norm: float = 1.0
    ema_update_every: int = 4
    goal_mask_prob: float = 0.3

    hidden_dim: int = 512
    cond_dim: int = 256
    target_dim: int = 5
    predict_horizon: int = 25
    map_attr_dim: int = 4 + 21
    tl_attr_dim: int = 2 + 9
    other_dim: int = 15
    inst_dim: int = 256
    predict_type: str = "v"
    model_dt: float = 0.2

    max_range: float = 100.0
    ego_range: float = 100.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_tl_points: int = 16
    num_map_type_classes: int = 21
    max_num_segments: int = 128
    max_points_per_segment: int = 128

    log_every: int = 100
    pbar_every: int = 10
    prefetch_size: int = 0
    recreate_dataloader_each_epoch: bool = True
    epoch_gc_every: int = 1
    log_jsonl_path: str = "./train_logs.jsonl"
    wandb_project: str = "waymax_vla"
    wandb_name: str = "pretrain_diffusion"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    data_parallel: bool = False
    jax_compilation_cache_dir: str = "./.jax_compilation_cache"

    preprocess_cfg: PreprocessConfig = PreprocessConfig()


def parse_args() -> DiffusionPretrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfrecord_path", type=str, default="/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord@1000")
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--shuffle_buffer_size", type=int, default=8)
    parser.add_argument("--tf_data_service_address", type=str, default=None)
    parser.add_argument("--dataset_num_shards", type=int, default=2)
    parser.add_argument("--dataset_max_num_rg_points", type=int, default=30000)
    parser.add_argument("--dataset_include_sdc_paths", action="store_true")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion")
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)
    parser.add_argument("--ema_update_every", type=int, default=4)
    parser.add_argument("--goal_mask_prob", type=float, default=0.3)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--target_dim", type=int, default=5)
    parser.add_argument("--predict_horizon", type=int, default=25)
    parser.add_argument("--map_attr_dim", type=int, default=25)
    parser.add_argument("--tl_attr_dim", type=int, default=11)
    parser.add_argument("--other_dim", type=int, default=15)
    parser.add_argument("--inst_dim", type=int, default=768)
    parser.add_argument("--predict_type", type=str, default="v", choices=["eps", "mu", "v"])
    parser.add_argument("--model_dt", type=float, default=0.2)

    parser.add_argument("--max_range", type=float, default=30.0)
    parser.add_argument("--ego_range", type=float, default=100.0)
    parser.add_argument("--max_velocity", type=float, default=25.0)
    parser.add_argument("--max_width", type=float, default=10.0)
    parser.add_argument("--max_tl_points", type=int, default=16)
    parser.add_argument("--num_map_type_classes", type=int, default=21)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--pbar_every", type=int, default=10)
    parser.add_argument("--prefetch_size", type=int, default=0)
    parser.add_argument("--recreate_dataloader_each_epoch", action="store_true")
    parser.add_argument("--no_recreate_dataloader_each_epoch", action="store_false", dest="recreate_dataloader_each_epoch")
    parser.set_defaults(recreate_dataloader_each_epoch=True)
    parser.add_argument("--epoch_gc_every", type=int, default=1)
    parser.add_argument("--log_jsonl_path", type=str, default="./train_logs.jsonl")
    parser.add_argument("--wandb_project", type=str, default="pretrain_diffusion")
    parser.add_argument("--wandb_name", type=str, default="pretrain_diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--jax_compilation_cache_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/.jax_compilation_cache")

    return DiffusionPretrainConfig(**vars(parser.parse_args()))


def config_to_dict(config: DiffusionPretrainConfig) -> dict:
    return asdict(config)
