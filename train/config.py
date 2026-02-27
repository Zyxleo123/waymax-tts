from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    seed: int = 0
    tfrecord_path: str = ""
    max_num_objects: int = 32
    # Per-device batch size when data_parallel is enabled; global batch otherwise.
    batch_size: int = 4096
    shuffle_seed: int | None = 0
    shuffle_buffer_size: int = 1024

    epochs: int = 100
    steps_per_epoch: int = 1000
    save_every: int = 10
    save_dir: str = "./train/checkpoints"
    resume_path: str | None = None

    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    grad_clip_norm: float = 0.5

    hidden_dim: int = 256
    cond_dim: int = 256
    target_dim: int = 5
    predict_horizon: int = 25
    map_attr_dim: int = 4 + 21
    tl_attr_dim: int = 2 + 7
    predict_type: str = "v"
    model_dt: float = 0.2

    max_range: float = 100.0
    ego_range: float = 100.0
    max_velocity: float = 25.0
    max_width: float = 10.0
    max_map_points: int = 1024
    max_tl_points: int = 16
    num_map_type_classes: int = 21

    log_every: int = 100
    log_jsonl_path: str = "./train_logs.jsonl"
    wandb_project: str = "reward_search"
    wandb_name: str = "diffusion_jax"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    data_parallel: bool = False
    data_parallel_fallback: bool = True

    transfer_profile_steps: int = 0
    transfer_profile_max_d2h_events: int = 0
    transfer_profile_dir: str = "./transfer_profiles"



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfrecord_path", type=str, required=True)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--shuffle_buffer_size", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./train/checkpoints")
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--target_dim", type=int, default=5)
    parser.add_argument("--predict_horizon", type=int, default=25)
    parser.add_argument("--map_attr_dim", type=int, default=25)
    parser.add_argument("--tl_attr_dim", type=int, default=9)
    parser.add_argument("--predict_type", type=str, default="v", choices=["eps", "mu", "v"])
    parser.add_argument("--model_dt", type=float, default=0.2)

    parser.add_argument("--max_range", type=float, default=100.0)
    parser.add_argument("--ego_range", type=float, default=100.0)
    parser.add_argument("--max_velocity", type=float, default=25.0)
    parser.add_argument("--max_width", type=float, default=10.0)
    parser.add_argument("--max_map_points", type=int, default=1024)
    parser.add_argument("--max_tl_points", type=int, default=16)
    parser.add_argument("--num_map_type_classes", type=int, default=21)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_jsonl_path", type=str, default="./train_logs.jsonl")
    parser.add_argument("--wandb_project", type=str, default="reward_search")
    parser.add_argument("--wandb_name", type=str, default="diffusion_jax")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--no_data_parallel_fallback", action="store_true")
    parser.add_argument("--transfer_profile_steps", type=int, default=0)
    parser.add_argument("--transfer_profile_max_d2h_events", type=int, default=0)
    parser.add_argument("--transfer_profile_dir", type=str, default="./transfer_profiles")

    args = parser.parse_args()
    values = vars(args)
    no_fallback = values.pop("no_data_parallel_fallback")
    cfg = TrainConfig(**values)
    cfg.data_parallel_fallback = not no_fallback
    return cfg


def config_to_dict(config: TrainConfig) -> dict:
    return asdict(config)
