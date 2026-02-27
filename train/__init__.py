"""JAX-native diffusion training utilities."""

from train.checkpoints import restore_checkpoint, save_checkpoint
from train.config import config_to_dict, parse_args


def train(*args, **kwargs):
    from train.train_diffusion import main

    return main(*args, **kwargs)

__all__ = [
    "config_to_dict",
    "parse_args",
    "restore_checkpoint",
    "save_checkpoint",
    "train",
]
