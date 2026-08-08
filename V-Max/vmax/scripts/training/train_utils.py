# Copyright 2025 Valeo.


"""Utility functions for training scripts."""

import json
import logging
import os
import pickle
import re
from argparse import ArgumentParser
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
from etils import epath


# tensorboardX is a training-only dependency, but this module is pulled in by the SAC
# package __init__, so an eager import makes the *inference* path unavailable in
# environments without it (the Diffusion-ES stack, which runs a V-Max policy in-process
# for online SAC initialization). setup_tensorboard raises if it is actually needed.
try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - depends on the environment

    class SummaryWriter:  # kept a class so the type annotations below stay valid
        def __init__(self, *args, **kwargs):
            raise ImportError("tensorboardX is required to log training to TensorBoard")

from vmax.simulator import datasets


logger = logging.getLogger(__name__)


def resolve_output_dir(
    algorithm_name: str,
    observation_type: str,
    encoder_config: dict,
    name_run: str,
    name_exp: str,
) -> str:
    """Determine the output directory based on training parameters.

    Args:
        algorithm_name: The name of the algorithm.
        observation_type: The observation type.
        reward_type: The reward type.
        encoder_config: The encoder configuration.
        name_run: The run name.
        name_exp: The experiment name.

    Returns:
        The output directory path.

    """
    _name_exp = "runs" if name_exp is None else name_exp

    if name_run is None:
        _name_run = f"{algorithm_name.upper()}_{observation_type.upper()}"

        encoder_type = encoder_config["type"]
        if encoder_type != "none":
            _name_run += f"_{encoder_type.upper()}"

        _name_run += f"_{datetime.now().strftime('%d-%m_%H:%M:%S')}/"
    else:
        _name_run = name_run

    return f"{_name_exp}/{_name_run}"


def apply_xla_flags(config: dict) -> None:
    """Apply XLA flags for performance, debugging, and caching.

    Args:
        config: Training configuration.

    """
    xla_flags = ""

    if config["debug_flag"]:
        xla_flags += "--xla_gpu_autotune_level=0 "  # SEEDING
    if config["perf_flag"]:
        xla_flags += "--xla_gpu_enable_pipelined_reduce_scatter=true "
        xla_flags += "--xla_gpu_enable_triton_softmax_fusion=true "
        xla_flags += "--xla_gpu_triton_gemm_any=true "

    os.environ["XLA_FLAGS"] = xla_flags

    if config["cache_flag"]:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def log_metrics(
    num_steps: int | None = None,
    metrics: dict | None = None,
    total_timesteps: int | None = None,
    writer: SummaryWriter = None,
) -> None:
    """Log and print training metrics and optionally send them to TensorBoard.

    Args:
        num_steps: Number of steps.
        metrics: Dictionary of metric names and values.
        total_timesteps: Total timesteps.
        writer: TensorBoard summary writer.

    """
    if total_timesteps is not None:
        logger.info(f"-> Step {num_steps}/{total_timesteps} - {(num_steps / total_timesteps) * 100:.2f}%")
        logger.info(f"-> Data time     : {metrics['runtime/data_time']:.2f}s")
        logger.info(f"-> Training time : {metrics['runtime/training_time']:.2f}s")
        logger.info(f"-> Log time      : {metrics['runtime/log_time']:.2f}s")
        logger.info(f"-> Eval time     : {metrics['runtime/eval_time']:.2f}s")

    for key, value in metrics.items():
        if writer:
            if total_timesteps is None:
                prefix = "evaluation/" if "/" not in key else ""
            else:
                prefix = "metrics/" if "/" not in key else ""
                if "ep_len_mean" in key or "ep_rew_mean" in key:
                    prefix = "rollout/"

            writer.add_scalar(f"{prefix}{key}", value, num_steps)
        logger.info(f"{key}: {value}")


def _wandb_scalar_name(key: str, *, is_eval: bool, total_timesteps: int | None) -> str:
    if "/" in key:
        return key
    if is_eval:
        return f"evaluation/{key}"
    if total_timesteps is not None and ("ep_len_mean" in key or "ep_rew_mean" in key):
        return f"rollout/{key}"
    return f"metrics/{key}"


def make_progress_fn(writer: SummaryWriter | None, *, use_wandb: bool = False):
    """Return a training progress callback for TensorBoard and optional wandb."""

    def progress(num_steps, metrics, total_timesteps=None):
        log_metrics(num_steps, metrics, total_timesteps, writer=writer)
        if not use_wandb:
            return

        import wandb

        is_eval = total_timesteps is None
        payload = {}
        for key, value in metrics.items():
            try:
                payload[_wandb_scalar_name(key, is_eval=is_eval, total_timesteps=total_timesteps)] = float(value)
            except (TypeError, ValueError):
                continue
        if payload:
            wandb.log(payload, step=int(num_steps))

    return progress


def setup_wandb(config: dict, run_path: str) -> dict | None:
    """Initialize wandb when config['wandb'] is true; return run metadata."""
    if not config.get("wandb"):
        return None

    import wandb

    run_name = config.get("wandb_name") or config.get("name_run")
    run = wandb.init(
        project=config.get("wandb_project", "vmax-womd"),
        entity=config.get("wandb_entity"),
        name=run_name,
        group=config.get("wandb_group"),
        mode=config.get("wandb_mode", "online"),
        config=config,
        dir=run_path,
    )
    meta = {
        "project": config.get("wandb_project", "vmax-womd"),
        "entity": config.get("wandb_entity"),
        "run_id": run.id,
        "run_name": run.name,
    }
    with open(os.path.join(run_path, "wandb_run.json"), "w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2)
    return meta


def finish_wandb(enabled: bool) -> None:
    if not enabled:
        return
    import wandb

    wandb.finish()


def log_eval_dir_to_wandb(run_dir: str, eval_dir: str) -> None:
    """Log final validation metrics to the training wandb run, if present."""
    try:
        meta_path = os.path.join(run_dir, "wandb_run.json")
        results_path = os.path.join(eval_dir, "evaluation_results.txt")
        if not os.path.isfile(meta_path) or not os.path.isfile(results_path):
            return

        import wandb

        with open(meta_path, encoding="utf-8") as fin:
            meta = json.load(fin)

        metrics = {}
        with open(results_path, encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2 and re.fullmatch(r"[-\d.]+", parts[1]):
                    metrics[f"val_full/{parts[0]}"] = float(parts[1])

        if not metrics:
            return

        wandb.init(
            project=meta.get("project"),
            entity=meta.get("entity"),
            id=meta.get("run_id"),
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        wandb.log(metrics)
        wandb.finish()
    except Exception as exc:
        logger.warning("wandb eval logging skipped: %s", exc)


def build_config_dicts(config: dict) -> tuple[dict, dict]:
    """Build separate configuration dictionaries for the environment and runtime.

    Args:
        config: The complete training configuration.

    Returns:
        A tuple containing the environment configuration and run configuration.

    """
    path_dataset = datasets.get_dataset(config["path_dataset"])
    path_dataset_eval = datasets.get_dataset(config["path_dataset_eval"])
    path_dataset_eval_failures = datasets.get_dataset(config.get("path_dataset_eval_failures"))

    sdc_paths_from_data = not config["waymo_dataset"]

    env_config = {
        "path_dataset": path_dataset,
        "path_dataset_eval": path_dataset_eval,
        "path_dataset_eval_failures": path_dataset_eval_failures,
        "sdc_paths_from_data": sdc_paths_from_data,
        "num_sdc_paths": config.get("num_sdc_paths"),
        "num_points_per_sdc_path": config.get("num_points_per_sdc_path"),
        "termination_keys": config["termination_keys"],
        "max_num_objects": config["max_num_objects"],
        "reward_type": config["reward_type"],
        "reward_config": config["reward_config"],
        "goal_shaping_config": config.get("goal_shaping"),
        "observation_type": config["observation_type"],
        "observation_config": config["observation_config"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
    }

    if config["network"]["encoder"]["type"] != "none":
        config["network"]["unflatten_config"] = config["observation_config"]

    network_config = config["network"]
    del config["algorithm"]["network"]

    if network_config["value"]["layer_sizes"] is None:
        del network_config["value"]

    run_config = {
        "total_timesteps": config["total_timesteps"],
        "scenario_length": config["scenario_length"],
        "log_freq": config["log_freq"],
        "save_freq": config["save_freq"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
        "eval_freq": config["eval_freq"],
        **config["algorithm"],
        "network_config": network_config,
    }
    del run_config["name"]

    if config["algorithm"]["name"] == "SAC":
        run_config["init_checkpoint"] = config.get("init_checkpoint")
        run_config["kl_coef"] = config.get("kl_coef", 0.0)
        run_config["kl_reference_checkpoint"] = config.get("kl_reference_checkpoint")

    return env_config, run_config


def print_hyperparameters(args: dict) -> None:
    """Print hyperparameters in a structured and readable format.

    Args:
        args: Dictionary of hyperparameters.

    """
    print(" Experiment Summary ".center(40, "="))
    print(f"- Algorithm          : {args['algorithm']['name']}")
    print(f"- Observation Type   : {args['observation_type']}")
    print(f"- Dataset Path       : {args['path_dataset']}")
    print(f"- Total Timesteps    : {args['total_timesteps']}")


def get_and_print_device_info() -> int:
    """Display and return the count of local JAX devices."""
    print(" Devices ".center(40, "="))
    print(f"- Backend: {jax.default_backend()}")
    print(f"- Devices: {jax.local_device_count()} -> {jax.local_devices()}")


def save_params(path: str, params: Any) -> None:
    """Serialize and save model parameters to a specified file."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


class BestCheckpointSaver:
    """Keep a single checkpoint: the one that maximizes ``min(reached_goal, accuracy)``.

    Scored on the **held-out evaluation set** (``path_dataset_eval``), never on the
    training rollouts. Two reasons, and they point the same way:

    * Generalization is the thing worth checkpointing on. The training-rollout metrics
      (logged under ``metrics/*``) are measured on the scenarios the policy is actively
      training on; the evaluation metrics (``evaluation/*``) come from a disjoint
      validation set.
    * ``accuracy`` only exists on the eval path anyway -- it is computed by
      ``metrics.collect`` only when ``termination_keys`` is passed, which the trainers do
      solely for evaluation. So there is no train-side ``accuracy`` to pair with.

    The ``eval_failures/*`` set, when configured, is deliberately ignored: it is a
    failure-case probe, not a held-out sample of the task.

    ``min`` (rather than a mean or a weighted sum) is the right aggregate for what we
    want here: a run that buys goal-reaching by shedding accuracy scores no better than
    its accuracy, so the checkpoint can never drift toward the aggressive-driving corner
    the way a sum would let it.
    """

    def __init__(
        self,
        checkpoint_logdir: str,
        metric_keys: Sequence[str] = ("reached_goal", "accuracy"),
        filename: str = "model_best.pkl",
    ) -> None:
        """Initialize the saver.

        Args:
            checkpoint_logdir: Directory the checkpoint is written to.
            metric_keys: Eval metrics to take the min over.
            filename: Checkpoint filename; overwritten in place on every improvement.
        """
        self._checkpoint_logdir = checkpoint_logdir
        self._metric_keys = tuple(metric_keys)
        self._path = f"{checkpoint_logdir}/{filename}"
        self._meta_path = f"{checkpoint_logdir}/{Path(filename).stem}.json"
        self._best_score = -float("inf")
        self._best_step = None

    def update(self, step: int, eval_metrics: dict, params: Any) -> dict:
        """Save ``params`` if this eval beats the best score so far, overwriting in place.

        Args:
            step: Current env step, recorded alongside the checkpoint.
            eval_metrics: Metrics from the held-out evaluation set.
            params: Unpmapped network params to serialize.

        Returns:
            Metrics to log (the running best, and the score of this eval). Empty when the
            required metrics are absent, so a trainer without them is simply a no-op.
        """
        missing = [key for key in self._metric_keys if key not in eval_metrics]
        if missing:
            return {}

        values = {key: float(eval_metrics[key]) for key in self._metric_keys}
        score = min(values.values())

        if score > self._best_score:
            self._best_score = score
            self._best_step = step

            save_params(self._path, params)
            with epath.Path(self._meta_path).open("w") as fout:
                json.dump({"step": step, "score": score, **values}, fout, indent=2)

        logged = {"best/score": score, "best/best_score": self._best_score}
        logged.update({f"best/{key}": value for key, value in values.items()})
        if self._best_step is not None:
            logged["best/best_step"] = self._best_step

        return logged

    @property
    def best_score(self) -> float:
        """Best score seen so far."""
        return self._best_score

    @property
    def best_step(self) -> int | None:
        """Env step at which the best score was seen."""
        return self._best_step


def setup_tensorboard(run_path: str) -> SummaryWriter:
    """Initialize and return a TensorBoard summary writer."""
    return SummaryWriter(log_dir=run_path)


def str2bool(v) -> bool:
    """Convert a string literal to a boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError("Boolean value expected.")
