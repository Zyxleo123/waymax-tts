"""Entrypoint: BC warm-up (expert) + SAC RL (failure cases) with V-Max BC_SAC.

Example (smoke, 1 GPU)::

    python -m rl.vmax_rl.refine \
        --failure-dir /zfsauton/scratch/mineuih/waymax_rs/failure_samples \
        --womd-dir /zfsauton/scratch/eshau/womd/tf_example/training \
        --save-dir /zfsauton/scratch/yixiz/waymax_rs/runs/bcsac_smoke \
        --total-timesteps 20000 --num-envs 8 --num-episode-per-epoch 2 \
        --learning-start 500 --buffer-size 50000 --limit-failures 32

The non-failure WOMD shards (everything except the shards that contain failure
cases) feed the behavior-cloning half; the harvested failure cases feed the SAC
half. Both run through V-Max's JAX pipeline on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import os

from rl.vmax_rl import compat  # noqa: F401

import jax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.scripts.training import train_utils

from rl.vmax_rl import bc_sac_dual, data


def _make_progress_fn(writer, use_wandb: bool):
    """Progress callback that logs to TensorBoard and (optionally) wandb.

    Called as ``progress(step, metrics, total_timesteps)`` during training and
    ``progress(step, metrics)`` during evaluation (``total_timesteps`` omitted).
    """
    def progress(num_steps, metrics, total_timesteps=None):
        train_utils.log_metrics(num_steps, metrics, total_timesteps, writer=writer)
        if use_wandb:
            import wandb

            is_eval = total_timesteps is None
            payload = {}
            for key, value in metrics.items():
                if "/" in key:
                    name = key
                elif is_eval:
                    name = f"evaluation/{key}"
                else:
                    name = f"rollout/{key}" if ("ep_len_mean" in key or "ep_rew_mean" in key) else f"metrics/{key}"
                try:
                    payload[name] = float(value)
                except (TypeError, ValueError):
                    continue
            if payload:
                wandb.log(payload, step=int(num_steps))

    return progress


def _build_vmax_config(args: argparse.Namespace) -> dict:
    """Compose the V-Max config (obs/reward/network/algorithm) via Hydra."""
    overrides = [
        "algorithm=bc_sac",
        "path_dataset=__unused__",
        f"max_num_objects={args.max_num_objects}",
        f"scenario_length={args.scenario_length}",
        f"num_envs={args.num_envs}",
        f"num_episode_per_epoch={args.num_episode_per_epoch}",
        f"total_timesteps={args.total_timesteps}",
        f"seed={args.seed}",
        f"algorithm.learning_start={args.learning_start}",
        f"algorithm.buffer_size={args.buffer_size}",
        f"algorithm.batch_size={args.batch_size}",
        f"algorithm.rl_learning_rate={args.rl_lr}",
        f"algorithm.imitation_learning_rate={args.imitation_lr}",
        f"algorithm.imitation_frequency={args.imitation_frequency}",
        f"algorithm.alpha={args.alpha}",
        f"algorithm.discount={args.gamma}",
    ]
    overrides += list(args.override or [])
    with initialize_config_dir(config_dir=PATH_TO_APP + "/config", version_base=None):
        cfg = compose(config_name="base_config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def main() -> None:
    args = _parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    model_dir = os.path.join(args.save_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    config = _build_vmax_config(args)
    alg = config["algorithm"]
    network_config = config["network"]
    if network_config["encoder"]["type"] != "none":
        network_config["unflatten_config"] = config["observation_config"]

    num_devices = jax.local_device_count()
    print(f"[refine] devices={num_devices} backend={jax.default_backend()}")

    # --- Environment (bicycle dynamics, vec obs, linear reward, waymo SDC paths) ---
    env = simulator.make_env_for_training(
        max_num_objects=args.max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=False,  # waymo_dataset mode: SDC route generated on reset
        observation_type=config["observation_type"],
        observation_config=config["observation_config"],
        reward_type=config["reward_type"],
        reward_config=config["reward_config"],
        termination_keys=config["termination_keys"],
    )

    # --- Data: failure cases (RL) + non-failure expert stream (imitation) ---
    print("[refine] Loading failure scenarios...")
    failure_state, num_failures = data.load_failure_scenarios(
        args.failure_dir, max_num_objects=args.max_num_objects, limit=args.limit_failures
    )
    rl_gen = data.make_failure_generator(
        failure_state,
        num_envs=args.num_envs,
        num_episode_per_epoch=args.num_episode_per_epoch,
        num_devices=num_devices,
        seed=args.seed,
    )

    excl = data.failure_shard_indices(args.failure_dir, total_shards=args.total_shards)
    print(f"[refine] Excluding {len(excl)} failure shards from BC: {sorted(excl)}")
    expert_path = data.build_expert_shard_path(
        args.womd_dir,
        exclude_shards=excl,
        total_shards=args.total_shards,
        work_dir=os.path.join(args.save_dir, "expert_shards"),
    )
    print(f"[refine] Expert dataset path: {expert_path}")
    expert_gen = data.make_expert_generator(
        expert_path,
        max_num_objects=args.max_num_objects,
        num_envs=args.num_envs,
        num_episode_per_epoch=args.num_episode_per_epoch,
        seed=args.seed + 1,
    )

    # --- Logging ---
    writer = train_utils.setup_tensorboard(args.save_dir)

    # --- Persist the resolved run config ---
    run_meta = {
        "algorithm": "BC_SAC_dual",
        "algorithm_name": alg["name"],
        "num_failures": num_failures,
        "excluded_shards": sorted(excl),
        "expert_path": expert_path,
        "max_num_objects": args.max_num_objects,
        "scenario_length": args.scenario_length,
        "observation_type": config["observation_type"],
        "observation_config": config["observation_config"],
        "reward_type": config["reward_type"],
        "reward_config": config["reward_config"],
        "termination_keys": config["termination_keys"],
        "network_config": network_config,
        "args": vars(args),
        "algorithm_config": alg,
    }
    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or os.path.basename(os.path.normpath(args.save_dir)),
            group=args.wandb_group,
            tags=args.wandb_tags or None,
            mode=args.wandb_mode,
            dir=args.save_dir,
            config=run_meta,
        )
        # Record identifiers so `evaluate.py` can log into the same run.
        run_meta["wandb"] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_id": wandb.run.id,
        }
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    progress = _make_progress_fn(writer, use_wandb)

    # --- Train ---
    bc_sac_dual.train(
        env=env,
        rl_data_generator=rl_gen,
        imitation_data_generator=expert_gen,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_episode_per_epoch=args.num_episode_per_epoch,
        scenario_length=args.scenario_length,
        log_freq=args.log_freq,
        seed=args.seed,
        learning_start=alg["learning_start"],
        alpha=alg["alpha"],
        discount=alg["discount"],
        tau=alg["tau"],
        imitation_frequency=alg["imitation_frequency"],
        imitation_unroll_length=alg["imitation_unroll_length"],
        loss_type=alg["loss_type"],
        save_freq=args.save_freq,
        buffer_size=alg["buffer_size"],
        batch_size=alg["batch_size"],
        rl_learning_rate=alg["rl_learning_rate"],
        imitation_learning_rate=alg["imitation_learning_rate"],
        grad_updates_per_step=alg["grad_updates_per_step"],
        unroll_length=alg["unroll_length"],
        network_config=network_config,
        progress_fn=progress,
        checkpoint_logdir=model_dir,
        disable_tqdm=not os.isatty(1),
    )
    if use_wandb:
        import wandb

        wandb.finish()
    print(f"[refine] Done. Checkpoints in {model_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC_SAC refinement (V-Max) on Waymax failure cases.")
    # Data
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--womd-dir", default="/zfsauton/scratch/eshau/womd/tf_example/training")
    p.add_argument("--total-shards", type=int, default=1000)
    p.add_argument("--limit-failures", type=int, default=None, help="Cap failure scenarios (smoke tests).")
    p.add_argument("--save-dir", required=True)
    # Run scale
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-episode-per-epoch", type=int, default=4)
    p.add_argument("--scenario-length", type=int, default=80)
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-freq", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=200)
    # Algorithm knobs (override the bc_sac.yaml defaults)
    p.add_argument("--learning-start", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rl-lr", type=float, default=1e-4)
    p.add_argument("--imitation-lr", type=float, default=5e-5)
    p.add_argument("--imitation-frequency", type=int, default=8,
                   help="1 of every N iters is a BC/imitation step (rest are SAC).")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--override", action="append", default=[],
                   help="Extra raw Hydra overrides, e.g. --override reward_config.overlap=-2.0")
    # Weights & Biases (opt-in; mirrors the same metrics as TensorBoard)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="vmax-bcsac")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None, help="Run name (default: save-dir basename).")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


if __name__ == "__main__":
    main()
