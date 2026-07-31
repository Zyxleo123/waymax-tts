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
from vmax.simulator.wrappers.reward import OFFROAD_MARGIN_M as _DEFAULT_OFFROAD_MARGIN_M


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
                elif key.startswith(("il_", "rl_")) and (
                    "ep_rew_mean" in key or "ep_len_mean" in key or "loss" in key
                ):
                    name = f"rollout/{key}"
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


def _apply_reward_overrides(config: dict, args: argparse.Namespace) -> None:
    """Patch termination keys + reward weights in place.

    By default (policy-friendly sim) we decouple overlap/offroad termination and
    downweight those penalties so policy rollouts — which deviate from the human
    log timing — get a learnable progression signal.  Legacy log-replay behaviour
    is restored with ``--log-replay-agents``.
    """
    from rl.vmax_rl import env_utils
    from vmax.simulator.wrappers import reward as reward_wrapper

    if not args.log_replay_agents:
        env_utils.apply_policy_friendly_reward_config(config)

    if args.termination_keys is not None:
        # Empty list (``--termination-keys`` with no args) => never terminate early.
        config["termination_keys"] = list(args.termination_keys)
    reward_cfg = config["reward_config"]
    weight_flags = {
        "overlap": args.r_overlap,
        "offroad": args.r_offroad,
        "off_route": args.r_off_route,
        "red_light": args.r_red_light,
        "progression": args.r_progression,
    }
    for key, value in weight_flags.items():
        if value is not None:
            if key not in reward_cfg:
                raise KeyError(
                    f"reward_config has no key '{key}' (keys: {sorted(reward_cfg)}); "
                    f"cannot apply --r-{key} override."
                )
            reward_cfg[key] = value

    # offroad_margin is not part of V-Max's stock reward_config, so it is added
    # rather than overridden.
    if args.r_offroad_margin is not None:
        reward_cfg["offroad_margin"] = args.r_offroad_margin
    if args.offroad_margin_m is not None:
        reward_wrapper.OFFROAD_MARGIN_M = args.offroad_margin_m


def main() -> None:
    args = _parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    model_dir = os.path.join(args.save_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    config = _build_vmax_config(args)
    _apply_reward_overrides(config, args)
    alg = config["algorithm"]
    network_config = config["network"]
    if network_config["encoder"]["type"] != "none":
        network_config["unflatten_config"] = config["observation_config"]

    num_devices = jax.local_device_count()
    print(f"[refine] devices={num_devices} backend={jax.default_backend()}")

    # --- Environment (bicycle dynamics, vec obs, linear reward, real SDC paths) ---
    env = simulator.make_env_for_training(
        max_num_objects=args.max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,  # use the dataset's curated SDC route paths
        observation_type=config["observation_type"],
        observation_config=config["observation_config"],
        reward_type=config["reward_type"],
        reward_config=config["reward_config"],
        termination_keys=config["termination_keys"],
    )
    use_reactive = not args.log_replay_agents
    if use_reactive:
        from rl.vmax_rl import env_utils

        env = env_utils.attach_idm_sim_agents(env, desired_vel=args.idm_desired_vel)
        print(f"[refine] Reactive IDM sim-agents ON (desired_vel={args.idm_desired_vel}).")
    else:
        print("[refine] Legacy log-replay sim-agents (--log-replay-agents).")

    print(f"[refine] termination_keys={config['termination_keys']} reward_config={config['reward_config']}")

    # --- Optional warm start from a pretrained checkpoint ---
    init_params = None
    if args.init_params:
        import pickle

        with open(args.init_params, "rb") as f:
            init_params = pickle.load(f)
        print(f"[refine] Warm start from {args.init_params} ({type(init_params).__name__})")

    # --- Data: failure cases (RL) + imitation source (expert stream OR failures) ---
    print("[refine] Loading failure scenarios...")
    failure_state, num_failures = data.load_failure_scenarios(
        args.failure_dir, max_num_objects=args.max_num_objects, limit=args.limit_failures,
        refit_sdc_log=args.refit_sdc_log,
    )
    rl_gen = data.make_failure_generator(
        failure_state,
        num_envs=args.num_envs,
        num_episode_per_epoch=args.num_episode_per_epoch,
        num_devices=num_devices,
        seed=args.seed,
    )

    if args.bc_source == "none":
        # Pure SAC on the failure cases (no behavior-cloning steps at all).
        print("[refine] BC source = none (pure SAC, no behavior cloning).")
        excl: set[int] = set()
        expert_path = None
        expert_gen = None
    elif args.bc_source == "failures":
        # Classic BC_SAC co-training: behavior-clone the logged expert trajectory of
        # the SAME failure scenarios SAC is solving (BC and SAC on one scenario set).
        print("[refine] BC source = failures (BC and SAC both on failure cases).")
        excl = set()
        expert_path = None
        expert_gen = data.make_failure_generator(
            failure_state,
            num_envs=args.num_envs,
            num_episode_per_epoch=args.num_episode_per_epoch,
            num_devices=num_devices,
            seed=args.seed + 1000,  # independent sampling stream from the RL half
        )
    else:
        # Dual-source: behavior-clone a disjoint non-failure expert stream so the
        # failure scenarios stay "unseen" by BC.
        excl = data.failure_shard_indices(args.failure_dir, total_shards=args.total_shards)
        print(f"[refine] BC source = expert. Excluding {len(excl)} failure shards: {sorted(excl)}")
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
        "bc_source": args.bc_source,
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
        "reactive_agents": not args.log_replay_agents,
        "log_replay_agents": args.log_replay_agents,
        "idm_desired_vel": args.idm_desired_vel,
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
        init_params=init_params,
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
    p.add_argument("--refit-sdc-log", action="store_true",
                   help="Refit each SDC's logged yaw/velocity to be bicycle-consistent "
                        "(optional/defensive; the decisive tracking fix is the per-env SDC "
                        "action selection in inference.expert_step).")
    p.add_argument("--bc-source", choices=["expert", "failures", "none"], default="expert",
                   help="Imitation (BC) data source: 'expert' = disjoint non-failure WOMD shards "
                        "(failures stay unseen by BC); 'failures' = the same failure cases SAC trains on "
                        "(classic BC_SAC co-training); 'none' = pure SAC, no behavior cloning. "
                        "'failures'/'none' ignore --womd-dir.")
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
    # Policy-friendly sim (default): IDM agents + softer overlap/offroad handling.
    p.add_argument("--log-replay-agents", action="store_true",
                   help="Legacy sim: rigid log-replay non-ego agents and V-Max default "
                        "overlap/offroad termination/penalties. Default is reactive IDM "
                        "+ decoupled collision/offroad (learnable policy rollouts).")
    p.add_argument("--termination-keys", nargs="*", default=None,
                   help="Override which metrics end an episode early. Pass with no args "
                        "(`--termination-keys`) to disable early termination entirely. "
                        "Default (policy-friendly): run_red_light only.")
    p.add_argument("--r-overlap", type=float, default=None, help="Collision penalty weight.")
    p.add_argument("--r-offroad", type=float, default=None, help="Offroad penalty weight.")
    p.add_argument("--init-params", type=str, default=None,
                   help="Warm start actor+critic from a V-Max checkpoint .pkl (SAC or BC_SAC "
                        "params). Without this, refine trains from RANDOM weights. The "
                        "checkpoint's network/observation config must match this run's — "
                        "e.g. a SAC run with layer_sizes [256,256] needs "
                        "--override algorithm.network.policy.layer_sizes=[256,256] "
                        "--override algorithm.network.value.layer_sizes=[256,256].")
    p.add_argument("--r-offroad-margin", type=float, default=None,
                   help="Weight for the continuous road-edge-clearance penalty (e.g. -0.5). "
                        "Unlike --r-offroad (a step function that only fires once the SDC is "
                        "already offroad) this ramps up as clearance shrinks, so the policy is "
                        "paid to keep a margin. Adds the 'offroad_margin' reward term.")
    p.add_argument("--offroad-margin-m", type=float, default=None,
                   help="Clearance (m) at which --r-offroad-margin starts to bite "
                        f"(default {_DEFAULT_OFFROAD_MARGIN_M}).")
    p.add_argument("--r-off-route", type=float, default=None, help="Off-route penalty weight.")
    p.add_argument("--r-red-light", type=float, default=None, help="Red-light penalty weight.")
    p.add_argument("--r-progression", type=float, default=None, help="Route-progression reward weight.")
    p.add_argument("--reactive-agents", action="store_true",
                   help="(Deprecated: now default.) Force reactive IDM; use --log-replay-agents to disable.")
    p.add_argument("--idm-desired-vel", type=float, default=30.0,
                   help="IDM free-road desired speed (m/s) when using reactive sim-agents.")
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
