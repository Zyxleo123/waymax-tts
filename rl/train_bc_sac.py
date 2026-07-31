"""BC warm-up + optional SAC fine-tuning on expert / expert+failure mix (SB3).

One entrypoint for all three modes:

* **BC only** — ``--total-timesteps 0`` (default eval on held-out expert scenarios)
* **BC + SAC unstaged** — ``--total-timesteps N`` and ``--actor-freeze-timesteps 0``
* **BC + SAC staged** — ``--total-timesteps N`` and ``--actor-freeze-timesteps K``

BC amount is controlled by ``--bc-scenarios`` and ``--bc-epochs`` (set epochs to 0
to skip BC). SAC samples a mix of non-failure expert stream + failure cases unless
BC-only.

Examples (GPU node):
    # BC only, eval on expert split
    python -m rl.train_bc_sac --save-dir runs/bc_expert --total-timesteps 0 \\
        --bc-scenarios all --eval-expert-scenarios 64

    # BC + unstaged SAC on expert+failure mix
    python -m rl.train_bc_sac --save-dir runs/bc_sac --total-timesteps 500000 \\
        --actor-freeze-timesteps 0

    # BC + staged SAC (freeze actor for Q warm-up)
    python -m rl.train_bc_sac --save-dir runs/bc_sac_staged --total-timesteps 500000 \\
        --actor-freeze-timesteps 25000
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.bc_cli import (
    add_bc_args,
    add_data_args,
    add_env_args,
    add_logging_args,
    add_sac_args,
    postprocess_args,
)
from rl.bc_core import (
    BCTrainPaths,
    evaluate_policy_on_source,
    failure_shard_indices,
    FrozenActorSACClass,
    load_expert_eval_source,
    load_or_build_bc_dataset,
    make_expert_scenario_generator,
    make_waymax_env,
    MixedScenarioSource,
    resolve_expert_path,
    run_bc_sanity_check,
    train_bc_actor,
)
from rl.encoders import add_encoder_args, build_policy_kwargs
from rl.sac_callbacks import (
    WaymaxEpisodeMetricsCallback,
    WaymaxPeriodicEvalCallback,
    make_monitored_env,
)
from rl.scenario_source import ScenarioSource


def _run_mode_label(total_timesteps: int, actor_freeze_timesteps: int) -> str:
    if total_timesteps <= 0:
        return "bc_only"
    if actor_freeze_timesteps > 0:
        return "bc_sac_staged"
    return "bc_sac_unstaged"


def _adjust_staged_sac_args(args, log_prefix: str) -> None:
    """Ensure critic warm-up uses BC rollouts from env step 0."""
    if args.total_timesteps <= 0 or int(args.actor_freeze_timesteps) <= 0:
        return
    freeze = int(args.actor_freeze_timesteps)
    if int(args.learning_starts) != 0:
        print(
            f"{log_prefix} staged SAC: setting learning_starts=0 "
            f"(was {args.learning_starts}) so critic trains on BC-policy "
            f"rollouts for all {freeze} actor-frozen env steps"
        )
        args.learning_starts = 0


def main() -> None:
    args = _parse_args()
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mode = _run_mode_label(args.total_timesteps, args.actor_freeze_timesteps)
    log_prefix = f"[train_bc_sac/{mode}]"

    paths = BCTrainPaths(failure_dir=args.failure_dir, womd_dir=args.womd_dir)
    expert_path = resolve_expert_path(
        paths, str(save_dir), total_shards=args.total_shards
    )

    bc_only = args.total_timesteps <= 0
    do_bc = int(args.bc_epochs) > 0 and (
        args.bc_scenarios is None or int(args.bc_scenarios) > 0
    )
    _adjust_staged_sac_args(args, log_prefix)

    if args.smoke and (args.bc_scenarios is None or args.bc_scenarios > 32):
        args.bc_scenarios = 32

    obs, act = None, None
    bc_scenarios: list[Any] = []
    bc_scenarios_collected = 0
    if do_bc:
        use_all_expert = args.bc_scenarios is None
        exclude_shards = failure_shard_indices(
            paths.failure_dir, total_shards=args.total_shards
        )
        cache_scenarios = bool(args.eval_on_bc_scenarios) and not use_all_expert
        if args.eval_on_bc_scenarios and use_all_expert:
            print(
                f"{log_prefix} skipping BC scenario cache / sanity on full expert split "
                f"(pass --bc-scenarios N with --eval-on-bc-scenarios for sanity)"
            )
        if not args.bc_no_cache and not args.bc_rebuild_cache:
            print(f"{log_prefix} BC cache dir: {args.bc_cache_dir}")
        elif args.bc_rebuild_cache:
            print(f"{log_prefix} rebuilding BC cache under {args.bc_cache_dir}")
        bc_gen = make_expert_scenario_generator(
            expert_path,
            max_num_objects=args.max_num_objects,
            batch_size=1,
            seed=args.bc_seed,
            repeat=1 if use_all_expert else None,
        )
        if use_all_expert:
            print(f"{log_prefix} BC data: all expert scenarios (non-failure shard pool)")
        else:
            print(f"{log_prefix} BC data: {args.bc_scenarios} expert scenarios")
        bc_result = load_or_build_bc_dataset(
            expert_path=expert_path,
            exclude_shards=exclude_shards,
            total_shards=args.total_shards,
            scenario_iter=bc_gen,
            args=args,
            cache_scenarios=cache_scenarios,
            cache_dir=args.bc_cache_dir,
            use_cache=not args.bc_no_cache,
            rebuild_cache=args.bc_rebuild_cache,
        )
        if cache_scenarios:
            obs, act, bc_scenarios_collected, bc_scenarios = bc_result
        else:
            obs, act, bc_scenarios_collected = bc_result
    else:
        print(f"{log_prefix} skipping BC (bc_epochs={args.bc_epochs}, bc_scenarios={args.bc_scenarios})")

    failure_source = None
    mixed_source = None
    expert_env_source = None
    if bc_only:
        expert_env_source = load_expert_eval_source(
            expert_path,
            max_num_objects=args.max_num_objects,
            num_scenarios=1,
            seed=args.bc_seed,
        )
    else:
        failure_source = ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            limit=args.limit_failures,
        )
        expert_sac_gen = make_expert_scenario_generator(
            expert_path,
            max_num_objects=args.max_num_objects,
            batch_size=1,
            seed=args.seed + 1,
        )
        mixed_source = MixedScenarioSource(
            expert_sac_gen,
            failure_source,
            expert_prob=args.expert_mix_prob,
            seed=args.seed,
        )

    if args.smoke:
        args.learning_starts = min(int(args.learning_starts), 32)
        args.buffer_size = min(int(args.buffer_size), 10_000)
        if not bc_only:
            args.total_timesteps = 128
            args.actor_freeze_timesteps = min(int(args.actor_freeze_timesteps), 64)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if bc_only:
        vec_env = DummyVecEnv([
            make_monitored_env(
                lambda: make_waymax_env(
                    expert_env_source, args, seed=args.seed, sequential=False
                )
            )
        ])
    else:
        env_fns = [
            make_monitored_env(
                lambda i=i: make_waymax_env(
                    mixed_source, args, seed=args.seed + i, sequential=False, mixed=True
                )
            )
            for i in range(args.n_envs)
        ]
        vec_env = DummyVecEnv(env_fns)

    eval_env = None
    n_eval_failures = 0
    if not bc_only and failure_source is not None:
        n_eval_failures = (
            len(failure_source) if args.eval_episodes is None else int(args.eval_episodes)
        )
        if args.eval_freq > 0 and not args.smoke:
            eval_env = make_waymax_env(
                failure_source, args, seed=args.seed + 10_000, sequential=True
            )

    ent_coef = args.ent_coef

    sac_kwargs = {
        # BC (bc_core.bc_actor_loss) calls actor.extract_features(), so it
        # trains through whichever feature extractor is configured here.
        "policy_kwargs": build_policy_kwargs(args),
        "learning_rate": args.lr,
        "buffer_size": args.buffer_size if not bc_only else min(10_000, obs.shape[0] if obs is not None else 10_000),
        "learning_starts": args.learning_starts if not bc_only else 32,
        "batch_size": args.batch_size if not bc_only else args.bc_batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "ent_coef": ent_coef,
        "verbose": 1,
        "device": device,
        "seed": args.seed,
        "tensorboard_log": str(save_dir),
    }

    use_frozen_actor = not bc_only and int(args.actor_freeze_timesteps) > 0
    if use_frozen_actor:
        model = FrozenActorSACClass(
            "MlpPolicy",
            vec_env,
            actor_freeze_timesteps=int(args.actor_freeze_timesteps),
            **sac_kwargs,
        )
    else:
        model = SAC("MlpPolicy", vec_env, **sac_kwargs)

    use_wandb = (not args.no_wandb) and (not args.smoke)
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            sync_tensorboard=True,
            save_code=False,
        )
        wandb.define_metric("bc/epoch")
        wandb.define_metric("bc/*", step_metric="bc/epoch")

    if do_bc and obs is not None and act is not None:
        print(f"{log_prefix} BC warm-up ({obs.shape[0]} transitions, {args.bc_epochs} epochs)")
        train_bc_actor(
            model,
            obs,
            act,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            log_prefix="bc",
            wandb_log=use_wandb,
        )

    sanity_metrics: dict[str, float] = {}
    if bc_scenarios and args.eval_on_bc_scenarios:
        sanity_metrics = run_bc_sanity_check(
            model, bc_scenarios, args, expert_path=expert_path
        )
        with open(save_dir / "bc_sanity_summary.json", "w", encoding="utf-8") as f:
            json.dump(sanity_metrics, f, indent=2)
        if run is not None and sanity_metrics:
            import wandb

            wandb.log(sanity_metrics)

    if bc_only:
        eval_n = int(args.eval_expert_scenarios)
        eval_source = load_expert_eval_source(
            expert_path,
            max_num_objects=args.max_num_objects,
            num_scenarios=eval_n,
            seed=args.bc_seed + 10_000,
        )
        print(f"{log_prefix} evaluating on {eval_n} expert scenarios")
        metrics = evaluate_policy_on_source(model, eval_source, args, n_episodes=eval_n)
        out_path = save_dir / "bc_waymax"
        model.save(out_path.as_posix())
        summary = {
            "mode": mode,
            "expert_path": expert_path,
            "bc_scenarios": args.bc_scenarios if args.bc_scenarios is not None else "all",
            "bc_scenarios_collected": bc_scenarios_collected,
            "bc_cache_dir": args.bc_cache_dir if not args.bc_no_cache else None,
            "bc_epochs": args.bc_epochs,
            "bc_transitions": int(obs.shape[0]) if obs is not None else 0,
            "eval_expert_scenarios": eval_n,
            **sanity_metrics,
            **metrics,
        }
        with open(save_dir / "eval_expert_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n==== Expert eval ====")
        print(f"CLEAN success     : {metrics['eval/clean_success_rate']:.3f}")
        print(f"goal reached      : {metrics['eval/goal_reached_rate']:.3f}")
        print(f"collision rate    : {metrics['eval/collision_rate']:.3f}")
        print(f"offroad rate      : {metrics['eval/offroad_rate']:.3f}")
        if run is not None:
            import wandb

            wandb.log(metrics)
        print(f"Saved model to {out_path}.zip")
        if run is not None:
            run.finish()
    else:
        callbacks = [WaymaxEpisodeMetricsCallback(window=args.train_metrics_window)]
        if eval_env is not None:
            callbacks.append(
                WaymaxPeriodicEvalCallback(
                    eval_env,
                    eval_freq=args.eval_freq,
                    n_episodes=n_eval_failures,
                    verbose=1,
                )
            )

        use_wandb = run is not None
        if use_wandb:
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(WandbCallback(verbose=1))

        callback = CallbackList(callbacks) if callbacks else None
        if use_frozen_actor:
            print(
                f"{log_prefix} SAC fine-tune: actor frozen for "
                f"{args.actor_freeze_timesteps} env steps (critic trains from step 0), "
                f"then full SAC ({args.total_timesteps} total steps)"
            )
        else:
            print(f"{log_prefix} SAC fine-tune on expert+failure mix ({args.total_timesteps} steps)")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=False,
        )

        out_path = save_dir / mode
        model.save(out_path.as_posix())
        meta = {
            "mode": mode,
            "expert_path": expert_path,
            "bc_transitions": int(obs.shape[0]) if obs is not None else 0,
            "bc_epochs": args.bc_epochs,
            "failure_scenarios": len(failure_source),
            "expert_mix_prob": args.expert_mix_prob,
            "actor_freeze_timesteps": args.actor_freeze_timesteps,
            "total_timesteps": args.total_timesteps,
        }
        with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"{log_prefix} Saved model to {out_path}.zip")

        if run is not None:
            run.finish()
    vec_env.close()


def _parse_args():
    p = argparse.ArgumentParser(
        description="BC warm-up + optional SAC (BC-only, unstaged, or staged via flags)."
    )
    add_data_args(p)
    add_env_args(p)
    add_bc_args(p)
    add_sac_args(p)
    add_logging_args(p)
    add_encoder_args(p)
    p.add_argument(
        "--eval-expert-scenarios",
        type=int,
        default=64,
        help="With --total-timesteps 0, eval on this many held-out expert scenarios after BC.",
    )
    p.add_argument(
        "--eval-on-bc-scenarios",
        action="store_true",
        help="After BC, eval policy + expert replay on the exact BC training scenarios "
        "(log-replay agents, matching BC collection). Writes bc_sanity_summary.json.",
    )
    p.add_argument(
        "--bc-sanity-reactive-agents",
        action="store_true",
        help="Also run BC sanity with IDM reactive agents (train/eval distribution mismatch).",
    )
    return postprocess_args(p.parse_args())


if __name__ == "__main__":
    main()
