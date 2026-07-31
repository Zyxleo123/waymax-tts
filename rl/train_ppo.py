"""Train a PPO policy (Stable-Baselines3) on Waymax scenarios.

Examples
--------
Smoke test (loads a few scenarios, runs a short PPO update, exits):

    python -m rl.train_ppo --tfrecord /path/to/training_tfexample.tfrecord-00000-of-01000 \
        --indices 0,1,2,3 --smoke

Train on failure cases produced by goal-reaching / reward-search runs:

    python -m rl.train_ppo --failure-dir /path/to/failure_samples \
        --total-timesteps 1000000 --save-dir runs/ppo_failures

Train on an explicit set of scenarios from one TFRecord:

    python -m rl.train_ppo --tfrecord /path/to/shard --num-scenarios 64 \
        --total-timesteps 500000
"""

from __future__ import annotations

# JAX must not preallocate the whole GPU; leave room and avoid surprising OOMs
# when sharing the box. Set before any JAX import happens transitively.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.encoders import add_encoder_args, build_policy_kwargs
from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv


def _parse_indices(indices: str | None, num_scenarios: int | None) -> list[int] | None:
    if indices:
        return [int(x) for x in indices.split(",") if x.strip() != ""]
    if num_scenarios:
        return list(range(int(num_scenarios)))
    return None


def build_scenario_source(args) -> ScenarioSource:
    if args.failure_dir:
        return ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            failures_only=not args.include_successes,
            limit=args.limit,
        )
    if args.tfrecord:
        indices = _parse_indices(args.indices, args.num_scenarios)
        if not indices:
            raise ValueError("Provide --indices or --num-scenarios with --tfrecord.")
        return ScenarioSource.from_tfrecord(
            args.tfrecord,
            indices,
            max_num_objects=args.max_num_objects,
        )
    raise ValueError("Provide either --failure-dir or --tfrecord.")


def make_env_fn(source: ScenarioSource, args, *, seed: int, sequential: bool):
    from stable_baselines3.common.monitor import Monitor

    reward_cfg = RewardConfig(
        progress=args.r_progress,
        action_penalty=args.r_action_penalty,
        collision=args.r_collision,
        offroad=args.r_offroad,
        goal_bonus=args.r_goal_bonus,
        goal_threshold_m=args.goal_threshold_m,
        terminate_on_offroad=args.terminate_on_offroad,
        route_reward=args.route_reward,
        lateral_penalty=args.r_lateral_penalty,
    )

    def _thunk():
        env = WaymaxGymEnv(
            source,
            reward_config=reward_cfg,
            max_episode_steps=args.max_episode_steps,
            sequential=sequential,
            seed=seed,
            action_space_type=args.action_space,
            delta_max_dx=args.delta_max_dx,
            delta_max_dy=args.delta_max_dy,
            delta_max_dyaw=args.delta_max_dyaw,
        )
        return Monitor(env)

    return _thunk


def main():
    args = _parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    source = build_scenario_source(args)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_ppo] {len(source)} scenarios | device={device}")

    env_fns = [
        make_env_fn(source, args, seed=args.seed + i, sequential=False)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    n_steps = args.n_steps
    if args.smoke:
        n_steps = 64

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=build_policy_kwargs(args),
        n_steps=n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        device=device,
        seed=args.seed,
        tensorboard_log=args.save_dir,
    )

    # Optional Weights & Biases logging (syncs SB3's TensorBoard metrics).
    run = None
    callback = None
    use_wandb = (not args.no_wandb) and (not args.smoke)
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            sync_tensorboard=True,
            save_code=False,
        )
        callback = WandbCallback(verbose=1)

    total_timesteps = 512 if args.smoke else args.total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "ppo_waymax"
    model.save(out_path.as_posix())
    print(f"[train_ppo] Saved model to {out_path}.zip")

    if run is not None:
        run.finish()


def _parse_args():
    p = argparse.ArgumentParser(description="PPO on Waymax scenarios.")
    # Data source.
    p.add_argument("--failure-dir", type=str, default=None,
                   help="Directory of per-scenario result JSONs (failure cases).")
    p.add_argument("--tfrecord", type=str, default=None,
                   help="Single TFRecord file to load scenarios from.")
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated scenario indices for --tfrecord.")
    p.add_argument("--num-scenarios", type=int, default=None,
                   help="Use indices [0, N) from --tfrecord.")
    p.add_argument("--include-successes", action="store_true",
                   help="With --failure-dir, also include successful scenarios.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of scenarios loaded from --failure-dir.")
    p.add_argument("--max-num-objects", type=int, default=None,
                   help="Truncate scene to N objects. Default None keeps all (WOMD=128); "
                        "truncation risks dropping the SDC.")

    # Env / reward.
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--action-space", type=str, default="bicycle",
                   choices=["bicycle", "delta"],
                   help="Control mode: 'bicycle' (accel, steering) or "
                        "'delta' (next-position dx, dy, dyaw in ego frame).")
    p.add_argument("--delta-max-dx", type=float, default=6.0,
                   help="Max per-step dx (m) for --action-space delta.")
    p.add_argument("--delta-max-dy", type=float, default=6.0,
                   help="Max per-step dy (m) for --action-space delta.")
    p.add_argument("--delta-max-dyaw", type=float, default=float(np.pi),
                   help="Max per-step dyaw (rad) for --action-space delta.")
    p.add_argument("--r-progress", type=float, default=1.0)
    p.add_argument("--r-action-penalty", type=float, default=0.01)
    p.add_argument("--r-collision", type=float, default=-10.0)
    p.add_argument("--r-offroad", type=float, default=-5.0)
    p.add_argument("--r-goal-bonus", type=float, default=10.0)
    p.add_argument("--terminate-on-offroad", action="store_true",
                   help="End the episode on offroad (default: penalize but continue). "
                        "Stops the policy from tolerating sustained off-road driving.")
    p.add_argument("--route-reward", action="store_true",
                   help="Reward progress along the expert log path (on-road, "
                        "collision-free) instead of straight-line distance to the "
                        "goal. Discourages beelining off-road / through agents.")
    p.add_argument("--r-lateral-penalty", type=float, default=0.5,
                   help="With --route-reward, penalty per meter of lateral "
                        "deviation from the expert path.")

    # PPO.
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # Policy architecture (feature extractor + head sizes).
    add_encoder_args(p)

    # Logging.
    p.add_argument("--save-dir", type=str, default="runs/ppo_waymax")
    p.add_argument("--wandb-project", type=str, default="ppo-waymax")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging (TensorBoard only).")

    # Misc.
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for a quick end-to-end sanity check.")
    return p.parse_args()


if __name__ == "__main__":
    main()
