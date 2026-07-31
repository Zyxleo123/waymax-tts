"""Train a SAC policy (Stable-Baselines3) on Waymax scenarios.

Examples
--------
Smoke test (loads a few scenarios, runs a short SAC update, exits):

    python -m rl.train_sac --tfrecord /path/to/training_tfexample.tfrecord-00000-of-01000 \
        --indices 0,1,2,3 --smoke

Train on failure cases produced by goal-reaching / reward-search runs:

    python -m rl.train_sac --failure-dir /path/to/failure_samples \
        --total-timesteps 1000000 --save-dir runs/sac_failures

Train on an explicit set of scenarios from one TFRecord:

    python -m rl.train_sac --tfrecord /path/to/shard --num-scenarios 64 \
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
from rl.sac_callbacks import (
    WaymaxEpisodeMetricsCallback,
    WaymaxPeriodicEvalCallback,
    make_monitored_env,
)
from rl.waymax_env import RewardConfig, WaymaxGymEnv


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest ``sac_<steps>_steps.zip`` in ``ckpt_dir``, by step count."""
    if not ckpt_dir.is_dir():
        return None

    def steps(p: Path) -> int:
        parts = [x for x in p.stem.split("_") if x.isdigit()]
        return int(parts[-1]) if parts else -1

    ckpts = [p for p in ckpt_dir.glob("sac_*_steps.zip") if steps(p) >= 0]
    return max(ckpts, key=steps) if ckpts else None


def _replay_buffer_path(checkpoint: Path) -> Path:
    """Buffer file SB3's ``CheckpointCallback`` writes beside a checkpoint.

    It uses ``{prefix}_{type}{steps}_steps.{ext}``, so the buffer for
    ``sac_400000_steps.zip`` is ``sac_replay_buffer_400000_steps.pkl`` -- the
    marker goes *before* the step count, not after the stem.
    """
    prefix, _, rest = checkpoint.stem.partition("_")
    return checkpoint.with_name(f"{prefix}_replay_buffer_{rest}.pkl")


def _parse_indices(indices: str | None, num_scenarios: int | None) -> list[int] | None:
    if indices:
        return [int(x) for x in indices.split(",") if x.strip() != ""]
    if num_scenarios:
        return list(range(int(num_scenarios)))
    return None


def build_scenario_source(args) -> ScenarioSource:
    # SDC paths are needed for the observation's ``path_target`` block (the route
    # the progression / off-route reward is defined against). WOMD 1.3.1 ships
    # them as ``path_samples`` (45 paths x 800 points).
    if args.failure_dir:
        return ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            failures_only=not args.include_successes,
            limit=args.limit,
            include_sdc_paths=True,
        )
    if args.tfrecord:
        indices = _parse_indices(args.indices, args.num_scenarios)
        if not indices:
            raise ValueError("Provide --indices or --num-scenarios with --tfrecord.")
        return ScenarioSource.from_tfrecord(
            args.tfrecord,
            indices,
            max_num_objects=args.max_num_objects,
            include_sdc_paths=True,
        )
    raise ValueError("Provide either --failure-dir or --tfrecord.")


def make_base_env_fn(source: ScenarioSource, args, *, seed: int, sequential: bool):
    reward_cfg = RewardConfig(
        progress=args.r_progress,
        action_penalty=args.r_action_penalty,
        collision=args.r_collision,
        offroad=args.r_offroad,
        goal_bonus=args.r_goal_bonus,
        goal_threshold_m=args.goal_threshold_m,
        terminate_on_collision=args.terminate_on_collision,
        terminate_on_offroad=args.terminate_on_offroad,
        route_reward=args.route_reward,
        lateral_penalty=args.r_lateral_penalty,
        off_route_threshold_m=args.off_route_threshold_m,
        off_route_penalty=args.r_off_route,
    )

    def _thunk():
        return WaymaxGymEnv(
            source,
            reward_config=reward_cfg,
            max_episode_steps=args.max_episode_steps,
            sequential=sequential,
            seed=seed,
            action_space_type=args.action_space,
            delta_max_dx=args.delta_max_dx,
            delta_max_dy=args.delta_max_dy,
            delta_max_dyaw=args.delta_max_dyaw,
            reactive_agents=args.reactive_agents,
            idm_desired_vel=args.idm_desired_vel,
        )

    return _thunk


def make_env_fn(source: ScenarioSource, args, *, seed: int, sequential: bool):
    return make_monitored_env(make_base_env_fn(source, args, seed=seed, sequential=sequential))


def main():
    args = _parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv

    source = build_scenario_source(args)

    if args.smoke:
        args.learning_starts = min(int(args.learning_starts), 32)
        args.buffer_size = min(int(args.buffer_size), 10_000)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_sac] {len(source)} scenarios | device={device} | encoder={args.encoder}")

    env_fns = [
        make_env_fn(source, args, seed=args.seed + i, sequential=False)
        for i in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    eval_env = None
    n_eval_episodes = len(source) if args.eval_episodes is None else int(args.eval_episodes)
    if args.eval_freq > 0 and not args.smoke:
        eval_env = make_base_env_fn(source, args, seed=args.seed + 10_000, sequential=True)()

    ent_coef: str | float = args.ent_coef
    if ent_coef != "auto":
        ent_coef = float(ent_coef)

    # Resume from the newest checkpoint if one exists. V-Max's reference run is
    # 25M steps, far past any single walltime, so hitting the limit has to mean
    # "resubmit", not "start over".
    ckpt_dir = Path(args.save_dir) / "checkpoints"
    resume_from = _latest_checkpoint(ckpt_dir) if args.resume else None

    if resume_from is not None:
        print(f"[train_sac] resuming from {resume_from}")
        model = SAC.load(resume_from.as_posix(), env=vec_env, device=device)
        buf = _replay_buffer_path(resume_from)
        if buf.is_file():
            model.load_replay_buffer(buf.as_posix())
            print(f"[train_sac] restored replay buffer ({model.replay_buffer.size()} transitions)")
        else:
            # Without the buffer SAC would train off an empty replay memory, so
            # re-serve the random-exploration window before updating again.
            model.learning_starts = model.num_timesteps + int(args.learning_starts)
            print("[train_sac] no replay buffer found; re-running exploration window")
        model.set_logger(configure(args.save_dir, ["stdout", "tensorboard"]))
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            policy_kwargs=build_policy_kwargs(args),
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=ent_coef,
            verbose=1,
            device=device,
            seed=args.seed,
            tensorboard_log=args.save_dir,
        )

    # Optional Weights & Biases logging (syncs SB3's TensorBoard metrics).
    run = None
    callbacks = [
        WaymaxEpisodeMetricsCallback(window=args.train_metrics_window),
    ]
    if eval_env is not None:
        callbacks.append(
            WaymaxPeriodicEvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_episodes=n_eval_episodes,
                verbose=1,
            )
        )
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
        callbacks.append(WandbCallback(verbose=1))

    if args.checkpoint_freq > 0 and not args.smoke:
        # save_freq is per-env, so divide to get the intended env-step cadence.
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_freq) // max(1, args.n_envs)),
                save_path=str(ckpt_dir),
                name_prefix="sac",
                save_replay_buffer=args.save_replay_buffer,
                verbose=1,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    total_timesteps = 128 if args.smoke else args.total_timesteps
    # On resume, learn() counts from the restored num_timesteps, so ask only for
    # what is left and keep the existing step counter.
    remaining = total_timesteps
    if resume_from is not None:
        remaining = max(0, total_timesteps - model.num_timesteps)
        print(f"[train_sac] {model.num_timesteps}/{total_timesteps} done; {remaining} to go")
    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=resume_from is None,
        )
    else:
        print("[train_sac] target already reached; nothing to train")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "sac_waymax"
    model.save(out_path.as_posix())
    print(f"[train_sac] Saved model to {out_path}.zip")

    if run is not None:
        run.finish()


def _parse_args():
    p = argparse.ArgumentParser(description="SAC on Waymax scenarios.")
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
                        "deviation from the expert path. Unbounded -- prefer "
                        "--off-route-threshold-m for the V-Max form.")
    p.add_argument("--off-route-threshold-m", type=float, default=None,
                   help="Switch the route term to V-Max's bounded indicator: charge "
                        "--r-off-route once per step when lateral deviation exceeds "
                        "this many meters, instead of the per-meter penalty.")
    p.add_argument("--r-off-route", type=float, default=-0.2,
                   help="Per-step off-route penalty (V-Max reward_config.off_route).")
    p.add_argument("--terminate-on-collision", action="store_true",
                   help="End the episode on collision (default: penalize but continue).")
    p.add_argument("--reactive-agents", action="store_true",
                   help="Use IDM sim agents for non-ego objects instead of log replay.")
    p.add_argument("--idm-desired-vel", type=float, default=30.0,
                   help="IDM free-road desired speed (m/s) for --reactive-agents.")

    # SAC.
    p.add_argument("--n-envs", type=int, default=16,
                   help="Parallel envs for data collection.")
    p.add_argument("--buffer-size", type=int, default=1_000_000,
                   help="Replay buffer capacity (transitions).")
    p.add_argument("--learning-starts", type=int, default=50_000,
                   help="Collect this many steps before the first gradient update.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005,
                   help="Soft update coefficient for target networks.")
    p.add_argument("--train-freq", type=int, default=1,
                   help="Update the model every N env steps (per env when vectorized).")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="Gradient steps per training call.")
    p.add_argument("--ent-coef", type=str, default="0.2",
                   help="Entropy coefficient ('auto' for automatic tuning, or a float).")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # Policy architecture (feature extractor + head sizes).
    add_encoder_args(p)

    # Logging.
    p.add_argument("--save-dir", type=str, default="runs/sac_waymax")
    p.add_argument("--wandb-project", type=str, default="sac-waymax")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging (TensorBoard only).")
    p.add_argument("--eval-freq", type=int, default=10000,
                   help="Run sequential eval every N env steps (0=disable). "
                        "Logs eval/clean_success_rate, collision, offroad, lateral.")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="Episodes per periodic eval (default: all loaded scenarios).")
    p.add_argument("--train-metrics-window", type=int, default=100,
                   help="Rolling window for train/clean_success_rate etc.")

    # Checkpoint / resume. V-Max's reference is 25M steps -- far beyond one
    # walltime -- so hitting the limit must mean resubmit, not restart.
    p.add_argument("--checkpoint-freq", type=int, default=100_000,
                   help="Save a checkpoint every N env steps (0=disable).")
    p.add_argument("--save-replay-buffer", action="store_true", default=True,
                   help="Also checkpoint the replay buffer so a resumed run keeps "
                        "its off-policy data (large: ~obs_dim x buffer_size).")
    p.add_argument("--no-save-replay-buffer", dest="save_replay_buffer",
                   action="store_false",
                   help="Skip the replay buffer; a resumed run then re-runs the "
                        "exploration window before training.")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the newest checkpoint under <save-dir>/checkpoints.")

    # Misc.
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for a quick end-to-end sanity check.")
    args = p.parse_args()

    if args.ent_coef != "auto":
        args.ent_coef = float(args.ent_coef)
    return args


if __name__ == "__main__":
    main()
