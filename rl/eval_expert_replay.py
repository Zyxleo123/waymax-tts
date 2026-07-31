"""Run expert (log-replay or IDM-reactive) golden-trajectory rollouts over a set of
Waymax scenarios, report metrics, and save per-scenario rollouts to disk in the
same schema ``rl.eval_sac`` / ``rl.eval_ppo`` use, so ``rl.visualize_rollouts``
can render videos of them directly.

This is the "expert replay" ceiling of ``rl.diagnose_failure_ceiling``, but with
per-scenario detail (which scenario failed, how, and the actual simulated
trajectory) instead of just aggregate rates.

Example
-------
    python -m rl.eval_expert_replay --failure-dir /path/to/failure_samples \
        --out-dir runs/expert_ceiling/log_replay

    python -m rl.eval_expert_replay --failure-dir /path/to/failure_samples \
        --reactive-agents --out-dir runs/expert_ceiling/idm

    python -m rl.visualize_rollouts --rollout-dir runs/expert_ceiling/log_replay \
        --mode video --filter failure
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# Avoid ScenarioSource silently dropping the first-loaded shard on a transient
# CUDA-init flake on GPU-less nodes (see rl/diagnose_failure_ceiling.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.bc_core import expert_action_sdc
from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv


def build_source(args) -> ScenarioSource:
    if args.failure_dir:
        return ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            failures_only=not args.include_successes,
            limit=args.limit,
        )
    if args.tfrecord:
        indices = [int(x) for x in args.indices.split(",")] if args.indices else list(
            range(int(args.num_scenarios or 1))
        )
        return ScenarioSource.from_tfrecord(
            args.tfrecord, indices, max_num_objects=args.max_num_objects
        )
    raise ValueError("Provide --failure-dir or --tfrecord.")


def main() -> None:
    args = _parse_args()
    source = build_source(args)
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
    )
    env = WaymaxGymEnv(
        source,
        reward_config=reward_cfg,
        max_episode_steps=args.max_episode_steps,
        sequential=True,
        action_space_type="bicycle",
        reactive_agents=args.reactive_agents,
        idm_desired_vel=args.idm_desired_vel,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = len(source) if args.num_episodes is None else int(args.num_episodes)
    n_reached = n_collision = n_offroad = n_clean = 0
    returns = []
    summary_rows = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        spec = info.get("scenario", {}) or {}
        done = False
        ep_return = 0.0
        ep_collision = ep_offroad = ep_reached = False
        actions = []
        last_goal_dist = float(info.get("goal_dist", float("nan")))
        ep_len = 0
        first_collision_step: int | None = None
        first_offroad_step: int | None = None
        while not done:
            action = expert_action_sdc(env._state, env._dynamics)[: env._action_dim]
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1).tolist())
            ep_return += reward
            if info["collision"] and first_collision_step is None:
                first_collision_step = ep_len
            if info["offroad"] and first_offroad_step is None:
                first_offroad_step = ep_len
            ep_collision = ep_collision or info["collision"]
            ep_offroad = ep_offroad or info["offroad"]
            ep_reached = ep_reached or info["reached"]
            last_goal_dist = float(info["goal_dist"])
            ep_len += 1
            done = terminated or truncated

        ep_clean = bool(ep_reached and not ep_collision and not ep_offroad)
        returns.append(ep_return)
        n_reached += int(ep_reached)
        n_collision += int(ep_collision)
        n_offroad += int(ep_offroad)
        n_clean += int(ep_clean)

        traj = env.ego_trajectories()
        record = {
            "tfrecord": spec.get("tfrecord"),
            "scenario_idx": spec.get("scenario_idx"),
            "reactive_agents": bool(args.reactive_agents),
            "success": bool(ep_reached),
            "reached": bool(ep_reached),
            "clean_success": ep_clean,
            "collision": bool(ep_collision),
            "offroad": bool(ep_offroad),
            "first_collision_step": first_collision_step,
            "first_offroad_step": first_offroad_step,
            "return": float(ep_return),
            "ep_len": int(ep_len),
            "final_goal_dist": last_goal_dist,
            "actions": actions,
            **traj,
        }
        idx = spec.get("scenario_idx")
        shard = Path(spec["tfrecord"]).name if spec.get("tfrecord") else f"ep{ep:04d}"
        fname = f"{shard}.scenario_{idx:05d}.json" if isinstance(idx, int) else f"{shard}.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(record, f)

        summary_rows.append(
            {k: record[k] for k in (
                "tfrecord", "scenario_idx", "success", "clean_success", "collision",
                "offroad", "first_collision_step", "first_offroad_step",
                "return", "ep_len", "final_goal_dist")}
        )
        print(
            f"[ep {ep:04d}] {fname}: reached={ep_reached} clean={ep_clean} "
            f"collision={ep_collision}@{first_collision_step} "
            f"offroad={ep_offroad}@{first_offroad_step}"
        )

    summary = {
        "reactive_agents": bool(args.reactive_agents),
        "episodes": n_episodes,
        "goal_reached": n_reached,
        "clean_success": n_clean,
        "collisions": n_collision,
        "offroads": n_offroad,
        "goal_reached_rate": n_reached / n_episodes,
        "clean_success_rate": n_clean / n_episodes,
        "collision_rate": n_collision / n_episodes,
        "offroad_rate": n_offroad / n_episodes,
        "mean_return": float(np.mean(returns)),
        "scenarios": summary_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Expert replay ceiling ====")
    print(f"episodes          : {n_episodes}")
    print(f"goal reached      : {n_reached}/{n_episodes} = {n_reached / n_episodes:.3f}")
    print(f"CLEAN success     : {n_clean}/{n_episodes} = {n_clean / n_episodes:.3f}")
    print(f"collision rate    : {n_collision}/{n_episodes} = {n_collision / n_episodes:.3f}")
    print(f"offroad rate      : {n_offroad}/{n_episodes} = {n_offroad / n_episodes:.3f}")
    print(f"\nSaved {n_episodes} rollouts + summary.json to {out_dir}")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate the expert/golden trajectory on Waymax scenarios.")
    p.add_argument("--failure-dir", type=str, default=None)
    p.add_argument("--tfrecord", type=str, default=None)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--num-scenarios", type=int, default=None)
    p.add_argument("--include-successes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-num-objects", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--r-progress", type=float, default=1.0)
    p.add_argument("--r-action-penalty", type=float, default=0.01)
    p.add_argument("--r-collision", type=float, default=-10.0)
    p.add_argument("--r-offroad", type=float, default=-5.0)
    p.add_argument("--r-goal-bonus", type=float, default=10.0)
    p.add_argument("--terminate-on-offroad", action="store_true")
    p.add_argument("--terminate-on-collision", action="store_true")
    p.add_argument("--route-reward", action="store_true")
    p.add_argument("--r-lateral-penalty", type=float, default=0.5)
    p.add_argument("--reactive-agents", action="store_true",
                   help="Use IDM sim agents for non-ego objects instead of log replay.")
    p.add_argument("--idm-desired-vel", type=float, default=30.0)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--out-dir", type=str, required=True,
                   help="Where to save per-scenario rollouts + summary.json.")
    return p.parse_args()


if __name__ == "__main__":
    main()
