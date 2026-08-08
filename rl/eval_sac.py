"""Run a trained SAC policy over a set of Waymax scenarios, report metrics, and
save per-scenario rollouts (trajectory + success/collision/offroad) to disk.

For each scenario it writes a JSON with the rolled-out ego trajectory (and the
expert log trajectory + goal for comparison), the actions taken, and the episode
outcome; plus a top-level ``summary.json``. Output goes to ``--out-dir`` (default
``<model_dir>/eval_rollouts``).

Example
-------
    python -m rl.eval_sac --model runs/sac_failures/sac_waymax.zip \
        --failure-dir /path/to/failure_samples --action-space bicycle
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.run_config import RUN_CONFIG_NAME, apply_run_config_defaults, find_run_config
from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv


def resolve_sb3_checkpoint(path: str) -> str:
    """Return a path suitable for ``SAC.load()`` (SB3 appends ``.zip`` itself)."""
    p = Path(path)
    if p.suffix == ".zip":
        return str(p.with_suffix(""))
    return path


def build_source(args) -> ScenarioSource:
    # include_sdc_paths must match training: without it the observation's
    # path_target block degrades to zeros, so the policy would be evaluated on a
    # different observation than it trained on -- silently, since the width is
    # unchanged.
    if args.failure_dir:
        return ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            failures_only=not args.include_successes,
            limit=args.limit,
            include_sdc_paths=True,
        )
    if args.tfrecord:
        indices = [int(x) for x in args.indices.split(",")] if args.indices else list(
            range(int(args.num_scenarios or 1))
        )
        return ScenarioSource.from_tfrecord(
            args.tfrecord,
            indices,
            max_num_objects=args.max_num_objects,
            include_sdc_paths=True,
        )
    raise ValueError("Provide --failure-dir or --tfrecord.")


def main():
    args = _parse_args()
    from stable_baselines3 import SAC

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
        off_route_threshold_m=args.off_route_threshold_m,
        off_route_penalty=args.r_off_route,
        progression_indicator=args.progression_indicator,
    )
    env = WaymaxGymEnv(
        source,
        reward_config=reward_cfg,
        max_episode_steps=args.max_episode_steps,
        sequential=True,
        action_space_type=args.action_space,
        delta_max_dx=args.delta_max_dx,
        delta_max_dy=args.delta_max_dy,
        delta_max_dyaw=args.delta_max_dyaw,
        reactive_agents=args.reactive_agents,
        idm_desired_vel=args.idm_desired_vel,
    )
    model = SAC.load(resolve_sb3_checkpoint(args.model), device=args.device)

    # Where to dump per-scenario rollouts. Defaults to a folder next to the
    # checkpoint so eval always leaves something on disk to inspect.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.model).resolve().parent / "eval_rollouts"
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
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1).tolist())
            ep_return += reward
            ep_collision = ep_collision or info["collision"]
            ep_offroad = ep_offroad or info["offroad"]
            ep_reached = ep_reached or info["reached"]
            last_goal_dist = float(info["goal_dist"])
            ep_len += 1
            done = terminated or truncated

        # "Clean" success is the metric that matters: reached the goal without
        # crashing or driving off-road. Raw goal-reaching can be gamed (e.g. a
        # DeltaLocal policy teleporting to the goal through other agents).
        ep_clean = bool(ep_reached and not ep_collision and not ep_offroad)
        returns.append(ep_return)
        n_reached += int(ep_reached)
        n_collision += int(ep_collision)
        n_offroad += int(ep_offroad)
        n_clean += int(ep_clean)

        # Per-scenario record: metrics + the rolled-out trajectory + actions.
        traj = env.ego_trajectories()
        record = {
            "tfrecord": spec.get("tfrecord"),
            "scenario_idx": spec.get("scenario_idx"),
            "action_space": args.action_space,
            "success": bool(ep_reached),
            "reached": bool(ep_reached),
            "clean_success": ep_clean,
            "collision": bool(ep_collision),
            "offroad": bool(ep_offroad),
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
                "offroad", "return", "ep_len", "final_goal_dist")}
        )
        print(
            f"[ep {ep:04d}] return={ep_return:8.2f} reached={ep_reached} "
            f"clean={ep_clean} collision={ep_collision} offroad={ep_offroad}"
        )

    summary = {
        "model": str(Path(args.model).resolve()),
        "action_space": args.action_space,
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

    print("\n==== Evaluation summary ====")
    print(f"episodes          : {n_episodes}")
    print(f"goal reached      : {n_reached}/{n_episodes} = {n_reached / n_episodes:.3f}")
    print(f"CLEAN success     : {n_clean}/{n_episodes} = {n_clean / n_episodes:.3f}"
          f"   (reached & no collision & no offroad)")
    print(f"collision rate    : {n_collision}/{n_episodes} = {n_collision / n_episodes:.3f}")
    print(f"offroad rate      : {n_offroad}/{n_episodes} = {n_offroad / n_episodes:.3f}")
    print(f"mean return       : {np.mean(returns):.3f}")
    print(f"\nSaved {n_episodes} rollouts + summary.json to {out_dir}")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a SAC policy on Waymax scenarios.")
    p.add_argument("--model", type=str, required=True, help="Path to SAC .zip checkpoint.")
    p.add_argument("--failure-dir", type=str, default=None)
    p.add_argument("--tfrecord", type=str, default=None)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--num-scenarios", type=int, default=None)
    p.add_argument("--include-successes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-num-objects", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--action-space", type=str, default="bicycle",
                   choices=["bicycle", "delta"],
                   help="Must match the action space the policy was trained with.")
    p.add_argument("--delta-max-dx", type=float, default=6.0)
    p.add_argument("--delta-max-dy", type=float, default=6.0)
    p.add_argument("--delta-max-dyaw", type=float, default=float(np.pi))
    p.add_argument("--r-progress", type=float, default=1.0)
    p.add_argument("--r-action-penalty", type=float, default=0.01)
    p.add_argument("--r-collision", type=float, default=-10.0)
    p.add_argument("--r-offroad", type=float, default=-5.0)
    p.add_argument("--r-goal-bonus", type=float, default=10.0)
    p.add_argument("--terminate-on-offroad", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--terminate-on-collision", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--route-reward", action="store_true")
    p.add_argument("--r-lateral-penalty", type=float, default=0.5)
    # Keep these in sync with train_sac.py: the launcher passes one shared
    # reward string to BOTH stages, so a flag missing here aborts the eval after
    # training has already finished.
    p.add_argument("--off-route-threshold-m", type=float, default=None,
                   help="Use V-Max's bounded off-route indicator instead of the "
                        "per-meter lateral penalty.")
    p.add_argument("--r-off-route", type=float, default=-0.2,
                   help="Per-step off-route penalty (V-Max reward_config.off_route).")
    p.add_argument("--progression-indicator", action="store_true",
                   help="Make the route progression term V-Max's bounded indicator: "
                        "award --r-progress on any step where route arclength "
                        "increased, instead of --r-progress per meter advanced.")
    p.add_argument("--reactive-agents", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Use IDM sim agents for non-ego objects instead of log replay. "
                        "Inherited from the checkpoint's run_config.json when present "
                        "-- BC-SAC trains with IDM agents, so evaluating against log "
                        "replay reports metrics from a different environment.")
    p.add_argument("--idm-desired-vel", type=float, default=30.0)
    p.add_argument("--run-config", type=str, default=None,
                   help="run_config.json describing the environment the checkpoint "
                        "was trained in. Defaults to the one saved beside the "
                        "checkpoint; --no-run-config evaluates on this script's own "
                        "defaults instead.")
    p.add_argument("--no-run-config", dest="use_run_config", action="store_false",
                   help="Ignore the checkpoint's saved environment config.")
    p.set_defaults(use_run_config=True)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to save per-scenario rollouts + summary.json. "
                        "Defaults to '<model_dir>/eval_rollouts'.")
    p.add_argument("--device", type=str, default="cpu")

    args = p.parse_args()
    if not args.use_run_config:
        return args

    cfg_path = Path(args.run_config) if args.run_config else find_run_config(args.model)
    if cfg_path is None or not Path(cfg_path).is_file():
        # Silence here is how training and eval drifted apart in the first place.
        print(
            f"[eval_sac] WARNING: no {RUN_CONFIG_NAME} found beside {args.model}; "
            "falling back to this script's defaults. Metrics are only comparable "
            "if the checkpoint was trained with "
            f"reactive_agents={args.reactive_agents}, action_space={args.action_space}, "
            f"terminate_on_collision={args.terminate_on_collision}."
        )
        return args

    applied = apply_run_config_defaults(p, cfg_path)
    # Re-parse against the seeded defaults so explicitly passed flags still win.
    args = p.parse_args()
    args.run_config_path = str(cfg_path)
    print(f"[eval_sac] environment inherited from {cfg_path}: {applied}")
    return args


if __name__ == "__main__":
    main()
