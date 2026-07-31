"""Compare BC vs expert-replay rollouts step-by-step (compounding drift check).

Usage (GPU node recommended):
    python -m rl.diagnose_bc_drift \\
        --model /zfsauton/scratch/yixiz/waymax_rs/runs/bc_expert/bc_waymax.zip \\
        --n-scenarios 64 --bc-seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.bc_cli import add_data_args, add_env_args
from rl.bc_core import (
    BCTrainPaths,
    expert_action_sdc,
    make_expert_scenario_generator,
    make_waymax_env,
    resolve_expert_path,
)
from rl.scenario_source import CachedScenarioSource, _ensure_unbatched_scenario, _to_host_pytree


def _rollout(env, *, use_expert: bool, model=None, deterministic: bool = True):
    """Return per-step arrays for one episode."""
    obs, _ = env.reset()
    pos_xy: list[tuple[float, float]] = []
    act_bc: list[np.ndarray] = []
    act_expert: list[np.ndarray] = []
    act_err: list[float] = []
    lateral: list[float] = []
    collision = offroad = reached = False

    done = False
    while not done:
        expert_a = expert_action_sdc(env._state, env._dynamics)[: env._action_dim]
        if use_expert:
            action = expert_a
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32).reshape(-1)[: env._action_dim]
            act_err.append(float(np.linalg.norm(action - expert_a)))

        is_sdc = np.asarray(env._state.object_metadata.is_sdc).astype(bool)
        ego = int(np.argmax(is_sdc))
        sim = env._state.sim_trajectory
        t = int(np.asarray(env._state.timestep))
        pos_xy.append(
            (float(np.asarray(sim.x)[ego, t]), float(np.asarray(sim.y)[ego, t]))
        )

        obs, _, terminated, truncated, info = env.step(action)
        lateral.append(float(info.get("lateral_deviation_m", 0.0)))
        collision |= bool(info.get("collision", False))
        offroad |= bool(info.get("offroad", False))
        reached |= bool(info.get("reached", False))
        done = terminated or truncated
        act_bc.append(action.copy())
        act_expert.append(expert_a.copy())

    clean = reached and not collision and not offroad
    return {
        "pos_xy": np.asarray(pos_xy, dtype=np.float64),
        "act_err": np.asarray(act_err, dtype=np.float64),
        "lateral": np.asarray(lateral, dtype=np.float64),
        "collision": collision,
        "offroad": offroad,
        "reached": reached,
        "clean": clean,
        "length": len(pos_xy),
    }


def _align_drift(expert_pos: np.ndarray, bc_pos: np.ndarray) -> np.ndarray:
    n = min(len(expert_pos), len(bc_pos))
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(bc_pos[:n] - expert_pos[:n], axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description="BC vs expert replay drift diagnostics.")
    add_data_args(p)
    add_env_args(p)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--n-scenarios", type=int, default=64)
    p.add_argument("--bc-seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="/tmp/bc_drift_diag")
    args = p.parse_args()

    from stable_baselines3 import SAC

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = BCTrainPaths(failure_dir=args.failure_dir, womd_dir=args.womd_dir)
    expert_path = resolve_expert_path(paths, str(save_dir), total_shards=args.total_shards)

    gen = make_expert_scenario_generator(
        expert_path,
        max_num_objects=args.max_num_objects,
        batch_size=1,
        seed=args.bc_seed,
        repeat=1,
        distributed=False,
    )
    scenarios = []
    for _ in range(int(args.n_scenarios)):
        scenarios.append(_ensure_unbatched_scenario(_to_host_pytree(next(gen))))
    source = CachedScenarioSource(scenarios, label=expert_path)

    env = make_waymax_env(
        source,
        args,
        seed=0,
        sequential=True,
        reactive_agents=bool(args.reactive_agents),
    )
    model = SAC.load(args.model, env=env, device="cpu")

    max_len = int(args.max_episode_steps)
    drifts_by_step = [[] for _ in range(max_len)]
    act_err_by_step = [[] for _ in range(max_len)]
    lat_bc_by_step = [[] for _ in range(max_len)]
    lat_expert_by_step = [[] for _ in range(max_len)]

    per_episode = []
    for ep in range(len(source)):
        single = CachedScenarioSource([scenarios[ep]], label=f"ep_{ep}")
        env._source = single
        env._seq_cursor = 0
        expert = _rollout(env, use_expert=True)
        env._seq_cursor = 0
        bc = _rollout(env, use_expert=False, model=model)
        drift = _align_drift(expert["pos_xy"], bc["pos_xy"])
        per_episode.append(
            {
                "episode": ep,
                "expert_clean": expert["clean"],
                "bc_clean": bc["clean"],
                "bc_offroad": bc["offroad"],
                "bc_collision": bc["collision"],
                "final_drift_m": float(drift[-1]) if len(drift) else 0.0,
                "max_drift_m": float(np.max(drift)) if len(drift) else 0.0,
                "mean_action_err": float(np.mean(bc["act_err"])) if len(bc["act_err"]) else 0.0,
            }
        )
        for t in range(len(drift)):
            drifts_by_step[t].append(float(drift[t]))
        for t in range(len(bc["act_err"])):
            act_err_by_step[t].append(float(bc["act_err"][t]))
        for t in range(len(bc["lateral"])):
            lat_bc_by_step[t].append(float(bc["lateral"][t]))
        for t in range(len(expert["lateral"])):
            lat_expert_by_step[t].append(float(expert["lateral"][t]))

    def _mean_curve(buckets: list[list[float]]) -> list[float]:
        return [float(np.mean(b)) if b else float("nan") for b in buckets]

    drift_curve = _mean_curve(drifts_by_step)
    act_curve = _mean_curve(act_err_by_step)
    lat_bc_curve = _mean_curve(lat_bc_by_step)
    lat_expert_curve = _mean_curve(lat_expert_by_step)

    # Compounding heuristic: final / early drift ratio (skip t=0).
    early = np.nanmean(drift_curve[1:6]) if len(drift_curve) > 5 else np.nan
    late = np.nanmean(drift_curve[-5:]) if len(drift_curve) >= 5 else np.nan
    ratio = float(late / early) if early and early > 1e-6 else float("nan")

    offroad_eps = [e for e in per_episode if e["bc_offroad"]]
    clean_eps = [e for e in per_episode if e["bc_clean"]]

    summary = {
        "n_scenarios": len(source),
        "reactive_agents": bool(args.reactive_agents),
        "expert_clean_rate": float(np.mean([e["expert_clean"] for e in per_episode])),
        "bc_clean_rate": float(np.mean([e["bc_clean"] for e in per_episode])),
        "mean_final_drift_m": float(np.mean([e["final_drift_m"] for e in per_episode])),
        "mean_max_drift_m": float(np.mean([e["max_drift_m"] for e in per_episode])),
        "drift_early_mean_m": early,
        "drift_late_mean_m": late,
        "drift_late_over_early": ratio,
        "mean_action_err_overall": float(np.mean([e["mean_action_err"] for e in per_episode])),
        "offroad_episodes": len(offroad_eps),
        "clean_episodes": len(clean_eps),
        "offroad_mean_final_drift_m": float(np.mean([e["final_drift_m"] for e in offroad_eps]))
        if offroad_eps
        else 0.0,
        "clean_mean_final_drift_m": float(np.mean([e["final_drift_m"] for e in clean_eps]))
        if clean_eps
        else 0.0,
        "drift_curve_mean_m": drift_curve,
        "action_err_curve_mean": act_curve,
        "lateral_bc_curve_mean_m": lat_bc_curve,
        "lateral_expert_curve_mean_m": lat_expert_curve,
        "per_episode": per_episode,
    }
    out = save_dir / "bc_drift_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BC drift diagnostic ({len(source)} scenarios, log-replay={not args.reactive_agents}) ===")
    print(f"expert clean={summary['expert_clean_rate']:.3f}  bc clean={summary['bc_clean_rate']:.3f}")
    print(f"mean position drift (BC vs expert replay): final={summary['mean_final_drift_m']:.3f}m  max={summary['mean_max_drift_m']:.3f}m")
    print(f"drift early (steps 1-5 avg)={early:.3f}m  late (last 5 steps avg)={late:.3f}m  ratio={ratio:.2f}x")
    print(f"mean action L2 err (BC vs expert@BC-state)={summary['mean_action_err_overall']:.4f}")
    print(f"offroad eps: final drift={summary['offroad_mean_final_drift_m']:.2f}m  "
          f"clean eps: final drift={summary['clean_mean_final_drift_m']:.2f}m")
    print("\nDrift curve (mean m vs expert replay), every 10 steps:")
    for t in range(0, min(len(drift_curve), max_len), 10):
        print(f"  t={t:2d}: drift={drift_curve[t]:.3f}m  act_err={act_curve[t]:.4f}  "
              f"lat_bc={lat_bc_curve[t]:.3f}  lat_expert={lat_expert_curve[t]:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
