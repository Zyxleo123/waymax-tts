"""Establish clean-success ceiling on failure scenarios before SAC sweeps.

Runs expert bicycle replay and (optionally) a BC/SAC checkpoint on the failure
JSON set with log-replay and IDM agents.

Usage (GPU node recommended for full set):
    python -m rl.diagnose_failure_ceiling \\
        --failure-dir /path/to/failure_samples

    python -m rl.diagnose_failure_ceiling \\
        --failure-dir /path/to/failure_samples \\
        --model runs/bc_sac_staged/bc_sac_staged.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# ScenarioSource._load() silently drops any shard whose load raises, including a
# transient "cuInit failed" from JAX's lazy CUDA backend probe on GPU-less nodes
# (whichever shard is first in iteration order eats the flake and vanishes).
# Force CPU so that probe never happens; matches rl/evaluate_pseudo_gt.py.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.bc_cli import add_data_args, add_env_args
from rl.bc_core import (
    evaluate_expert_replay_on_source,
    evaluate_policy_on_source,
)
from rl.scenario_source import ScenarioSource


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_env_args(p)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional SB3 checkpoint (.zip) for BC/SAC policy eval on failures.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap failure scenarios.")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write JSON summary here (default: stdout only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    source = ScenarioSource.from_failure_dir(
        args.failure_dir,
        max_num_objects=args.max_num_objects,
        limit=args.limit,
    )
    n = len(source)
    print(f"[diagnose_failure_ceiling] {n} failure scenarios from {args.failure_dir}")

    summary: dict[str, float | int | str] = {"n_failures": n}

    for reactive, suffix in ((False, "log_replay"), (True, "idm")):
        expert = evaluate_expert_replay_on_source(
            source,
            args,
            n_episodes=n,
            reactive_agents=reactive,
            metrics_prefix=f"expert_replay/{suffix}",
        )
        summary.update(expert)
        print(
            f"[diagnose_failure_ceiling] expert replay ({suffix}): "
            f"clean={expert[f'expert_replay/{suffix}/clean_success_rate']:.3f} "
            f"collision={expert[f'expert_replay/{suffix}/collision_rate']:.3f} "
            f"offroad={expert[f'expert_replay/{suffix}/offroad_rate']:.3f}"
        )

        if args.model:
            from stable_baselines3 import SAC

            from rl.eval_sac import resolve_sb3_checkpoint

            model = SAC.load(resolve_sb3_checkpoint(args.model), device=args.device)
            policy = evaluate_policy_on_source(
                model,
                source,
                args,
                n_episodes=n,
                reactive_agents=reactive,
                metrics_prefix=f"policy/{suffix}",
            )
            summary.update(policy)
            print(
                f"[diagnose_failure_ceiling] policy ({suffix}): "
                f"clean={policy[f'policy/{suffix}/clean_success_rate']:.3f} "
                f"collision={policy[f'policy/{suffix}/collision_rate']:.3f} "
                f"offroad={policy[f'policy/{suffix}/offroad_rate']:.3f}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[diagnose_failure_ceiling] wrote {out_path}")


if __name__ == "__main__":
    main()
