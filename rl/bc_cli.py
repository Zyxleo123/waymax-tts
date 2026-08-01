"""Shared CLI flags for SB3 BC / BC+SAC training scripts."""

from __future__ import annotations

import argparse

import numpy as np

from rl.bc_core import DEFAULT_BC_CACHE_DIR, DEFAULT_FAILURE_DIR, DEFAULT_WOMD_DIR


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--failure-dir", type=str, default=DEFAULT_FAILURE_DIR)
    p.add_argument("--womd-dir", type=str, default=DEFAULT_WOMD_DIR)
    p.add_argument("--limit-failures", type=int, default=None,
                   help="Cap failure scenarios loaded for SAC / mixed sampling.")
    p.add_argument("--max-num-objects", type=int, default=None)
    p.add_argument("--total-shards", type=int, default=1000)


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--action-space", type=str, default="bicycle", choices=["bicycle", "delta"])
    p.add_argument("--delta-max-dx", type=float, default=6.0)
    p.add_argument("--delta-max-dy", type=float, default=6.0)
    p.add_argument("--delta-max-dyaw", type=float, default=float(np.pi))
    p.add_argument("--r-progress", type=float, default=1.0)
    p.add_argument("--r-action-penalty", type=float, default=0.01)
    p.add_argument("--r-collision", type=float, default=-0.25)
    p.add_argument("--r-offroad", type=float, default=-0.25)
    p.add_argument("--r-goal-bonus", type=float, default=10.0)
    p.add_argument("--route-reward", action="store_true")
    p.add_argument("--r-lateral-penalty", type=float, default=0.5)
    p.add_argument("--terminate-on-offroad", action=argparse.BooleanOptionalAction,
                   default=False)
    # See train_sac.py: store_true here silently overrode the RewardConfig
    # default of True.
    p.add_argument("--terminate-on-collision", action=argparse.BooleanOptionalAction,
                   default=True)
    reactive = p.add_mutually_exclusive_group()
    reactive.add_argument(
        "--reactive-agents", dest="reactive_agents", action="store_true",
        help="IDM sim agents for non-ego objects (default).",
    )
    reactive.add_argument(
        "--no-reactive-agents", dest="reactive_agents", action="store_false",
        help="Log-replay non-ego agents (legacy).",
    )
    p.set_defaults(reactive_agents=True)
    p.add_argument("--idm-desired-vel", type=float, default=30.0)


def _parse_bc_scenarios(value: str) -> int | None:
    if str(value).lower() == "all":
        return None
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(
            "--bc-scenarios must be a non-negative integer or 'all'"
        )
    return n


def add_bc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bc-scenarios",
        type=_parse_bc_scenarios,
        default="all",
        help="Expert scenarios for BC warm-up: positive int samples that many "
        "from the shuffled expert split; 'all' (default) uses every scenario in "
        "the non-failure shard pool once; 0 skips BC collection.",
    )
    p.add_argument("--bc-epochs", type=int, default=20,
                   help="Supervised BC passes over the collected dataset (0 = skip BC).")
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--bc-seed", type=int, default=0)
    p.add_argument(
        "--bc-cache-dir",
        type=str,
        default=DEFAULT_BC_CACHE_DIR,
        help="Directory for persisted BC transition datasets (obs/actions + meta.json).",
    )
    p.add_argument(
        "--bc-rebuild-cache",
        action="store_true",
        help="Ignore any existing BC cache and re-collect transitions.",
    )
    p.add_argument(
        "--bc-no-cache",
        action="store_true",
        help="Do not read or write the BC transition cache.",
    )


def add_sac_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--ent-coef", type=str, default="auto")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--expert-mix-prob", type=float, default=0.5,
                   help="Probability of sampling an expert (non-failure) scenario in SAC.")
    p.add_argument(
        "--actor-freeze-timesteps",
        type=int,
        default=0,
        help="Env steps with actor frozen before full SAC (0 = unstaged / actor trains immediately).",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)


def add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--wandb-project", type=str, default="sb3-bc-sac")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--eval-freq", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--train-metrics-window", type=int, default=100)
    p.add_argument("--smoke", action="store_true")


def postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    if isinstance(args.ent_coef, str) and not args.ent_coef.startswith("auto"):
        args.ent_coef = float(args.ent_coef)
    return args
