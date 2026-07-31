"""Overfit SAC on failure cases as a *solvability diagnostic*.

The question this answers is not "can one policy generalize across the 245
failures" (that is what ``train_sac`` / the BC+SAC pipelines chase) but the
prerequisite: **can SAC solve each failure at all when it is allowed to overfit
a tiny set with full capacity and a generous step budget?** A case the policy
cannot even overfit is either genuinely unsolvable under this action space /
sim, or exposes a reward/termination artifact — both worth knowing before
sweeping the whole set.

Three modes (``--mode``):

1. ``single`` — one scenario per unit. Train until it is solved cleanly
   (goal reached, no offroad, no collision → ``eval/clean_success_rate == 1``),
   then stop and move to the next scenario.
2. ``batch`` — ``N`` scenarios per unit (``--n-per-unit``, default 10). Train
   until ``eval/clean_success_rate >= --success-threshold`` (default 0.95),
   then stop.
3. ``grouped`` — assign every failure to one primary reason group (offroad /
   overlap / goal_miss / …), apply a small per-group reward shaping, then
   overfit. By default each group is one unit (``--n-per-unit 0``); set
   ``--n-per-unit N`` to chunk a group into batches of N like mode 2.

Each unit trains a *fresh* SAC policy with its own replay buffer (overfitting,
not continual learning). Per-unit result records and an aggregate summary are
written under ``--save-dir``; existing unit results are skipped so a long run
resumes (use ``--overwrite`` to force).

Weights & Biases: **one run for the whole job** (not one per unit). Live eval
curves go under ``live/`` on a global step axis; SB3 RL scalars (actor/critic
loss, entropy, rollout reward, train collision/offroad) under ``train/`` /
``rollout/`` on the same axis; per-unit finals under ``unit/`` vs ``unit_idx``;
a summary table is logged at the end. Disable with ``--no-wandb``.

Runs on a GPU node (JAX/Waymax sim on GPU, SB3 MLP on CPU — same torch+JAX rule
as the other RL scripts). Login node has no GPU; launch via
``~/slurm/waymax/overfit_failures.sh`` or an ``srun`` alloc.

Examples
--------
    # Mode 1: solve each failure individually (subset for a quick look).
    python -m rl.overfit_failures --mode single --limit 20 \
        --save-dir runs/overfit_single

    # Mode 2: 10 per unit, stop at 95% clean.
    python -m rl.overfit_failures --mode batch --n-per-unit 10 \
        --save-dir runs/overfit_batch

    # Mode 3: one unit per reason group (whole group), shaped reward.
    python -m rl.overfit_failures --mode grouped --n-per-unit 0 \
        --save-dir runs/overfit_grouped

    # Mode 3 variant: chunk each group into batches of 10.
    python -m rl.overfit_failures --mode grouped --n-per-unit 10 \
        --save-dir runs/overfit_grouped_batches

    # Smoke: tiny end-to-end (needs a GPU).
    python -m rl.overfit_failures --mode single --smoke --save-dir /tmp/ovf_smoke
"""

from __future__ import annotations

# JAX must not preallocate the whole GPU (shared box); set before any JAX import.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import dataclasses
import gc
import json
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from rl.scenario_source import (
    CachedScenarioSource,
    ScenarioSource,
    _is_success,
)
from rl.sac_callbacks import (
    WandbSB3MetricsCallback,
    WaymaxEpisodeMetricsCallback,
    WaymaxPeriodicEvalCallback,
    make_monitored_env,
)
from rl.waymax_env import RewardConfig, WaymaxGymEnv

DEFAULT_FAILURE_DIR = "/zfsauton/scratch/mineuih/waymax_rs/failure_samples"

# Reason groups, in the priority order used to assign a case with several tags to
# a single primary group. "Fix the most upstream failure first": a case that did
# not even reach the goal is a goal_miss regardless of what else went wrong; among
# cases that reached, collisions rank above offroad. Override with --group-priority.
DEFAULT_GROUP_PRIORITY = ("goal_miss", "overlap", "offroad", "tl_violation", "other")

# Per-group reward shaping applied in --mode grouped. Safety terms dominate;
# progress/goal stay small (large values made reach saturate at 1.0 while clean
# stayed ~0.2–0.3). Clean-success stop criterion unchanged. Keys are
# RewardConfig fields; env-level knobs (reactive_agents) are handled in
# _apply_group_env_overrides.
GROUP_REWARD_SHAPING: dict[str, dict[str, Any]] = {
    # Left the drivable area: heavy offroad penalty + terminate; mild collision
    # so we do not trade offroad for crashes.
    "offroad": dict(
        offroad=-5.0, collision=-1.0, terminate_on_offroad=True,
    ),
    # Collided: heavy collision penalty + terminate; mild offroad so we do not
    # dodge into the curb (reactive IDM agents stay on).
    "overlap": dict(
        collision=-5.0, offroad=-1.0, terminate_on_collision=True,
    ),
    # Never reached: tiny goal bump over the already-small base + safety
    # terminate (do not buy reach with crashes).
    "goal_miss": dict(
        progress=0.2, goal_bonus=5.0,
        collision=-2.0, offroad=-2.0,
        terminate_on_collision=True, terminate_on_offroad=True,
    ),
    # No shaping for TL / uncategorized (env has no TL reward term today).
    "tl_violation": {},
    "other": {},
}


@dataclasses.dataclass
class OverfitUnit:
    """One overfitting job: a fixed scenario subset + its stop criterion."""

    unit_id: str
    indices: list[int]                 # indices into the full ScenarioSource
    reward_cfg: RewardConfig
    success_threshold: float
    budget: int                        # max env steps before giving up
    group: str | None = None
    reactive_agents: bool | None = None  # None -> use the CLI default


# --------------------------------------------------------------------------- #
# Failure reasons / grouping
# --------------------------------------------------------------------------- #
def load_failure_reasons(failure_dir: str) -> dict[tuple[str, int], list[str]]:
    """Map ``(tfrecord, scenario_idx) -> [reason tags]`` for every failure JSON.

    Tags: ``goal_miss`` (``goal_reached`` false), ``overlap``, ``offroad``,
    ``tl_violation``; ``other`` if a failure carries none of these flags.
    """
    reasons: dict[tuple[str, int], list[str]] = {}
    for jp in sorted(glob(str(Path(failure_dir) / "**" / "*.json"), recursive=True)):
        if Path(jp).name == "summary.json" or jp.endswith(".instructions.json"):
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict) or "tfrecord" not in rec or "scenario_idx" not in rec:
            continue
        if _is_success(rec):
            continue
        tags: list[str] = []
        if not rec.get("goal_reached", False):
            tags.append("goal_miss")
        if rec.get("overlap", False):
            tags.append("overlap")
        if rec.get("offroad", False):
            tags.append("offroad")
        if rec.get("tl_violation", False):
            tags.append("tl_violation")
        if not tags:
            tags.append("other")
        reasons[(str(rec["tfrecord"]), int(rec["scenario_idx"]))] = tags
    return reasons


def primary_group(tags: list[str], priority: tuple[str, ...]) -> str:
    for g in priority:
        if g in tags:
            return g
    return tags[0] if tags else "other"


# --------------------------------------------------------------------------- #
# Reward presets
# --------------------------------------------------------------------------- #
# safety_first: terminate on collision/offroad with huge penalties so those are
# never worth it; keep progress dense so the policy can learn a safe path
# instead of waiting for a sparse goal signal; modest goal_bonus so arrival is
# still desirable once safe. Route progress (optional flag) further discourages
# beelining through agents / off-road.
SAFETY_FIRST_PRESET: dict[str, Any] = dict(
    r_progress=2.0,
    r_collision=-50.0,
    r_offroad=-50.0,
    r_goal_bonus=5.0,
    terminate_on_collision=True,
    terminate_on_offroad=True,
    route_reward=True,
    r_lateral_penalty=1.0,
)


def _apply_reward_preset(args: argparse.Namespace) -> None:
    """Mutate ``args`` in place when ``--preset`` is set."""
    if not args.preset:
        return
    if args.preset == "safety-first":
        for k, v in SAFETY_FIRST_PRESET.items():
            setattr(args, k, v)
        # Grouped shaping would soften these; force base reward for every group.
        args.no_group_shaping = True
        print(
            "[overfit] preset=safety-first: "
            f"progress={args.r_progress}, collision={args.r_collision}, "
            f"offroad={args.r_offroad}, goal_bonus={args.r_goal_bonus}, "
            f"terminate_col/off=True, route_reward={args.route_reward}"
        )
        return
    raise ValueError(f"Unknown --preset {args.preset!r}")


# --------------------------------------------------------------------------- #
# Env / reward construction
# --------------------------------------------------------------------------- #
def base_reward_cfg(args) -> RewardConfig:
    return RewardConfig(
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


def _apply_group_env_overrides(group: str | None, default_reactive: bool) -> bool:
    """Reactive-agent choice per group (overlap benefits most from IDM)."""
    if group == "overlap":
        return True
    return default_reactive


def _make_base_env(source, args, reward_cfg: RewardConfig, *, seed: int,
                   sequential: bool, reactive_agents: bool) -> WaymaxGymEnv:
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
        reactive_agents=reactive_agents,
        idm_desired_vel=args.idm_desired_vel,
    )


# --------------------------------------------------------------------------- #
# Unit construction per mode
# --------------------------------------------------------------------------- #
def _chunk(seq: list[int], n: int) -> list[list[int]]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def build_units(args, full: ScenarioSource) -> list[OverfitUnit]:
    base = base_reward_cfg(args)
    n_total = len(full)

    if args.mode == "single":
        thr = 1.0 if args.success_threshold is None else args.success_threshold
        return [
            OverfitUnit(
                unit_id=f"single_{i:04d}",
                indices=[i],
                reward_cfg=base,
                success_threshold=thr,
                budget=args.budget,
            )
            for i in range(n_total)
        ]

    thr = 0.95 if args.success_threshold is None else args.success_threshold

    if args.mode == "batch":
        units = []
        for u, idxs in enumerate(_chunk(list(range(n_total)), args.n_per_unit)):
            units.append(
                OverfitUnit(
                    unit_id=f"batch_{u:03d}",
                    indices=idxs,
                    reward_cfg=base,
                    success_threshold=thr,
                    budget=args.budget,
                )
            )
        return units

    if args.mode == "grouped":
        priority = tuple(args.group_priority.split(",")) if args.group_priority else DEFAULT_GROUP_PRIORITY
        reasons = load_failure_reasons(args.failure_dir)
        # Group indices by primary reason (using each cached scenario's spec).
        by_group: dict[str, list[int]] = defaultdict(list)
        n_unmatched = 0
        for i in range(n_total):
            spec = full.spec(i)
            tags = reasons.get((spec.tfrecord, spec.scenario_idx))
            if tags is None:
                n_unmatched += 1
                g = "other"
            else:
                g = primary_group(tags, priority)
            by_group[g].append(i)
        if n_unmatched:
            print(f"[overfit] warning: {n_unmatched} cached scenarios had no matching "
                  f"failure JSON; assigned to 'other'.")

        # n_per_unit <= 0 → one unit per whole group (default for this mode).
        chunk_n = args.n_per_unit if args.n_per_unit > 0 else None
        units = []
        for g in sorted(by_group, key=lambda k: (-len(by_group[k]), k)):
            idxs_all = by_group[g]
            shaping = {} if args.no_group_shaping else GROUP_REWARD_SHAPING.get(g, {})
            reward_cfg = dataclasses.replace(base, **shaping) if shaping else base
            reactive = _apply_group_env_overrides(g, args.reactive_agents)
            chunks = [idxs_all] if chunk_n is None else _chunk(idxs_all, chunk_n)
            for u, idxs in enumerate(chunks):
                unit_id = (
                    f"group_{g}" if chunk_n is None else f"group_{g}_batch_{u:03d}"
                )
                units.append(
                    OverfitUnit(
                        unit_id=unit_id,
                        indices=idxs,
                        reward_cfg=reward_cfg,
                        success_threshold=thr,
                        budget=args.budget,
                        group=g,
                        reactive_agents=reactive,
                    )
                )
        print(f"[overfit] groups: "
              + ", ".join(f"{g}={len(by_group[g])}" for g in sorted(by_group))
              + (f" | whole-group units" if chunk_n is None
                 else f" | {chunk_n}/unit chunks"))
        return units

    raise ValueError(f"Unknown mode {args.mode!r}")


# --------------------------------------------------------------------------- #
# Training a single unit
# --------------------------------------------------------------------------- #
def run_unit(unit: OverfitUnit, full: ScenarioSource, args, *, device: str,
             out_dir: Path, log_fn=None, wandb_step_offset: int = 0,
             log_rl_to_wandb: bool = False) -> dict[str, Any]:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    reactive = args.reactive_agents if unit.reactive_agents is None else unit.reactive_agents
    sub = CachedScenarioSource(
        [full.get(i) for i in unit.indices],
        [full.spec(i) for i in unit.indices],
        label=unit.unit_id,
    )

    def _base_fn(seed: int):
        return lambda: _make_base_env(
            sub, args, unit.reward_cfg, seed=seed, sequential=False, reactive_agents=reactive
        )

    vec_env = DummyVecEnv(
        [make_monitored_env(_base_fn(args.seed + i)) for i in range(args.n_envs)]
    )
    eval_env = _make_base_env(
        sub, args, unit.reward_cfg, seed=args.seed + 10_000, sequential=True,
        reactive_agents=reactive,
    )

    ent_coef: str | float = args.ent_coef
    if isinstance(ent_coef, str) and ent_coef != "auto":
        ent_coef = float(ent_coef)

    # Need a TB writer for SB3 to keep logger scalars populated; always on when
    # we are mirroring those scalars to wandb.
    want_tb = bool(args.tensorboard) or bool(log_rl_to_wandb)
    tb_dir = str(out_dir / unit.unit_id) if want_tb else None
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=ent_coef,
        verbose=1 if (args.verbose and log_rl_to_wandb) else 0,
        device=device,
        seed=args.seed,
        tensorboard_log=tb_dir,
    )

    eval_cb = WaymaxPeriodicEvalCallback(
        eval_env,
        eval_freq=args.eval_every,
        n_episodes=len(sub),
        deterministic=True,
        verbose=1 if args.verbose else 0,
        stop_threshold=unit.success_threshold,
        stop_patience=args.stop_patience,
        log_fn=log_fn,
    )
    cb_list: list = [
        WaymaxEpisodeMetricsCallback(window=args.train_metrics_window),
        eval_cb,
    ]
    if log_rl_to_wandb:
        cb_list.append(
            WandbSB3MetricsCallback(
                step_offset=int(wandb_step_offset),
                log_freq=max(100, int(args.eval_every) // 4),
            )
        )
    callbacks = CallbackList(cb_list)

    t0 = time.time()
    model.learn(total_timesteps=unit.budget, callback=callbacks, progress_bar=False)
    wall = time.time() - t0

    final = eval_cb.last_metrics or {}
    if args.save_models and eval_cb.solved:
        model.save((out_dir / unit.unit_id / "model").as_posix())

    record = {
        "unit_id": unit.unit_id,
        "mode": args.mode,
        "group": unit.group,
        "n_scenarios": len(unit.indices),
        "indices": unit.indices,
        "scenarios": [
            {"tfrecord": full.spec(i).tfrecord, "scenario_idx": full.spec(i).scenario_idx}
            for i in unit.indices
        ],
        "success_threshold": unit.success_threshold,
        "solved": bool(eval_cb.solved),
        "timesteps_used": int(model.num_timesteps),
        "timesteps_budget": int(unit.budget),
        "solved_at_timestep": eval_cb.solved_at_timestep,
        "final_clean_success_rate": float(final.get("eval/clean_success_rate", float("nan"))),
        "final_goal_reached_rate": float(final.get("eval/goal_reached_rate", float("nan"))),
        "final_collision_rate": float(final.get("eval/collision_rate", float("nan"))),
        "final_offroad_rate": float(final.get("eval/offroad_rate", float("nan"))),
        "reactive_agents": bool(reactive),
        "reward_config": dataclasses.asdict(unit.reward_cfg),
        "wall_time_s": round(wall, 1),
    }

    # Free GPU (JAX env) + torch model between units.
    del model, vec_env, eval_env, eval_cb, callbacks, sub
    gc.collect()
    return record


# --------------------------------------------------------------------------- #
# Wandb: one run for the whole overfit job (not one run per unit)
# --------------------------------------------------------------------------- #
def _init_wandb(args, *, n_units: int, n_scenarios: int, device: str):
    """Start a single wandb run spanning all overfit units. Returns the run or None."""
    use_wandb = (not args.no_wandb) and (not args.smoke) and (not args.dry_run)
    if not use_wandb:
        return None
    import wandb

    run_name = args.wandb_run_name or f"overfit_{args.mode}_{Path(args.save_dir).name}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        config={**vars(args), "n_units": n_units, "n_scenarios": n_scenarios, "device": device},
        # Do NOT sync_tensorboard: SB3 would restart step counters every unit and
        # pollute one run with overlapping curves. We log explicitly instead.
        sync_tensorboard=False,
        save_code=False,
    )
    # Cross-unit overview plotted against unit index.
    wandb.define_metric("unit_idx")
    wandb.define_metric("unit/*", step_metric="unit_idx")
    # Within-unit live evals + SB3 RL scalars share one global step axis so
    # curves do not collide across units (sync_tensorboard would reset each unit).
    wandb.define_metric("global_step")
    wandb.define_metric("live/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("rollout/*", step_metric="global_step")
    wandb.define_metric("time/*", step_metric="global_step")
    return run


def _make_unit_log_fn(unit: OverfitUnit, unit_idx: int, budget: int):
    """Log live eval metrics into the shared wandb run under ``live/``."""
    import wandb

    # Offset so unit k's timesteps sit in [k*budget, (k+1)*budget).
    offset = int(unit_idx) * int(budget)

    def _log(metrics: dict[str, float], timesteps: int) -> None:
        payload = {
            "global_step": offset + int(timesteps),
            "live/unit_idx": int(unit_idx),
            "live/clean_success_rate": float(metrics.get("eval/clean_success_rate", float("nan"))),
            "live/goal_reached_rate": float(metrics.get("eval/goal_reached_rate", float("nan"))),
            "live/collision_rate": float(metrics.get("eval/collision_rate", float("nan"))),
            "live/offroad_rate": float(metrics.get("eval/offroad_rate", float("nan"))),
            "live/mean_max_lateral_m": float(
                metrics.get("eval/mean_max_lateral_m", float("nan"))
            ),
            "live/unit_timestep": int(timesteps),
        }
        # Per-group live curves (handy in grouped mode).
        g = unit.group or "all"
        payload[f"live/{g}/clean_success_rate"] = payload["live/clean_success_rate"]
        wandb.log(payload)

    return _log


def _log_unit_to_wandb(record: dict[str, Any], unit_idx: int, summary: dict[str, Any]) -> None:
    import wandb

    wandb.log({
        "unit_idx": int(unit_idx),
        "unit/id": record["unit_id"],
        "unit/group": record.get("group") or "",
        "unit/n_scenarios": int(record["n_scenarios"]),
        "unit/solved": int(bool(record["solved"])),
        "unit/timesteps_used": int(record["timesteps_used"]),
        "unit/solved_at_timestep": (
            int(record["solved_at_timestep"])
            if record.get("solved_at_timestep") is not None else -1
        ),
        "unit/clean_success_rate": float(record["final_clean_success_rate"]),
        "unit/goal_reached_rate": float(record["final_goal_reached_rate"]),
        "unit/collision_rate": float(record["final_collision_rate"]),
        "unit/offroad_rate": float(record["final_offroad_rate"]),
        "unit/wall_time_s": float(record["wall_time_s"]),
        # Running aggregates (same step axis).
        "unit/running_solved_rate": float(summary["solved_rate"]),
        "unit/running_n_solved": int(summary["n_solved"]),
    })


def _finish_wandb(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import wandb

    cols = [
        "unit_id", "group", "n_scenarios", "solved", "timesteps_used",
        "solved_at_timestep", "final_clean_success_rate", "final_goal_reached_rate",
        "final_collision_rate", "final_offroad_rate", "wall_time_s",
    ]
    table = wandb.Table(columns=cols)
    for r in records:
        table.add_data(*[r.get(c) for c in cols])
    payload = {f"summary/{k}": v for k, v in summary.items() if not isinstance(v, dict)}
    payload["summary/units"] = table
    if isinstance(summary.get("by_group"), dict):
        for g, gs in summary["by_group"].items():
            for k, v in gs.items():
                if not isinstance(v, dict):
                    payload[f"summary/group/{g}/{k}"] = v
    wandb.log(payload)
    wandb.finish()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _rate(rs):
        return sum(int(r["solved"]) for r in rs) / max(len(rs), 1)

    def _median_steps(rs):
        s = [r["solved_at_timestep"] for r in rs if r["solved"] and r["solved_at_timestep"]]
        return float(np.median(s)) if s else None

    summary: dict[str, Any] = {
        "n_units": len(records),
        "n_solved": sum(int(r["solved"]) for r in records),
        "solved_rate": _rate(records),
        "n_scenarios_total": sum(r["n_scenarios"] for r in records),
        "median_timesteps_to_solve": _median_steps(records),
    }
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[r.get("group") or "_all_"].append(r)
    if set(by_group) != {"_all_"}:
        summary["by_group"] = {
            g: {
                "n_units": len(rs),
                "n_solved": sum(int(r["solved"]) for r in rs),
                "solved_rate": _rate(rs),
                "median_timesteps_to_solve": _median_steps(rs),
            }
            for g, rs in sorted(by_group.items())
        }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()

    import torch  # noqa: F401 (import here so --help works without torch)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mode-specific defaults for --n-per-unit.
    if args.n_per_unit is None:
        args.n_per_unit = 0 if args.mode == "grouped" else 10

    if args.smoke:
        args.limit = args.limit or 2
        args.budget = min(args.budget, 400)
        args.learning_starts = min(args.learning_starts, 50)
        args.buffer_size = min(args.buffer_size, 5_000)
        args.eval_every = min(args.eval_every, 100)
        if args.n_per_unit > 0:
            args.n_per_unit = min(args.n_per_unit, 2)

    _apply_reward_preset(args)

    full = ScenarioSource.from_failure_dir(
        args.failure_dir,
        max_num_objects=args.max_num_objects,
        limit=args.limit,
        drop_init_violations=not args.keep_init_violations,
        scenario_idxs=args.scenario_idx or None,
        tfrecord_substr=args.tfrecord_substr,
    )
    print(f"[overfit] {len(full)} failure scenarios | mode={args.mode} | device={device}")
    print(
        f"[overfit] reward: progress={args.r_progress} collision={args.r_collision} "
        f"offroad={args.r_offroad} goal_bonus={args.r_goal_bonus} "
        f"term_col={args.terminate_on_collision} term_off={args.terminate_on_offroad} "
        f"route={args.route_reward}"
    )

    units = build_units(args, full)
    print(f"[overfit] {len(units)} unit(s) to run.")

    if args.dry_run:
        base = base_reward_cfg(args)
        for u in units:
            line = (f"  {u.unit_id}: {len(u.indices)} scene(s), "
                    f"thr={u.success_threshold:.2f}, budget={u.budget}")
            if u.group is not None:
                shaped = {k: v for k, v in dataclasses.asdict(u.reward_cfg).items()
                          if getattr(base, k) != v}
                line += f", group={u.group}, reactive={u.reactive_agents}"
                if shaped:
                    line += ", shaped=" + str(shaped)
            print(line)
        print(f"[overfit] dry-run: {len(units)} unit(s); no training performed.")
        return

    save_dir = Path(args.save_dir)
    units_dir = save_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps({**vars(args), "n_units": len(units), "device": device}, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )

    wb_run = _init_wandb(args, n_units=len(units), n_scenarios=len(full), device=device)
    if wb_run is not None:
        print(f"[overfit] wandb run: {wb_run.name} ({wb_run.url})")

    records: list[dict[str, Any]] = []
    try:
        for k, unit in enumerate(units):
            result_path = units_dir / f"{unit.unit_id}.json"
            if result_path.exists() and not args.overwrite:
                record = json.loads(result_path.read_text())
                records.append(record)
                print(f"[overfit] ({k + 1}/{len(units)}) {unit.unit_id}: cached, skipping.")
                if wb_run is not None:
                    _log_unit_to_wandb(record, k, summarize(records))
                continue

            print(f"[overfit] ({k + 1}/{len(units)}) {unit.unit_id}: "
                  f"{len(unit.indices)} scene(s), thr={unit.success_threshold:.2f}, "
                  f"budget={unit.budget}"
                  + (f", group={unit.group}" if unit.group else ""))
            log_fn = (
                _make_unit_log_fn(unit, k, unit.budget) if wb_run is not None else None
            )
            record = run_unit(
                unit, full, args, device=device, out_dir=units_dir, log_fn=log_fn,
                wandb_step_offset=k * unit.budget,
                log_rl_to_wandb=wb_run is not None,
            )
            result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)

            status = "SOLVED" if record["solved"] else "unsolved"
            print(f"[overfit]   -> {status} @ {record['timesteps_used']} steps "
                  f"(clean={record['final_clean_success_rate']:.2f}, "
                  f"{record['wall_time_s']}s)")

            # Rewrite the running summary after every unit (cheap; survives crashes).
            summary = summarize(records)
            (save_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            if wb_run is not None:
                _log_unit_to_wandb(record, k, summary)
    finally:
        summary = summarize(records)
        (save_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        _write_summary_csv(save_dir / "summary.csv", records)
        if wb_run is not None:
            _finish_wandb(records, summary)

    print(f"[overfit] DONE: {summary['n_solved']}/{summary['n_units']} units solved "
          f"(rate={summary['solved_rate']:.2f}). Summary -> {save_dir / 'summary.json'}")


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    cols = [
        "unit_id", "mode", "group", "n_scenarios", "solved", "timesteps_used",
        "timesteps_budget", "solved_at_timestep", "final_clean_success_rate",
        "final_goal_reached_rate", "final_collision_rate", "final_offroad_rate",
        "wall_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c) for c in cols])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overfit SAC on failure cases (solvability diagnostic).")

    # Mode / unit selection.
    p.add_argument("--mode", choices=["single", "batch", "grouped"], required=True)
    p.add_argument("--n-per-unit", type=int, default=None,
                   help="Scenarios per unit. batch default 10; grouped default 0 "
                        "(= one unit per whole reason group). Set >0 to chunk.")
    p.add_argument("--success-threshold", type=float, default=None,
                   help="Clean-success rate to stop a unit. Default: 1.0 (single), 0.95 (batch/grouped).")
    p.add_argument("--stop-patience", type=int, default=1,
                   help="Consecutive evals meeting the threshold before stopping (default 1).")
    p.add_argument("--budget", type=int, default=50_000,
                   help="Max env steps per unit before giving up (default 50k).")
    p.add_argument("--group-priority", type=str, default=None,
                   help="Comma-separated reason priority for --mode grouped "
                        f"(default: {','.join(DEFAULT_GROUP_PRIORITY)}).")
    p.add_argument("--no-group-shaping", action="store_true",
                   help="Use the base reward for every group (mode grouped).")
    p.add_argument("--keep-init-violations", action="store_true",
                   help="Keep scenes where the SDC is already overlapped/offroad "
                        "at reset (default: drop them as unsolvable).")

    # Data.
    p.add_argument("--failure-dir", type=str, default=DEFAULT_FAILURE_DIR)
    p.add_argument("--limit", type=int, default=None, help="Cap failure scenarios loaded.")
    p.add_argument("--scenario-idx", type=int, nargs="+", default=None,
                   help="Only load these WOMD scenario_idx values (pin one unsolved case).")
    p.add_argument("--tfrecord-substr", type=str, default=None,
                   help="Require this substring in the tfrecord path (e.g. "
                        "tfrecord-00000-of-01000) to disambiguate shared indices.")
    p.add_argument("--max-num-objects", type=int, default=None)
    p.add_argument(
        "--preset",
        choices=["safety-first"],
        default=None,
        help="Reward preset. safety-first: huge col/off penalties + terminate, "
             "dense progress=2, modest goal_bonus=5, route progress on.",
    )

    # Env / reward (defaults mirror the policy-friendly bc_cli defaults).
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--action-space", type=str, default="bicycle", choices=["bicycle", "delta"])
    p.add_argument("--delta-max-dx", type=float, default=2.0)
    p.add_argument("--delta-max-dy", type=float, default=0.5)
    p.add_argument("--delta-max-dyaw", type=float, default=0.2)
    p.add_argument("--r-progress", type=float, default=0.2,
                   help="Per-meter progress reward (default 0.2; 1.0+ overweights reach).")
    p.add_argument("--r-action-penalty", type=float, default=0.01)
    p.add_argument("--r-collision", type=float, default=-0.25)
    p.add_argument("--r-offroad", type=float, default=-0.25)
    p.add_argument("--r-goal-bonus", type=float, default=3.0,
                   help="One-time goal arrival bonus (default 3; 10–20 was too high).")
    p.add_argument("--route-reward", action="store_true")
    p.add_argument("--r-lateral-penalty", type=float, default=0.5)
    p.add_argument("--terminate-on-offroad", action="store_true")
    p.add_argument("--terminate-on-collision", action="store_true")
    reactive = p.add_mutually_exclusive_group()
    reactive.add_argument("--reactive-agents", dest="reactive_agents", action="store_true",
                          help="IDM sim agents for non-ego objects (default).")
    reactive.add_argument("--no-reactive-agents", dest="reactive_agents", action="store_false",
                          help="Log-replay non-ego agents (legacy).")
    p.set_defaults(reactive_agents=True)
    p.add_argument("--idm-desired-vel", type=float, default=30.0)

    # SAC (overfit-tuned defaults: fast learning start, modest buffer).
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--ent-coef", type=str, default="auto")
    p.add_argument("--eval-every", type=int, default=2_000,
                   help="Run the clean-success eval / stop check every N env steps.")
    p.add_argument("--train-metrics-window", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # Output / control.
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--save-models", action="store_true",
                   help="Save the SAC checkpoint for each solved unit.")
    p.add_argument("--tensorboard", action="store_true",
                   help="Write per-unit TensorBoard logs under save-dir/units/<unit>.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run units even if a result JSON already exists.")
    p.add_argument("--verbose", action="store_true", help="Print per-eval metrics.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and print the unit/group plan, then exit (no training).")
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end run (needs a GPU).")

    # Wandb: one run for the whole job (not one per unit). On by default.
    p.add_argument("--wandb-project", type=str, default="overfit-failures")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None,
                   help="Optional wandb group tag (e.g. grouped-n10).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb (also off for --smoke / --dry-run).")

    return p.parse_args()


if __name__ == "__main__":
    main()
