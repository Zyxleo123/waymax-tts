"""Entrypoint for the reward-resolution experiment arms (Stage 0 / Stage 1 / Stage 2).

Runs the *instrumented* Diffusion-ES with a chosen selection score over a set of
scenes, reusing the vendored, planner-agnostic ``simulation.runner.run`` for
rollout + the original binary success evaluation. Final success is ALWAYS the
original binary criterion (goal_reached & ~overlap & ~offroad), regardless of the
score ES used internally.

Arms (pair with identical --seed and --scenario_indices):
  --selection binary   reference arm  (Stage 0)
  --selection dense    dense-return oracle arm  (Stage 1 arm B)
  --init_select diverse_safe --init_bank_multiplier 4
                       Stage-2 diversified (safe-aware) initialization
  --init_select sac_safe --sac_bank_path ... --sac_manifest ...
                       Stage-3 offline SAC traj init, *replacing* the diffusion
                       init bank (29 SMX-hit scenes)
  --init_select diverse_sac_safe --init_bank_multiplier 4 --sac_frac 0.5
                       Stage-4 mixed init: SAC trajs *augment* the diffusion
                       bank, sac_frac*K slots reserved for SAC members

Example (one tfrecord):
  python experiments/run_experiment.py \
    --diffusion_ckpt_path $CKPT \
    --tfrecord_dir es_baseline/repro/tf00000 \
    --scenario_indices 71,92,113,134,187,252,315,377,0,1,2,3 \
    --selection dense --seed 0 --num_worlds 12 --num_scenarios 12 \
    --exp_dir es_baseline/experiments/out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# --- GPU memory hygiene: must run before TF/JAX touch the device ------------
# The vendored stack imports TensorFlow (data/scenario_loader.py for tfrecord
# parsing, viz/render.py for video writing). TF's default GPU allocator claims
# essentially the whole device on first use, so with XLA_PYTHON_CLIENT_PREALLOCATE
# =false JAX later fails to grow and dies with RESOURCE_EXHAUSTED mid-rollout.
# Nothing here needs TF on the GPU, so hide the device from it entirely.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
try:
    import tensorflow as _tf

    _tf.config.set_visible_devices([], "GPU")
except Exception as _e:  # TF missing, or devices already initialized
    print(f"[run_experiment] could not hide GPU from TensorFlow: {_e}")
# ---------------------------------------------------------------------------

from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import orbax_patch
orbax_patch.apply()  # fix arg-less orbax restore before any checkpoint load

from train_diffusion.utils.infer import load_model_for_inference
from simulation.runner import run
from experiments.es_instrumented import (
    InstrumentedESPlanner, INIT_SELECT_CHOICES, MIXED_INIT_CHOICES, SAC_INIT_CHOICES,
)
from experiments.dense_scorer import DenseConfig, DenseWeights
from experiments.sac_bank import load_sac_init_bank
from experiments.wandb_log import WandbLogger

os.environ["PYTHONWARNINGS"] = "ignore"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_ckpt_path", required=True)
    p.add_argument("--tfrecord_dir", required=True,
                   help="dir containing a 'training/' subdir with the tfrecord(s)")
    p.add_argument("--split", default="training")
    p.add_argument("--lane_graph_dir",
                   default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs")
    p.add_argument("--exp_dir", default=str(REPO_ROOT / "experiments" / "out"))
    p.add_argument("--tag", default=None)
    p.add_argument("--selection", choices=["binary", "dense"], default="binary")
    p.add_argument("--init_select",
                   choices=list(INIT_SELECT_CHOICES),
                   default="none",
                   help="Init: none=vanilla K; diverse/diverse_safe=FPS over an "
                        "oversized diffusion bank; sac/sac_safe=offline SAC traj bank "
                        "*replaces* the diffusion init; diverse_sac/diverse_sac_safe="
                        "SAC trajs *augment* the diffusion bank (see --sac_frac)")
    p.add_argument("--sac_frac", type=float, default=0.5,
                   help="diverse_sac*: fraction of K reserved for SAC members "
                        "(0=pure diffusion, 1=pure SAC); rest comes from the "
                        "diffusion bank. Each side is safe-first + FPS.")
    p.add_argument("--init_bank_multiplier", type=int, default=1,
                   help="oversample factor for the diffusion bank (e.g. 4 -> sample 4K, keep K)")
    p.add_argument("--sac_bank_path", default=None,
                   help="scratch NPZ from dump_sac_init_bank.py (required for sac*)")
    p.add_argument("--sac_manifest", default=None,
                   help="es_scenes_manifest.json (required for sac*)")
    p.add_argument("--tf_shard", default=None,
                   help="WOMD shard id e.g. 00000 (auto-inferred from tfrecord_dir if omitted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario_indices", default=None)
    p.add_argument("--num_worlds", type=int, default=12)
    p.add_argument("--num_scenarios", type=int, default=12)
    p.add_argument("--max_num_objects", type=int, default=128)
    p.add_argument("--start_timestep", type=int, default=10)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--instruction_interval_steps", type=int, default=40)
    p.add_argument("--population_size", type=int, default=64)
    p.add_argument("--resample_timesteps", type=int, default=3)
    p.add_argument("--elite_size", type=int, default=4)
    p.add_argument("--num_es_iterations", type=int, default=3)
    p.add_argument("--mask_goal", action="store_true", default=False)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--visualize_mode", default="none",
                   choices=["all", "success", "failure", "none"])
    p.add_argument("--save_trajectory", action="store_true", default=True)
    p.add_argument("--save_instruction", action="store_true", default=False)
    # dense-oracle weights (only used when --selection dense)
    p.add_argument("--w_collision", type=float, default=1.0)
    p.add_argument("--w_offroad", type=float, default=1.0)
    p.add_argument("--w_progress", type=float, default=1.0)
    p.add_argument("--w_closeness", type=float, default=0.5)
    p.add_argument("--w_separation", type=float, default=0.25)
    p.add_argument("--w_speed", type=float, default=0.0)
    p.add_argument("--w_comfort", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=1.0)
    # metric logging (trajectories stay on disk; only metrics go to wandb)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default="es-reward-resolution")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_dir", default=None,
                   help="wandb staging dir; keep on scratch for offline runs")
    return p.parse_args()


def _infer_tf_shard(tfrecord_dir: str, explicit: str | None) -> str | None:
    if explicit:
        return str(explicit).zfill(5) if str(explicit).isdigit() else str(explicit)
    import re
    m = re.search(r"tf(\d{5})", tfrecord_dir)
    return m.group(1) if m else None


def main() -> None:
    args = parse_args()
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    # arm label used in output dir + diagnostics filename
    arm = args.selection
    if args.init_select != "none":
        if args.init_select in ("sac", "sac_safe"):
            arm = f"{args.selection}_{args.init_select}"
        elif args.init_select in MIXED_INIT_CHOICES:
            arm = (f"{args.selection}_{args.init_select}"
                   f"_m{args.init_bank_multiplier}_f{args.sac_frac:g}")
        else:
            arm = f"{args.selection}_{args.init_select}_m{args.init_bank_multiplier}"
    exp_name = f"es_{arm}"
    if args.tag:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.exp_dir, exp_name)
    args.arm = arm

    sac_bank = None
    tf_shard = _infer_tf_shard(args.tfrecord_dir, args.tf_shard)
    if args.init_select in SAC_INIT_CHOICES:
        if not args.sac_bank_path or not args.sac_manifest:
            raise SystemExit(
                f"--sac_bank_path and --sac_manifest required for {args.init_select}")
        if tf_shard is None:
            raise SystemExit("Could not infer --tf_shard from tfrecord_dir; pass explicitly")
        sac_bank = load_sac_init_bank(args.sac_bank_path, args.sac_manifest)
        # Fail fast if any requested scene is missing from the bank.
        if args.scenario_indices:
            missing = []
            for idx in [int(x) for x in args.scenario_indices.split(",") if x != ""]:
                key = (tf_shard, idx)
                if key not in sac_bank.key_to_row:
                    missing.append(f"{tf_shard}/{idx}")
            if missing:
                raise SystemExit(
                    "SAC bank missing requested scenarios (no diffusion fallback): "
                    + ", ".join(missing)
                )
    args.tf_shard = tf_shard

    inference = load_model_for_inference(
        checkpoint_path=args.diffusion_ckpt_path, use_ema=args.use_ema,
        metadata_path=None, seed=args.seed, skip_checkpoint_load=False)
    policy = nnx.merge(inference.graphdef, inference.params_state, inference.nonparam_state)

    dense_cfg = DenseConfig(
        weights=DenseWeights(
            collision=args.w_collision, offroad=args.w_offroad,
            progress=args.w_progress, closeness=args.w_closeness,
            separation=args.w_separation, speed=args.w_speed, comfort=args.w_comfort),
        gamma=args.gamma,
        dt=float(inference.preprocess_cfg.model_dt),
    )
    planner = InstrumentedESPlanner(
        policy=policy, preprocess_cfg=inference.preprocess_cfg,
        population_size=args.population_size, resample_timesteps=args.resample_timesteps,
        elite_size=args.elite_size, num_iterations=args.num_es_iterations,
        num_worlds=args.num_worlds, metrics=["collision", "offroad"],
        selection_mode=args.selection, dense_cfg=dense_cfg,
        init_bank_multiplier=args.init_bank_multiplier,
        init_select=args.init_select,
        sac_bank=sac_bank,
        sac_frac=args.sac_frac,
        tf_shard=tf_shard)

    # simulation.runner expects these fields (normally set by run_simulation.py).
    args.planner = "es"
    args.use_subgoal = False

    logger = WandbLogger(args, enabled=bool(args.wandb))
    try:
        run(args, planner)
    finally:
        # dump instrumentation next to the run outputs even if the sim died, so a
        # crashed shard still yields the diagnostics for the steps it completed.
        diag_path = Path(args.output_dir) / f"diagnostics_{arm}.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump({"args": vars(args), "records": planner.diagnostics}, f, indent=2)
        print(f"wrote {len(planner.diagnostics)} diagnostic records -> {diag_path}")

        scene_indices = ([int(x) for x in args.scenario_indices.split(",") if x != ""]
                         if args.scenario_indices else [])
        logger.warn_on_bank_mismatch(planner.diagnostics, scene_indices)
        logger.log_diagnostics(planner.diagnostics, scene_indices, tf_shard)
        logger.log_run_summary(args.output_dir, planner.diagnostics, arm)
        logger.finish()


if __name__ == "__main__":
    main()
