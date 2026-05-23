from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from flax import nnx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.utils.infer import (
    load_model_for_inference,
)
from simulation.runner import run

os.environ["PYTHONWARNINGS"] = "ignore"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_20260517_044407/latest/",
    )
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default="/zfsauton/scratch/eshau/womd/tf_example/",
    )
    parser.add_argument(
        "--split", type=str, default="training", choices=["training", "validation"]
    )
    parser.add_argument(
        "--lane_graph_dir",
        type=str, 
        default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs"
    )
    parser.add_argument(
        "--exp_dir",
        type=str, 
        default="/zfsauton/scratch/mineuih/waymax_rs/exp"
    )
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=10)
    parser.add_argument("--num_scenarios", type=int, default=1000)
    parser.add_argument("--replan_interval_steps", type=int, default=5)
    parser.add_argument("--visualize_mode", type=str, default="all", choices=["all", "success", "failure", "none"])
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    exp_name = f"simulation_{args.split}"
    if args.tag is not None:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.exp_dir, exp_name)
    
    
    from planner.diffusion_planner import DiffusionPlanner
    inference = load_model_for_inference(
        checkpoint_path=args.checkpoint_path,
        use_ema=args.use_ema,
        metadata_path=None,
        seed=args.seed,
        skip_checkpoint_load=False,
    )
    policy = nnx.merge(
        inference.graphdef,
        inference.params_state,
        inference.nonparam_state,
    )
    planner = DiffusionPlanner(
        policy=policy,
        preprocess_cfg=inference.preprocess_cfg,
        num_worlds=args.num_worlds,
    )
    run(args, planner)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
