from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


os.environ["PYTHONWARNINGS"] = "ignore"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goal-reaching evaluation with DiffusionPlanner."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/zfsauton/scratch/mineuih/waymax_rs/checkpoints/goal_timestep_cond/epoch_0300",
    )
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default="/zfsauton/datasets/womd/tf_example",
    )
    parser.add_argument(
        "--split", type=str, default="training", choices=["training", "validation"]
    )
    parser.add_argument("--exp_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/exp")
    parser.add_argument("--lane_graph_dir", type=str, default="/zfsauton/scratch/mineuih/waymax_rs/lane_graphs")
    parser.add_argument("--task", type=str, default="goal_reaching")
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--num_worlds", type=int, default=20)
    parser.add_argument("--num_scenarios", type=int, default=1000)
    parser.add_argument("--use_es", action="store_true", default=False)
    parser.add_argument("--population_size", type=int, default=128)
    parser.add_argument("--resample_timesteps", type=int, default=5)
    parser.add_argument("--elite_size", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--replan_interval_steps", type=int, default=5)
    parser.add_argument("--goal_threshold_m", type=float, default=2.0)
    parser.add_argument("--metrics", type=str, default="overlap,offroad")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=91)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--px_per_meter", type=float, default=4.0)
    parser.add_argument("--show_agent_id", action="store_true", default=False)
    parser.add_argument("--front_x", type=float, default=30.0)
    parser.add_argument("--back_x", type=float, default=30.0)
    parser.add_argument("--front_y", type=float, default=30.0)
    parser.add_argument("--back_y", type=float, default=30.0)
    parser.add_argument("--save_traj", action="store_true", default=False)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    args.output_dir = os.path.join(args.exp_dir, f"{args.task}_{args.split}")
    if args.use_es:
        args.output_dir += "_es"
        print("Using DiffusionESPlanner with config:")
        print(f"  population_size: {args.population_size}")
        print(f"  resample_timesteps: {args.resample_timesteps}")
        print(f"  elite_size: {args.elite_size}")
        print(f"  num_iterations: {args.num_iterations}")
    else:
        print("Using DiffusionPlanner without evolutionary search.")
        args.population_size = 1  # Override population size to 1 when not using ES
        args.elite_size = 1  # Override elite size to 1 when not using ES
        args.resample_timesteps = 0  # No resampling when not using ES
        args.num_iterations = 0  # Only one iteration when not using ES
    if args.task == "goal_reaching":
        from simulation_utils.goal_reaching import run
        run(args)
    else:
        from simulation_utils.lane_following import run
        run(args)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
