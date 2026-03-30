from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from simulation_utils.reward_search import run, run_with_improvement
from simulation_utils.reward_search_candidates import run_with_improvement_from_candidates
import time
import os

os.environ["PYTHONWARNINGS"] = "ignore"




def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DiffusionPlanner on Waymax scenarios, evaluate metrics, and render videos."
    )
    parser.add_argument("--checkpoint_path", type=str, default="/data/user_data/mineuih/checkpoints/diffusion_lr-0p0001_20260301_215223/epoch_0420")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--tfrecord_path", type=str, default="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example")
    parser.add_argument("--output_dir", type=str, default="/data/user_data/mineuih/waymax_rs")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=16)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--resample_timesteps", type=int, default=3)
    parser.add_argument("--replan_interval_steps", type=int, default=5)
    parser.add_argument("--speed_limit_mps", type=float, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="overlap,offroad",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=91)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--px_per_meter", type=float, default=4.0)
    parser.add_argument("--show_agent_id", action="store_true", default=True)
    parser.add_argument("--front_x", type=float, default=30.0)
    parser.add_argument("--back_x", type=float, default=30.0)
    parser.add_argument("--front_y", type=float, default=30.0)
    parser.add_argument("--back_y", type=float, default=30.0)
    parser.add_argument("--render_original", action="store_true")
    parser.add_argument("--save_predictions_npz", action="store_true")
    parser.add_argument("--task", type=str, default="overtake")
    parser.add_argument("--num_worlds", type=int, default=1)
    parser.add_argument("--use_candidates", action="store_true", default=False)
    parser.add_argument("--use_baseline", action="store_true", default=False)
    parser.add_argument("--job", type=str, default="reward_search")
    parser.add_argument("--rs_iterations", type=int, default=10)
    return parser.parse_args()

def main(args: argparse.Namespace):
    if args.task == "follow_lane":
        code = """
done = self.follow_lane(self.current_lane)
while not done():
    yield
        """
    elif args.task == "change_lane_left":
        code = """
done = self.follow_lane(self.left_lane)
while not done():
    yield
        """
    elif args.task == "change_lane_right":
        code = """
done = self.follow_lane(self.right_lane)
while not done():
    yield
        """
    elif args.task == "overtake":
        if args.job == "test" or args.job == "test_finetuned":
            code = """
target_vehicle_idx = None
while target_vehicle_idx is None:
    target_vehicle_idx = self.get_vehicle_front()
    yield
self.overtake(target_vehicle_idx)
"""
        else:
            code = """
target_vehicle_idx = self.get_vehicle_front()
if target_vehicle_idx is None:
    while True:
        yield
else:
    if self.left_lane is not None:
        done = self.follow_lane(self.left_lane)
        while not done():
            yield
        target_speed = self.get_current_speed(target_vehicle_idx) + 5
        self.set_speed(target_speed)
        while not self.is_ahead_of(target_vehicle_idx):
            yield
        done = self.follow_lane(self.right_lane)
        while not done():
            yield
    elif self.right_lane is not None:
        done = self.follow_lane(self.right_lane)
        while not done():
            yield
        target_speed = self.get_current_speed(target_vehicle_idx) + 5
        self.set_speed(target_speed)
        while not self.is_ahead_of(target_vehicle_idx):
            yield
        done = self.follow_lane(self.left_lane)
        while not done():
            yield
"""
    elif args.task == "give_way":
        code = """
target_vehicle_idx = self.get_vehicle_behind()
if target_vehicle_idx is None:
    while True:
        yield
else:
    if self.right_lane is not None:
        done = self.follow_lane(self.right_lane)
        while not done():
            yield
        target_speed = self.get_current_speed(target_vehicle_idx) * 0.5
        self.set_speed(target_speed)
        while not self.is_behind_of(target_vehicle_idx):
            yield
        done = self.follow_lane(self.left_lane)
        while not done():
            yield
    elif self.left_lane is not None:
        done = self.follow_lane(self.left_lane)
        while not done():
            yield
        target_speed = self.get_current_speed(target_vehicle_idx) * 0.5
        self.set_speed(target_speed)
        while not self.is_behind_of(target_vehicle_idx):
            yield
        done = self.follow_lane(self.right_lane)
        while not done():
            yield
"""
    elif args.task == "pull_over":
        code = """
while self.right_lane is not None:
    done = self.follow_lane(self.right_lane)
    while not done():
        yield
done = self.move_to_roadside()
while not done():
    yield
self.stop()
"""
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    args.output_dir = args.output_dir + f"/{args.task}"
    if args.use_candidates:
        args.output_dir = args.output_dir + f"/{args.job}_baseline" if args.use_baseline else args.output_dir + f"/{args.job}"
        candidate_scenarios = json.load(open("/data/user_data/mineuih/waymax_rs/overtake_scenarios.json", "r"))
        candidate_scenarios = {file_name: scenario_indices for file_name, scenario_indices in candidate_scenarios.items() if int(file_name.split("-")[1]) in [0, 1, 2, 3]}
        print(candidate_scenarios.keys())
        run_with_improvement_from_candidates(args, code, candidate_scenarios, use_baseline=args.use_baseline)
    else:
        run_with_improvement(args, code)

if __name__ == "__main__":
    args = _parse_args()
    main(args)
