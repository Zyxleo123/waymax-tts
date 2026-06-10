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

from train_diffusion.utils.infer import (
    load_model_for_inference,
)
from simulation.runner import run

os.environ["PYTHONWARNINGS"] = "ignore"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_ckpt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vla_ckpt_path",
        type=str,
        default=None,
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
    parser.add_argument("--planner", type=str, default="diffusion", choices=["diffusion", "vla"])
    parser.add_argument("--start_timestep", type=int, default=10)
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=10)
    parser.add_argument("--num_scenarios", type=int, default=1000)
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--visualize_mode", type=str, default="all", choices=["all", "success", "failure", "none"])
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--save_instruction", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--scenario_indices", type=str, default=None)
    parser.add_argument("--dummy_instruction", action="store_true", default=False)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    exp_name = f"simulation_{args.planner}_{args.split}"
    if args.tag is not None:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.exp_dir, exp_name)
    
    from planner.diffusion_planner import DiffusionPlanner
    inference = load_model_for_inference(
        checkpoint_path=args.diffusion_ckpt_path,
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
    if args.planner == "diffusion":
        planner = DiffusionPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
        )
    elif args.planner == "vla":
        from model.vla.gemma_vla import VecSceneGemmaVLA
        from model.vla.embedding_gemma_encoder import EmbeddingGemmaEncoder
        from planner.vla_planner import VLAPlanner

        gemma_vla = VecSceneGemmaVLA.from_pretrained(args.vla_ckpt_path)
        instruction_encoder = EmbeddingGemmaEncoder(
            model_name="google/embeddinggemma-300m",
            device="cuda:1",
            max_length=128,
            target_dim=786,
        )
        planner = VLAPlanner(
            policy=policy,
            gemma_vla=gemma_vla,
            instruction_encoder=instruction_encoder,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
            dummy_instruction=args.dummy_instruction,
        )
    run(args, planner)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
