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
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffusion_ckpt_path",
        type=str,
        default=None,
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
    parser.add_argument("--planner", type=str, default="diffusion",
                         choices=["diffusion", "es", "vla", "log-replay", "temporal-vla"])
    parser.add_argument("--start_timestep", type=int, default=10)
    parser.add_argument("--mask_goal", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=128)
    parser.add_argument("--num_worlds", type=int, default=5)
    parser.add_argument("--num_scenarios", type=int, default=10000)
    parser.add_argument("--replan_interval_steps", type=int, default=10)
    parser.add_argument("--instruction_interval_steps", type=int, default=10)
    parser.add_argument("--visualize_mode", type=str, default="all", choices=["all", "success", "failure", "none"])
    parser.add_argument("--save_trajectory", action="store_true", default=False)
    parser.add_argument("--save_instruction", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--scenario_indices", type=str, default=None)
    parser.add_argument("--dummy_instruction", action="store_true", default=False)
    parser.add_argument("--use_subgoal", action="store_true", default=False)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--resample_timesteps", type=int, default=3)
    parser.add_argument("--elite_size", type=int, default=4)
    parser.add_argument("--num_es_iterations", type=int, default=3)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.tfrecord_dir = os.path.join(args.tfrecord_dir, args.split)
    exp_name = f"simulation_{args.planner}_{args.split}"
    if args.tag is not None:
        exp_name += f"_{args.tag}"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.join(args.exp_dir, exp_name)
    
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
        from planner.diffusion_planner import DiffusionPlanner
        planner = DiffusionPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
        )
    elif args.planner == "es":
        from planner.diffusion_es_planner import DiffusionESPlanner
        planner = DiffusionESPlanner(
            policy=policy,
            preprocess_cfg=inference.preprocess_cfg,
            population_size=args.population_size,
            resample_timesteps=args.resample_timesteps,
            elite_size=args.elite_size,
            num_iterations=args.num_es_iterations,
            num_worlds=args.num_worlds,
            metrics=["collision", "offroad"],
        )
    elif args.planner == "vla":
        # from model.vla.gemma_vla import VecSceneGemmaVLA
        from model.vla.gemma_vla_v2 import VecSceneGemmaVLA
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
    elif args.planner == "temporal-vla":
        from model.vla.temporal_vla import TemporalGemmaVLA
        from model.vla.embedding_gemma_encoder import EmbeddingGemmaEncoder
        from planner.temporal_vla_planner import TemporalVLAPlanner

        temporal_vla = TemporalGemmaVLA.from_pretrained(args.vla_ckpt_path)
        instruction_encoder = EmbeddingGemmaEncoder(
            model_name="google/embeddinggemma-300m",
            device="cuda:1",
            max_length=128,
            target_dim=786,
        )
        planner = TemporalVLAPlanner(
            policy=policy,
            temporal_vla=temporal_vla,
            instruction_encoder=instruction_encoder,
            preprocess_cfg=inference.preprocess_cfg,
            num_worlds=args.num_worlds,
            dummy_instruction=args.dummy_instruction,
            population_size=args.population_size,
            resample_timesteps=args.resample_timesteps,
            elite_size=args.elite_size,
            num_iterations=args.num_es_iterations,
        )
    elif args.planner == "log-replay":
        planner = None
    run(args, planner, log_replay=(args.planner == "log-replay"))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
