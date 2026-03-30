from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from flax import nnx
from tqdm import tqdm



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# from model.diffusion_planner import (  # noqa: E402
#     DiffusionPlanner,
#     PlannerResult,
#     PlannerTiming,
# )
# from model.diffusion_planner_v2 import (
#     DiffusionPlanner,
#     PlannerResult,
# )
from model.diffusion_language_planner import (  # noqa: E402
    DiffusionLanguagePlanner,
    PlannerResult,
)
from train.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from train.preprocess import preprocess_simulator_state
from viz.render import _load_scenario_state_fast, _load_scenario_state_batch_fast, render_videos_batched  # noqa: E402
from simulation_utils.visualize import _save_population_xy_score_image  # noqa: E402
from simulation_utils.utils import (
    _parse_int_csv,
    _summarize_metric,
    _save_json,
)
from simulation_utils.planner_utils import _predict_planner_trajectories_with_periodic_replan
from scores.scorer import Scorer  # noqa: E402
from gemini.query import query_gemini
from gemini.utils import task_to_command
import time
from waymax import config as waymax_config

timestamp = time.strftime("%Y%m%d_%H%M%S")

def get_file_name(dir_path, type, index):
    max_index = 150 if type == "validation" else 1000
    file_name = f"{dir_path}/{type}/{type}_tfexample.tfrecord-{index:05d}-of-{max_index:05d}"
    return file_name

def run(args, file_path, planner : DiffusionLanguagePlanner, scenario_indices, iteration, codes, commands, seed=0) -> None:
    if len(scenario_indices) == 0:
        return []
    metric_names = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    output_dir = Path(args.output_dir) / f"exp_{timestamp}" / f"iter{iteration:02d}"
    data_dir = Path(args.output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(file_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(len(scenario_indices),),
        shuffle_seed=0,
    )

    base_rng_key = jax.random.PRNGKey(int(args.seed))
    try:
        sim_state, _ = _load_scenario_state_batch_fast(ds_cfg, scenario_indices)
    except ValueError as exc:
        if "is out of range" in str(exc):
            return []
    rng_key = jax.random.fold_in(base_rng_key, seed)
    planner.reset(num_worlds=len(scenario_indices))
    target_vehicle_indices = []
    for world_idx in range(len(scenario_indices)):
        planner.world_pointer = world_idx
        planner.timestep = 0
        planner.sim_state = sim_state
        if args.task == "overtake":
            target_vehicle_idx = planner.get_vehicle_front()
        elif args.task == "give_way":
            target_vehicle_idx = planner.get_vehicle_behind()
        else:
            target_vehicle_idx = None
        target_vehicle_indices.append(target_vehicle_idx)

    if codes is not None:
        planner.set_codes(codes)
    if commands is None:
        commands = [task_to_command(args.task, target_vehicle=target_vehicle_indices[world_idx]) for world_idx in range(len(scenario_indices))]

    pred, final_scores, task_results = _predict_planner_trajectories_with_periodic_replan(
        sim_state,
        planner,
        task=args.task,
        target_vehicle_indices=target_vehicle_indices,
        output_dir=output_dir,
        scenario_indices=scenario_indices,
        rng_key=rng_key,
        elite_size=int(args.elite_size),
        num_iterations=int(args.num_iterations),
        replan_interval_steps=int(args.replan_interval_steps),
        front_x=float(args.front_x),
        back_x=float(args.back_x),
        front_y=float(args.front_y),
        back_y=float(args.back_y),
    )

    pred_traj = np.asarray(pred.trajectories_world_bkt5)
    start_t = np.asarray(pred.start_t_b, dtype=np.int32)

    rollout = rollout_predicted_trajectories_with_metrics(
        sim_state,
        pred,
        metric_names=metric_names,
        rng_key=rng_key,
    )

    overlap_timeseries = np.asarray(rollout.metric_timeseries["overlap"])
    offroad_timeseries = np.asarray(rollout.metric_timeseries["offroad"])
    overlap = (overlap_timeseries.sum(axis=(1, 2)) < 0)
    offroad = (offroad_timeseries.sum(axis=(1, 2)) < 0)
    if args.task == "overtake":
        task_results = task_results & ~overlap & ~offroad
    elif args.task == "give_way":
        task_results = task_results & ~overlap & ~offroad
    elif args.task == "pull_over":
        task_results = task_results & ~overlap
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print(f"Scenario indices: {list(scenario_indices)}, Task success: {task_results}, Overlap: {overlap}, Offroad: {offroad}")
    video_requests = [
        (str(file_path), int(source_scenario_index)) for source_scenario_index in scenario_indices
    ]
    ego_start_times = [int(start_t[i]) for i in range(len(scenario_indices))]
    ego_trajectories = [
        pred_traj[i, 0] for i in range(len(scenario_indices))
    ]

    predicted_video_paths = render_videos_batched(
        tfrecord_scenarios=video_requests,
        target_vehicles = target_vehicle_indices,
        output_dir=output_dir.as_posix(),
        fps=int(args.fps),
        num_frames=int(args.num_frames),
        max_num_objects=int(args.max_num_objects),
        width=int(args.width),
        height=int(args.height),
        px_per_meter=float(args.px_per_meter),
        show_agent_id=bool(args.show_agent_id),
        use_log_traj=True,
        ego_start_times=ego_start_times,
        ego_trajectories=ego_trajectories,
        front_x=float(args.front_x),
        back_x=float(args.back_x),
        front_y=float(args.front_y),
        back_y=float(args.back_y),
        align_ego_heading_up=True
    )

    for world_idx in range(len(scenario_indices)):
        if bool(task_results[world_idx]):
            data_path = data_dir / f"{Path(file_path).name}.scenario_{scenario_indices[world_idx]:04d}.npz"
            np.savez(
                data_path,
                trajectories_world_bkt5=pred_traj[world_idx],
                start_t_b=start_t[world_idx]
            )
    return [{
        'task': args.task,
        'ego_idx': planner.get_ego_idx(world_idx),
        'target_idx': target_vehicle_indices[world_idx],
        'command': commands[world_idx],
        'video_path': predicted_video_paths[world_idx].as_posix(),
        'tfrecord': file_path,
        'scenario_idx': scenario_indices[world_idx],
        'task_result': bool(task_results[world_idx]),
        'overlap': bool(overlap[world_idx]),
        'offroad': bool(offroad[world_idx]),
    } for world_idx in range(len(scenario_indices))]


def save_results(results):
    log_paths = []
    for result in results:
        log_path = result['video_path'].replace(".mp4", ".log.json")
        _save_json(Path(log_path), result)
        if not result['task_result']:
            log_paths.append(log_path)
    return log_paths

def get_codes(log_paths):
    if len(log_paths) == 0:
        return [], []
    else:
        asyncio.run(query_gemini(log_paths))
        llm_result_paths = [log_path.replace(".json", ".result.json") for log_path in log_paths]
        codes, commands = [], []
        for llm_result_path in llm_result_paths:
            if Path(llm_result_path).exists():
                with open(llm_result_path, "r") as f:
                    llm_result = json.load(f)
                codes.append(llm_result.get("actor_code", ""))
                commands.append(llm_result.get("critic_command", ""))
            else:
                codes.append("")
                commands.append("")
        return codes, commands


def run_with_improvement(
    args, baseline_code,
):
    from dotenv import load_dotenv
    load_dotenv()

    inference = load_model_for_inference(
        args.checkpoint_path,
        use_ema=bool(args.use_ema),
        metadata_path=args.metadata_path,
        seed=int(args.seed),
        skip_checkpoint_load=bool(args.skip_checkpoint_load),
    )
    policy = nnx.merge(
        inference.graphdef,
        inference.params_state,
        inference.nonparam_state,
    )
    planner = DiffusionLanguagePlanner(
        policy,
        inference.preprocess_cfg,
        population_size=int(args.population_size),
        resample_timesteps=int(args.resample_timesteps),
        num_worlds=int(args.num_worlds),
        code=baseline_code
    )

    file_idx = 0
    scenario_idx = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        scorer = Scorer()
        scenario_indices, checked_scenarios = [], 0
        
        while True:
            simulate = False
            if len(scenario_indices) == 0:
                pbar = tqdm(total=2 * int(args.num_worlds), desc="collecting candidate scenarios (0 / 0)")
                file_name = get_file_name(args.tfrecord_path, "validation", file_idx)
                check_cfg = dataclasses.replace(
                    waymax_config.WOD_1_3_1_TRAINING,
                    path=str(file_name),
                    max_num_objects=int(args.max_num_objects),
                    batch_dims=(1,),
                    shuffle_seed=0,
                )
            try:
                sim_state, _ = _load_scenario_state_batch_fast(check_cfg, [scenario_idx])
                if args.task == "overtake":
                    target_vehicle_idx = scorer.get_vehicle_front(sim_state, 0, 0)
                    if target_vehicle_idx is not None:
                        if scorer.get_left_lane(sim_state, 0, 0) is not None or scorer.get_right_lane(sim_state, 0, 0) is not None:
                            scenario_indices.append(scenario_idx)
                            pbar.update(1)
                elif args.task == "give_way":
                    target_vehicle_idx = scorer.get_vehicle_behind(sim_state, 0, 0)
                    if target_vehicle_idx is not None:
                        if scorer.get_left_lane(sim_state, 0, 0) is not None or scorer.get_right_lane(sim_state, 0, 0) is not None:
                            scenario_indices.append(scenario_idx)
                            pbar.update(1)
                else:
                    scenario_indices.append(scenario_idx)
                    pbar.update(1)
                scenario_idx += 1
                checked_scenarios += 1
                pbar.set_description(f"collecting candidate scenarios ({len(scenario_indices)} / {checked_scenarios})")
                if len(scenario_indices) == 2 * int(args.num_worlds):
                    simulate = True
                    
            except ValueError as exc:
                print(str(exc))
                simulate = True
                file_idx += 1
                scenario_idx = 0
                
            if simulate:
                pbar.close()
                scenario_indices_1 = scenario_indices[:len(scenario_indices) // 2]
                scenario_indices_2 = scenario_indices[len(scenario_indices) // 2:]

                results_1 = run(args, file_name, planner, scenario_indices_1, iteration=0, codes=[baseline_code] * len(scenario_indices_1), commands=None, seed=0)
                log_paths_1 = save_results(results_1)
                scenario_indices_1 = [scenario_indices_1[i] for i in range(len(results_1)) if not results_1[i]['task_result']]
                future_1 = executor.submit(get_codes, log_paths_1)

                results_2 = run(args, file_name, planner, scenario_indices_2, iteration=0, codes=[baseline_code] * len(scenario_indices_2), commands=None, seed=0)
                log_paths_2 = save_results(results_2)
                scenario_indices_2 = [scenario_indices_2[i] for i in range(len(results_2)) if not results_2[i]['task_result']]
                future_2 = executor.submit(get_codes, log_paths_2)

                for iteration in range(1, int(args.rs_iterations) + 1):
                    codes_1, commands_1 = future_1.result()
                    results_1 = run(args, file_name, planner, scenario_indices_1, iteration=iteration, codes=codes_1, commands=commands_1, seed=iteration)
                    log_paths_1 = save_results(results_1)
                    if iteration < int(args.rs_iterations):
                        scenario_indices_1 = [scenario_indices_1[i] for i in range(len(results_1)) if not results_1[i]['task_result']]
                        future_1 = executor.submit(get_codes, log_paths_1)

                    codes_2, commands_2 = future_2.result()
                    results_2 = run(args, file_name, planner, scenario_indices_2, iteration=iteration, codes=codes_2, commands=commands_2, seed=iteration)
                    log_paths_2 = save_results(results_2)
                    if iteration < int(args.rs_iterations):
                        scenario_indices_2 = [scenario_indices_2[i] for i in range(len(results_2)) if not results_2[i]['task_result']]
                        future_2 = executor.submit(get_codes, log_paths_2)

                pbar = tqdm(total=2 * int(args.num_worlds), desc="collecting candidate scenarios")
                scenario_indices, checked_scenarios = [], 0