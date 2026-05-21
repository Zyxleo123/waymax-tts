from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# from model.diffusion_planner import (  # noqa: E402
#     DiffusionPlanner,
#     PlannerResult,
#     PlannerTiming,
# )
from model.diffusion_planner_v2 import (
    DiffusionPlanner,
    PlannerResult,
)
from train.utils.infer import (  # noqa: E402
    PredictionBatch,
    _apply_ego_replacements_to_expanded_state,
    load_model_for_inference,
    rollout_predicted_trajectories_with_metrics,
)
from train.preprocess import preprocess_simulator_state
from viz.render import _load_scenario_state_fast, render_videos_batched  # noqa: E402
from viz import viz as viz_module  # noqa: E402
import time


def _parse_int_csv(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DiffusionPlanner on Waymax scenarios, evaluate metrics, and render videos."
    )
    parser.add_argument("--checkpoint_path", type=str, default="/data/user_data/mineuih/checkpoints/diffusion_lr-0p0001_20260301_215223/epoch_0420")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--tfrecord_path", type=str, default="/data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example/training/training_tfexample.tfrecord-00100-of-01000")
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_es_output")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--skip_checkpoint_load", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--resample_timesteps", type=int, default=5)
    parser.add_argument("--replan_interval_steps", type=int, default=5)
    parser.add_argument("--speed_limit_mps", type=float, default=None)
    parser.add_argument("--render_sample_indices", type=str, default="0")
    parser.add_argument("--rollout_sample_indices", type=str, default=None)
    parser.add_argument("--rollout_num_steps", type=int, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default="overlap,offroad,sdc_progression",
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
    return parser.parse_args()


def _summarize_metric(value_bkt: np.ndarray, valid_bkt: np.ndarray) -> dict[str, Any]:
    batch_size, num_samples, _ = value_bkt.shape
    per_entry: list[list[dict[str, Any]]] = []
    for b in range(batch_size):
        scenario_entries: list[dict[str, Any]] = []
        for k in range(num_samples):
            valid = valid_bkt[b, k].astype(bool)
            if np.any(valid):
                vals = value_bkt[b, k][valid]
                final = float(vals[-1])
                mean = float(vals.mean())
                max_value = float(vals.max())
                valid_steps = int(valid.sum())
            else:
                final = float("nan")
                mean = float("nan")
                max_value = float("nan")
                valid_steps = 0
            scenario_entries.append(
                {
                    "final": final,
                    "mean": mean,
                    "max": max_value,
                    "valid_steps": valid_steps,
                }
            )
        per_entry.append(scenario_entries)
    return {"per_scenario_sample": per_entry}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _planner_result_to_prediction_batch(result: PlannerResult) -> PredictionBatch:
    return PredictionBatch(
        start_t_b=result.start_t_b,
        trajectories_world_bkt5=result.trajectory_world_bt5[:, None, :, :],
        world_t_seconds_bk=result.world_t_seconds_bt,
        world_t_valid_bk=result.world_t_valid_bt,
        aux=result.aux,
    )


def _save_population_xy_score_image(
    *,
    sim_state,
    timestep: int,
    current_world_bkt5: np.ndarray,
    current_scores_bk: np.ndarray,
    output_path: Path,
    title: str,
    front_x: float,
    back_x: float,
    front_y: float,
    back_y: float,
    current_lanes: list[np.ndarray] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    world_bkt5 = np.asarray(current_world_bkt5, dtype=np.float32)
    scores_bk = np.asarray(current_scores_bk, dtype=np.float32)
    if world_bkt5.ndim != 4:
        raise ValueError(f"current_world_bkt5 must be rank-4 [B, K, T, 5], got {world_bkt5.shape}")
    if scores_bk.ndim != 2:
        raise ValueError(f"current_scores_bk must be rank-2 [B, K], got {scores_bk.shape}")

    batch_size = int(world_bkt5.shape[0])
    if int(scores_bk.shape[0]) != batch_size or int(scores_bk.shape[1]) != int(world_bkt5.shape[1]):
        raise ValueError(
            "current_world_bkt5 and current_scores_bk shapes are inconsistent: "
            f"{world_bkt5.shape} vs {scores_bk.shape}"
        )

    ncols = batch_size
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 6.0), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for b in range(batch_size):
        ax = axes[0, b]
        state_b = viz_module._index_pytree(sim_state, b)
        t = int(np.clip(timestep, 0, int(np.asarray(state_b.log_trajectory.x).shape[-1]) - 1))

        viz_module.plot_roadgraph_points(ax, state_b.roadgraph_points, verbose=False)

        if current_lanes is not None and b < len(current_lanes):
            lane_points = np.asarray(current_lanes[b], dtype=np.float32)
            if lane_points.ndim == 2 and lane_points.shape[0] >= 2 and lane_points.shape[1] >= 2:
                ax.plot(
                    lane_points[:, 0],
                    lane_points[:, 1],
                    color="deepskyblue",
                    linewidth=3.0,
                    alpha=0.95,
                    zorder=5,
                )

        is_ego = np.asarray(state_b.object_metadata.is_sdc).astype(bool)
        num_obj = int(state_b.log_trajectory.num_objects)
        is_controlled = np.zeros((num_obj,), dtype=bool)
        viz_module.plot_trajectory(
            ax,
            state_b.log_trajectory,
            is_controlled=is_controlled,
            time_idx=t,
            indices=None,
            past_traj_length=0,
            is_ego=is_ego,
        )

        traj_kt2 = world_bkt5[b, :, :, :2]
        score_k = scores_bk[b]
        for k in range(int(traj_kt2.shape[0])):
            xy_t2 = traj_kt2[k]
            ax.plot(
                xy_t2[:, 0],
                xy_t2[:, 1],
                color=cmap(float(score_k[k])),
                linewidth=1.2,
                alpha=0.95,
            )
            ax.scatter(
                xy_t2[0, 0],
                xy_t2[0, 1],
                color=cmap(float(score_k[k])),
                s=8,
                alpha=0.9,
            )

        current_xy = np.asarray(state_b.log_trajectory.xy[:, t, :])
        if np.any(is_ego):
            center_xy = current_xy[is_ego][0]
        else:
            center_xy = np.nanmean(current_xy, axis=0)
        ax.axis(
            (
                float(center_xy[0]) - float(back_x),
                float(center_xy[0]) + float(front_x),
                float(center_xy[1]) - float(back_y),
                float(center_xy[1]) + float(front_y),
            )
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"batch={b}, t={t}, K={traj_kt2.shape[0]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="score")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _predict_planner_trajectories_with_periodic_replan(
    sim_state,
    planner: DiffusionPlanner,
    *,
    debug_label: str,
    output_dir: Path,
    scenario_index: int,
    rng_key: jax.Array,
    elite_size: int,
    num_iterations: int,
    replan_interval_steps: int,
    front_x: float,
    back_x: float,
    front_y: float,
    back_y: float,
) -> tuple[PredictionBatch, np.ndarray, list[dict[str, Any]]]:
    if int(replan_interval_steps) <= 0:
        raise ValueError(f"replan_interval_steps must be > 0, got {replan_interval_steps}.")

    episode_num_steps = int(sim_state.log_trajectory.x.shape[-1])
    start_t_b1 = np.zeros((1,), dtype=np.int32)
    planner_debug: list[dict[str, Any]] = []

    rng_key, key_initial = jax.random.split(rng_key)
    init_pred_start = time.time()

    initial_current_lanes: list[np.ndarray] = []
    for world_idx in range(int(sim_state.log_trajectory.x.shape[0])):
        planner.scorers[world_idx].update_lane_points(sim_state, world_idx=world_idx)
        # planner.scorers[world_idx].update_objects(sim_state, timestep=0, world_idx=world_idx)
        lane_pts = planner.scorers[world_idx].get_right_lane(sim_state, timestep=0, world_idx=world_idx)
        if lane_pts is None:
            lane_pts = planner.scorers[world_idx].get_left_lane(sim_state, timestep=0, world_idx=world_idx)
        if lane_pts is None:
            lane_pts = planner.scorers[world_idx].get_current_lane(sim_state, timestep=0, world_idx=world_idx)
        initial_current_lanes.append(np.asarray(lane_pts, dtype=np.float32))
        planner.scorers[world_idx].add_metric("follow_lane", target=lane_pts)
        vehicle_front = planner.scorers[world_idx].get_vehicle_front(sim_state, timestep=0, world_idx=world_idx)
        print(vehicle_front)
    if vehicle_front is None:
        return None, None, None, None

    initial_result = planner.plan_trajectory(
        sim_state,
        rng=key_initial,
        elite_size=int(elite_size),
        num_iterations=int(num_iterations),
        timestep=0,
    )
    _save_population_xy_score_image(
        sim_state=sim_state,
        timestep=0,
        current_world_bkt5=np.asarray(initial_result.current_world_bkt5),
        current_scores_bk=np.asarray(initial_result.current_scores_bk),
        output_path=output_dir / "population_xy_scores" / f"scenario_{int(scenario_index):05d}" / f"step_{0:04d}.png",
        title=f"{debug_label} step=0 population xy (color=score)",
        front_x=float(front_x),
        back_x=float(back_x),
        front_y=float(front_y),
        back_y=float(back_y),
        current_lanes=initial_current_lanes,
    )
    init_pred_end = time.time()
    # print("initial_plan_time", init_pred_end - init_pred_start)

    initial_pred = _planner_result_to_prediction_batch(initial_result)
    initial_traj_bl5 = np.asarray(initial_result.trajectory_world_bt5, dtype=np.float32)
    model_horizon_len = int(initial_traj_bl5.shape[1])

    traj_b1l5 = np.zeros((1, episode_num_steps, 5), dtype=np.float32)
    init_copy_len = min(episode_num_steps, model_horizon_len)
    traj_b1l5[:, :init_copy_len, :] = initial_traj_bl5[:, :init_copy_len, :]
    current_score_b = np.asarray(initial_result.best_score_b, dtype=np.float32)
    planner_debug.append(
        {
            "step_offset": 0,
            "best_score_b": current_score_b.tolist(),
        }
    )

    max_steps = episode_num_steps - 1
    interval = int(replan_interval_steps)
    for step_offset in range(interval, max_steps, interval):
        for world_idx in range(int(sim_state.log_trajectory.x.shape[0])):
            if planner.scorers[world_idx].on_target_lane(sim_state, timestep=step_offset, world_idx=world_idx):
                current_speed = planner.scorers[world_idx].get_current_speed(sim_state, timestep=step_offset, world_idx=world_idx)
                target_speed = 1.5 * float(current_speed)
                planner.scorers[world_idx].add_metric("set_speed", target=target_speed)

        replaced_state = _apply_ego_replacements_to_expanded_state(
            sim_state,
            start_t_bk=start_t_b1,
            traj_bkl5=traj_b1l5,
        )
        rng_key, key_replan = jax.random.split(rng_key)
        repl_start = time.time()

        repl_result = planner.plan_trajectory(
            replaced_state,
            rng=key_replan,
            elite_size=int(elite_size),
            num_iterations=int(num_iterations),
            timestep=int(step_offset),
        )
        _save_population_xy_score_image(
            sim_state=replaced_state,
            timestep=int(step_offset),
            current_world_bkt5=np.asarray(repl_result.current_world_bkt5),
            current_scores_bk=np.asarray(repl_result.current_scores_bk),
            output_path=output_dir / "population_xy_scores" / f"scenario_{int(scenario_index):05d}" / f"step_{int(step_offset):04d}.png",
            title=f"{debug_label} step={int(step_offset)} population xy (color=score)",
            front_x=float(front_x),
            back_x=float(back_x),
            front_y=float(front_y),
            back_y=float(back_y),
            current_lanes=initial_current_lanes,
        )
        repl_end = time.time()
        # print("replan_time",
        #     f"step_offset={int(step_offset)}",
        #     f"time={repl_end - repl_start:.3f}s",
        # )

        repl_score_b = np.asarray(repl_result.best_score_b, dtype=np.float32)
        repl_traj_b1l5 = np.asarray(repl_result.trajectory_world_bt5, dtype=np.float32)

        remaining = episode_num_steps - int(step_offset)
        if remaining > 0:
            copy_len = min(remaining, int(repl_traj_b1l5.shape[1]))
            traj_b1l5[:, step_offset : step_offset + copy_len, :] = repl_traj_b1l5[:, :copy_len, :]

        current_score_b = repl_score_b
        planner_debug.append(
            {
                "step_offset": int(step_offset),
                "best_score_b": repl_score_b.tolist()
            }
        )

    world_dt = float(np.asarray(initial_pred.aux["world_dt_seconds"], dtype=np.float32).reshape(-1)[0])
    world_t = np.arange(episode_num_steps, dtype=np.float32)[None, :] * world_dt
    world_valid = np.ones((1, episode_num_steps), dtype=bool)

    pred = PredictionBatch(
        start_t_b=jnp.zeros((1,), dtype=jnp.int32),
        trajectories_world_bkt5=jnp.asarray(traj_b1l5[:, None, :, :], dtype=jnp.float32),
        world_t_seconds_bk=jnp.asarray(world_t, dtype=jnp.float32),
        world_t_valid_bk=jnp.asarray(world_valid),
        aux=initial_pred.aux,
    )
    return pred, current_score_b[:, None], planner_debug, vehicle_front


def main() -> None:
    args = _parse_args()

    if int(args.num_episodes) < 1:
        raise ValueError(f"num_episodes must be >= 1, got {args.num_episodes}.")

    render_sample_indices = _parse_int_csv(args.render_sample_indices)
    rollout_sample_indices = (
        _parse_int_csv(args.rollout_sample_indices)
        if args.rollout_sample_indices is not None
        else render_sample_indices
    )
    metric_names = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    planner = DiffusionPlanner(
        policy,
        inference.preprocess_cfg,
        population_size=int(args.population_size),
        resample_timesteps=int(args.resample_timesteps),
    )

    from waymax import config as waymax_config

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(args.tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(1,),
        shuffle_seed=0,
    )

    processed_scenario_indices: list[int] = []
    episode_summaries: list[dict[str, Any]] = []
    all_predicted_videos: list[dict[str, Any]] = []
    all_original_videos: list[str] = []

    npz_start_t: list[np.ndarray] = []
    npz_traj: list[np.ndarray] = []
    npz_world_t_seconds: list[np.ndarray] = []
    npz_world_t_valid: list[np.ndarray] = []
    npz_final_scores: list[np.ndarray] = []
    npz_best_sample_index: list[np.ndarray] = []

    base_rng_key = jax.random.PRNGKey(int(args.seed))

    for scenario_index in range(int(args.num_episodes)):
        start_from = 10
        source_scenario_index = start_from + scenario_index
        try:
            sim_state = _load_scenario_state_fast(ds_cfg, int(source_scenario_index))
        except ValueError as exc:
            if "is out of range" in str(exc):
                break
            raise

        rng_key = jax.random.fold_in(base_rng_key, int(scenario_index))
        pred, final_scores, planner_debug, target_idx = _predict_planner_trajectories_with_periodic_replan(
            sim_state,
            planner,
            debug_label=f"scenario={scenario_index}",
            output_dir=output_dir,
            scenario_index=int(scenario_index),
            rng_key=rng_key,
            elite_size=int(args.elite_size),
            num_iterations=int(args.num_iterations),
            replan_interval_steps=int(args.replan_interval_steps),
            front_x=float(args.front_x),
            back_x=float(args.back_x),
            front_y=float(args.front_y),
            back_y=float(args.back_y),
        )
        if target_idx is None:
            print(f"Skipping scenario {scenario_index} due to missing vehicle front.")
            continue

        pred_traj = np.asarray(pred.trajectories_world_bkt5)
        start_t = np.asarray(pred.start_t_b, dtype=np.int32)
        best_sample_index = np.zeros((pred_traj.shape[0],), dtype=np.int32)
        best_score = np.asarray(final_scores[:, 0], dtype=np.float32)

        if np.any(np.asarray(render_sample_indices) >= pred_traj.shape[1]):
            raise IndexError(
                f"render_sample_indices out of bounds for population_size={pred_traj.shape[1]}: {render_sample_indices}"
            )
        if np.any(np.asarray(rollout_sample_indices) >= pred_traj.shape[1]):
            raise IndexError(
                f"rollout_sample_indices out of bounds for population_size={pred_traj.shape[1]}: {rollout_sample_indices}"
            )

        rollout = rollout_predicted_trajectories_with_metrics(
            sim_state,
            pred,
            metric_names=metric_names,
            rollout_num_steps=args.rollout_num_steps,
            sample_indices=rollout_sample_indices,
            rng_key=rng_key,
        )

        video_requests = [
            (str(args.tfrecord_path), int(source_scenario_index)) for _ in render_sample_indices
        ]
        ego_start_times = [int(start_t[0]) for _ in render_sample_indices]
        ego_trajectories = [
            pred_traj[0, int(sample_index)] for sample_index in render_sample_indices
        ]
        video_manifest = [
            {
                "scenario_index": int(scenario_index),
                "source_scenario_index": int(source_scenario_index),
                "batch_index": 0,
                "sample_index": int(sample_index),
                "start_t": int(start_t[0]),
                "final_score": float(final_scores[0, int(sample_index)]),
                "best_sample_index": int(best_sample_index[0]),
                "best_score": float(best_score[0]),
            }
            for sample_index in render_sample_indices
        ]

        predicted_video_paths = render_videos_batched(
            tfrecord_scenarios=video_requests,
            output_dir=(output_dir / "predicted_videos").as_posix(),
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
            target_idx=target_idx,
        )

        original_video_paths: list[str] = []
        if args.render_original:
            original_video_paths = [
                p.as_posix()
                for p in render_videos_batched(
                    tfrecord_scenarios=[(str(args.tfrecord_path), int(source_scenario_index))],
                    output_dir=(output_dir / "original_videos").as_posix(),
                    fps=int(args.fps),
                    num_frames=int(args.num_frames),
                    max_num_objects=int(args.max_num_objects),
                    width=int(args.width),
                    height=int(args.height),
                    px_per_meter=float(args.px_per_meter),
                    show_agent_id=bool(args.show_agent_id),
                    use_log_traj=True,
                    front_x=float(args.front_x),
                    back_x=float(args.back_x),
                    front_y=float(args.front_y),
                    back_y=float(args.back_y),
                )
            ]

        metric_summary = {
            metric_name: _summarize_metric(
                np.asarray(rollout.metric_timeseries[metric_name]),
                np.asarray(rollout.metric_valid[metric_name]),
            )
            for metric_name in rollout.metric_names
        }

        episode_predicted_videos = [
            {
                **video_manifest[i],
                "path": predicted_video_paths[i].as_posix(),
            }
            for i in range(len(predicted_video_paths))
        ]
        episode_summary = {
            "scenario_index": int(scenario_index),
            "source_scenario_index": int(source_scenario_index),
            "predict_horizon_world_steps": int(pred_traj.shape[2]),
            "start_t": int(start_t[0]),
            "best_sample_index": int(best_sample_index[0]),
            "best_score": float(best_score[0]),
            "metric_names": list(rollout.metric_names),
            "metric_summary": metric_summary,
            "planner_replan_debug": planner_debug,
            "predicted_videos": episode_predicted_videos,
            "original_videos": original_video_paths,
        }

        processed_scenario_indices.append(int(scenario_index))
        episode_summaries.append(episode_summary)
        all_predicted_videos.extend(episode_predicted_videos)
        all_original_videos.extend(original_video_paths)

        if args.save_predictions_npz:
            npz_start_t.append(start_t.copy())
            npz_traj.append(pred_traj.copy())
            npz_world_t_seconds.append(np.asarray(pred.world_t_seconds_bk).copy())
            npz_world_t_valid.append(np.asarray(pred.world_t_valid_bk).copy())
            npz_final_scores.append(final_scores.copy())
            npz_best_sample_index.append(best_sample_index.copy())

        # print(
        #     "processed_episode",
        #     f"scenario={scenario_index}",
        #     f"best_sample={int(best_sample_index[0])}",
        #     f"best_score={float(best_score[0]):.6f}",
        # )
        # for item in episode_predicted_videos:
        #     print(
        #         "predicted_video",
        #         f"scenario={item['scenario_index']}",
        #         f"sample={item['sample_index']}",
        #         f"score={item['final_score']:.6f}",
        #         f"path={item['path']}",
        #     )
        for path in original_video_paths:
            print("original_video", f"path={path}")

    if not episode_summaries:
        raise ValueError(
            f"No scenarios were processed from tfrecord '{args.tfrecord_path}'."
        )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "metadata_path": args.metadata_path,
        "tfrecord_path": str(args.tfrecord_path),
        "num_episodes_requested": int(args.num_episodes),
        "num_episodes_processed": len(episode_summaries),
        "scenario_indices": processed_scenario_indices,
        "render_sample_indices": [int(x) for x in render_sample_indices],
        "rollout_sample_indices": [int(x) for x in rollout_sample_indices],
        "population_size": int(args.population_size),
        "elite_size": int(args.elite_size),
        "num_iterations": int(args.num_iterations),
        "resample_timesteps": int(args.resample_timesteps),
        "replan_interval_steps": int(args.replan_interval_steps),
        "episodes": episode_summaries,
        "predicted_videos": all_predicted_videos,
        "original_videos": all_original_videos,
    }
    _save_json(output_dir / "summary.json", summary)

    if args.save_predictions_npz:
        np.savez_compressed(
            output_dir / "predictions.npz",
            scenario_indices=np.asarray(processed_scenario_indices, dtype=np.int32),
            start_t=np.concatenate(npz_start_t, axis=0),
            trajectories_world_bkt5=np.concatenate(npz_traj, axis=0),
            world_t_seconds_bk=np.concatenate(npz_world_t_seconds, axis=0),
            world_t_valid_bk=np.concatenate(npz_world_t_valid, axis=0),
            best_score_b=np.concatenate(npz_final_scores, axis=0)[:, 0],
            best_sample_index=np.concatenate(npz_best_sample_index, axis=0),
        )

    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
