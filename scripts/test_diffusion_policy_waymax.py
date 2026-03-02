from __future__ import annotations

import dataclasses
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.infer import (  # noqa: E402
    load_model_for_inference,
    predict_replacement_trajectories_for_batch,
    rollout_predicted_trajectories_with_metrics,
)
from viz.render import _load_scenario_state_batch_fast, render_videos_batched  # noqa: E402


def _parse_int_csv(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained diffusion policy checkpoint, test on Waymax scenarios, and render videos."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--tfrecord_path", type=str, required=True)
    parser.add_argument("--scenario_indices", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_objects", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=4)
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
    parser.add_argument("--show_agent_id", action="store_true")
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


def main() -> None:
    args = _parse_args()

    scenario_indices = _parse_int_csv(args.scenario_indices)
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
    )

    from waymax import config as waymax_config

    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(args.tfrecord_path),
        max_num_objects=int(args.max_num_objects),
        batch_dims=(len(scenario_indices),),
        shuffle_seed=0,
    )
    sim_state, scenario_to_batch_idx = _load_scenario_state_batch_fast(ds_cfg, scenario_indices)
    ordered_batch_indices = [scenario_to_batch_idx[idx] for idx in scenario_indices]

    rng_key = jax.random.PRNGKey(int(args.seed))
    pred = predict_replacement_trajectories_for_batch(
        sim_state,
        inference,
        rng_key=rng_key,
        num_samples=int(args.num_samples),
    )

    pred_traj = np.asarray(pred.trajectories_world_bkt5)
    start_t = np.asarray(pred.start_t_b, dtype=np.int32)

    if np.any(np.asarray(render_sample_indices) >= pred_traj.shape[1]):
        raise IndexError(
            f"render_sample_indices out of bounds for num_samples={pred_traj.shape[1]}: {render_sample_indices}"
        )
    if np.any(np.asarray(rollout_sample_indices) >= pred_traj.shape[1]):
        raise IndexError(
            f"rollout_sample_indices out of bounds for num_samples={pred_traj.shape[1]}: {rollout_sample_indices}"
        )

    rollout = rollout_predicted_trajectories_with_metrics(
        sim_state,
        pred,
        metric_names=metric_names,
        rollout_num_steps=args.rollout_num_steps,
        sample_indices=rollout_sample_indices,
        rng_key=rng_key,
    )

    video_requests: list[tuple[str, int]] = []
    ego_start_times: list[int] = []
    ego_trajectories: list[np.ndarray] = []
    video_manifest: list[dict[str, Any]] = []

    for order_idx, scenario_index in enumerate(scenario_indices):
        batch_idx = ordered_batch_indices[order_idx]
        for sample_index in render_sample_indices:
            video_requests.append((str(args.tfrecord_path), int(scenario_index)))
            ego_start_times.append(int(start_t[batch_idx]))
            ego_trajectories.append(pred_traj[batch_idx, int(sample_index)])
            video_manifest.append(
                {
                    "scenario_index": int(scenario_index),
                    "batch_index": int(batch_idx),
                    "sample_index": int(sample_index),
                    "start_t": int(start_t[batch_idx]),
                }
            )

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
    )

    original_video_paths: list[str] = []
    if args.render_original:
        original_video_paths = [
            p.as_posix()
            for p in render_videos_batched(
                tfrecord_scenarios=[(str(args.tfrecord_path), int(idx)) for idx in scenario_indices],
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

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "metadata_path": args.metadata_path,
        "tfrecord_path": str(args.tfrecord_path),
        "scenario_indices": [int(x) for x in scenario_indices],
        "render_sample_indices": [int(x) for x in render_sample_indices],
        "rollout_sample_indices": [int(x) for x in rollout_sample_indices],
        "num_samples_generated": int(pred_traj.shape[1]),
        "predict_horizon_world_steps": int(pred_traj.shape[2]),
        "start_t_by_scenario": {
            str(int(scenario_indices[i])): int(start_t[ordered_batch_indices[i]])
            for i in range(len(scenario_indices))
        },
        "metric_names": list(rollout.metric_names),
        "metric_summary": metric_summary,
        "predicted_videos": [
            {
                **video_manifest[i],
                "path": predicted_video_paths[i].as_posix(),
            }
            for i in range(len(predicted_video_paths))
        ],
        "original_videos": original_video_paths,
    }
    _save_json(output_dir / "summary.json", summary)

    if args.save_predictions_npz:
        np.savez_compressed(
            output_dir / "predictions.npz",
            scenario_indices=np.asarray(scenario_indices, dtype=np.int32),
            batch_indices=np.asarray(ordered_batch_indices, dtype=np.int32),
            start_t=start_t,
            trajectories_world_bkt5=pred_traj,
            world_t_seconds_bk=np.asarray(pred.world_t_seconds_bk),
            world_t_valid_bk=np.asarray(pred.world_t_valid_bk),
        )

    print(f"Saved summary to {output_dir / 'summary.json'}")
    for item in summary["predicted_videos"]:
        print(
            "predicted_video",
            f"scenario={item['scenario_index']}",
            f"sample={item['sample_index']}",
            f"path={item['path']}",
        )
    if original_video_paths:
        for path in original_video_paths:
            print("original_video", f"path={path}")


if __name__ == "__main__":
    main()
