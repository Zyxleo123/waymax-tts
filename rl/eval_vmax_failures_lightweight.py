#!/usr/bin/env python3
"""Lightweight V-Max SAC eval on the ScenarioMax failure tfrecord.

Avoids the full ``jax.lax.while_loop`` + ``vmap`` JIT used by
``vmax.scripts.evaluate.evaluate`` (peaks >8 GiB on CPU and gets cgroup-OOM
killed on small allocations). Rolls with a Python step loop (same as the
render path), dumps per-scenario CSV, optionally writes a few mp4s.

Example::

    JAX_PLATFORMS=cpu python -m rl.eval_vmax_failures_lightweight \\
      --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/_eval_staged_overfit_best_261120 \\
      --path-dataset /zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord \\
      --out-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/_eval_failures_best_261120_lw \\
      --render-indices 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vmax.scripts.evaluate import utils  # noqa: E402
from vmax.simulator import datasets, make_data_generator  # noqa: E402
from vmax.simulator.metrics.collector import check_episode_success  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Hydra run dir with model/ + .hydra/")
    p.add_argument(
        "--path-dataset",
        default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip scenarios with index < start-index (for chunked runs).",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing evaluation_episodes.csv instead of overwriting.",
    )
    p.add_argument(
        "--render-indices",
        default="",
        help="Comma-separated scenario indices to write mp4 for (after metrics).",
    )
    p.add_argument("--max-num-objects", type=int, default=64)
    return p.parse_args()


def _flatten_metric_scalar(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(np.asarray(arr).reshape(-1)[0])


def _rollout_metrics(env, step_fn, reset_fn, scenario, rng_key, termination_keys, max_steps=90):
    """Python-loop rollout; returns (per_key episode aggregates, steps_done, done)."""
    rng_key, reset_key = jax.random.split(rng_key)
    reset_key = jax.random.split(reset_key, 1)
    env_transition = reset_fn(scenario, reset_key)

    series: dict[str, list[float]] = {k: [] for k in env_transition.metrics}
    steps = 0
    done = bool(np.asarray(env_transition.done).reshape(-1)[0])

    while not done and steps < max_steps:
        rng_key, step_key = jax.random.split(rng_key)
        step_key = jax.random.split(step_key, 1)
        env_transition, _ = step_fn(env_transition, key=step_key)
        for k, v in env_transition.metrics.items():
            series[k].append(_flatten_metric_scalar(v))
        steps += 1
        done = bool(np.asarray(env_transition.done).reshape(-1)[0])

    # Aggregate like V-Max's metric operands: most are mean; some are max/any.
    from vmax.simulator.metrics.collector import _metrics_operands

    batch_metrics = {}
    for key, vals in series.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        metric_key = key.split("/")[-1]
        operand = _metrics_operands.get(metric_key, np.mean)
        if not isinstance(operand, dict):
            batch_metrics[metric_key] = float(operand(arr))
        else:
            for sub_key, sub_operand in operand.items():
                batch_metrics[sub_key] = float(sub_operand(arr))

    accuracy = check_episode_success(batch_metrics, termination_keys)
    batch_metrics["accuracy"] = float(accuracy)
    batch_metrics["episode_length"] = float(steps)
    return batch_metrics, steps


def _render_one(env, step_fn, reset_fn, scenario, rng_key, out_mp4: Path, max_steps=90):
    import mediapy

    images, _ = utils.run_scenario_render(
        scenario, rng_key, env=env, step_fn=step_fn, reset_fn=reset_fn, render_pov=False
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out_mp4), images, fps=10)
    images.clear()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    # setup_evaluation expects src_dir/path_model layout.
    src_dir = str(run_dir.parent)
    path_model = run_dir.name

    print(f"[lw-eval] devices={jax.devices()}", flush=True)
    env, step_fn, _eval_path, termination_keys = utils.setup_evaluation(
        "ai",
        path_model,
        src_dir,
        args.path_dataset,
        str(out_dir / "ai_setup"),
        args.max_num_objects,
        False,
        True,
    )
    # JIT only the inner step/reset (cheap vs while_loop).
    jitted_step = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    data_generator = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),  # unbatched leaf for brax Wrapper.reset assert
        seed=args.seed,
        repeat=1,
    )

    import json

    csv_path = out_dir / "evaluation_episodes.csv"
    rows: list[dict] = []
    if args.append and csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                for k, v in list(r.items()):
                    if k == "scenario_index":
                        r[k] = int(v)
                    else:
                        try:
                            r[k] = float(v)
                        except (TypeError, ValueError):
                            pass
        print(f"[lw-eval] resumed with {len(rows)} existing rows", flush=True)

    end_index = args.limit  # keep historical meaning: stop before this absolute index
    fieldnames: list[str] | None = list(rows[0].keys()) if rows else None
    csv_file = None
    writer = None
    if fieldnames is not None:
        csv_file = csv_path.open("a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    rng = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    chunk_rows: list[dict] = []
    try:
        for idx, scenario in enumerate(data_generator):
            if idx < args.start_index:
                continue
            if end_index is not None and idx >= end_index:
                break
            rng, scenario_key = jax.random.split(rng)
            metrics, steps = _rollout_metrics(
                env, jitted_step, jitted_reset, scenario, scenario_key, termination_keys
            )
            row = {"scenario_index": idx, **metrics}
            chunk_rows.append(row)
            rows.append(row)

            if fieldnames is None:
                fieldnames = list(row.keys())
                csv_file = csv_path.open("w", newline="")
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
            assert writer is not None and csv_file is not None
            writer.writerow(row)
            csv_file.flush()

            if (idx + 1) % 5 == 0 or idx == args.start_index:
                acc = np.mean([r["accuracy"] for r in rows])
                rg = np.mean([r.get("reached_goal", float("nan")) for r in rows])
                print(
                    f"[lw-eval] idx={idx} | n={len(rows)} acc={acc:.3f} reached={rg:.3f} "
                    f"last_len={steps} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
            # Bound JAX compile/cache growth on small-memory cgroups.
            if (idx + 1) % 20 == 0:
                jax.clear_caches()
    finally:
        if csv_file is not None:
            csv_file.close()

    if not rows:
        raise SystemExit("[lw-eval] no scenarios rolled — check dataset path")

    assert fieldnames is not None
    summary = {
        k: float(np.nanmean([float(r[k]) for r in rows]))
        for k in fieldnames
        if k != "scenario_index"
    }
    joint = float(
        np.mean(
            [
                1.0 if (float(r.get("accuracy", 0)) >= 0.5 and float(r.get("reached_goal", 0)) >= 0.5) else 0.0
                for r in rows
            ]
        )
    )
    summary["joint_accuracy_and_reached_goal"] = joint
    summary["n"] = float(len(rows))
    summary["chunk_n"] = float(len(chunk_rows))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[lw-eval] wrote {csv_path} (+{len(chunk_rows)} this chunk)", flush=True)
    print(f"[lw-eval] summary={summary_path}", flush=True)
    print(
        f"[lw-eval] n={len(rows)} accuracy={summary.get('accuracy', float('nan')):.4f} "
        f"reached_goal={summary.get('reached_goal', float('nan')):.4f} "
        f"joint={joint:.4f}",
        flush=True,
    )

    render_idxs = [int(x) for x in args.render_indices.split(",") if x.strip() != ""]
    if render_idxs:
        # Re-walk generator for render targets (repeat=1 already consumed).
        data_generator = make_data_generator(
            path=datasets.get_dataset(args.path_dataset),
            max_num_objects=args.max_num_objects,
            include_sdc_paths=True,
            batch_dims=(1,),
            seed=args.seed,
            repeat=1,
        )
        rng = jax.random.PRNGKey(args.seed + 1)
        wanted = set(render_idxs)
        for idx, scenario in enumerate(data_generator):
            if idx not in wanted:
                continue
            rng, scenario_key = jax.random.split(rng)
            mp4 = out_dir / "mp4" / f"eval_{idx:04d}.mp4"
            print(f"[lw-eval] rendering scenario {idx} -> {mp4}", flush=True)
            _render_one(env, jitted_step, jitted_reset, scenario, scenario_key, mp4)
            wanted.discard(idx)
            if not wanted:
                break

    print("[lw-eval] done", flush=True)


if __name__ == "__main__":
    main()
