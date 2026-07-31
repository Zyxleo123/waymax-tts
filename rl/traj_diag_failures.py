#!/usr/bin/env python3
"""Low-memory trajectory diagnostics for a V-Max SAC ckpt on failure cases.

Records ego (x,y), done reason proxies, and goal distance per step — no
frame rendering — then writes PNG path plots + a small JSON summary. Designed
to survive ~1–2 GiB free cgroup headroom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vmax.scripts.evaluate import utils  # noqa: E402
from vmax.simulator import datasets, make_data_generator  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--path-dataset",
        default="/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _sdc_xy(state) -> tuple[float, float]:
    # PlanningAgent / brax state carries waymax SimulatorState under .state
    sim = state.state if hasattr(state, "state") else state
    traj = sim.sim_trajectory
    # is_sdc mask may be under object_metadata
    is_sdc = np.asarray(sim.object_metadata.is_sdc).reshape(-1)
    sdc_idx = int(np.argmax(is_sdc))
    t = int(np.asarray(sim.timestep).reshape(-1)[0])
    x = float(np.asarray(traj.x).reshape(-1, traj.x.shape[-1])[sdc_idx, t])
    y = float(np.asarray(traj.y).reshape(-1, traj.y.shape[-1])[sdc_idx, t])
    return x, y


def main() -> None:
    args = _parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.indices.split(",") if x.strip() != ""}

    run_dir = Path(args.run_dir).resolve()
    env, step_fn, _, term_keys = utils.setup_evaluation(
        "ai",
        run_dir.name,
        str(run_dir.parent),
        args.path_dataset,
        str(out / "ai_setup"),
        64,
        False,
        True,
    )
    jitted_step = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    gen = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=args.seed,
        repeat=1,
    )

    rng = jax.random.PRNGKey(args.seed)
    summaries = []
    for idx, scenario in enumerate(gen):
        if idx not in wanted:
            continue
        rng, key = jax.random.split(rng)
        key, rkey = jax.random.split(key)
        rkey = jax.random.split(rkey, 1)
        env_tr = jitted_reset(scenario, rkey)

        xs, ys, dgoals = [], [], []
        overlaps, offroads, reds = [], [], []
        steps = 0
        done = bool(np.asarray(env_tr.done).reshape(-1)[0])
        while not done and steps < 90:
            x, y = _sdc_xy(env_tr)
            xs.append(x)
            ys.append(y)
            metrics = env_tr.metrics
            dgoals.append(float(np.asarray(metrics.get("distance_to_goal", metrics.get("min_distance_to_goal", np.nan))).reshape(-1)[0]))
            overlaps.append(float(np.asarray(metrics.get("overlap", 0.0)).reshape(-1)[0]))
            offroads.append(float(np.asarray(metrics.get("offroad", 0.0)).reshape(-1)[0]))
            reds.append(float(np.asarray(metrics.get("run_red_light", 0.0)).reshape(-1)[0]))
            key, skey = jax.random.split(key)
            skey = jax.random.split(skey, 1)
            env_tr, _ = jitted_step(env_tr, key=skey)
            steps += 1
            done = bool(np.asarray(env_tr.done).reshape(-1)[0])

        # Final metrics after last step already recorded partially; pull episode flags.
        reached = float(np.asarray(env_tr.metrics.get("reached_goal", 0.0)).reshape(-1)[0]) if "reached_goal" in env_tr.metrics else float("nan")
        # reached_goal in vmax is often end-of-episode aggregate; also check min distance.
        min_d = float(np.nanmin(dgoals)) if dgoals else float("nan")
        hit_overlap = any(v > 0.5 for v in overlaps)
        hit_offroad = any(v > 0.5 for v in offroads)
        hit_red = any(v > 0.5 for v in reds)
        summary = {
            "scenario_index": idx,
            "steps": steps,
            "min_distance_to_goal": min_d,
            "final_distance_to_goal": dgoals[-1] if dgoals else float("nan"),
            "reached_goal_metric": reached,
            "hit_overlap": hit_overlap,
            "hit_offroad": hit_offroad,
            "hit_red": hit_red,
            "safe": not (hit_overlap or hit_offroad or hit_red),
        }
        summaries.append(summary)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(xs, ys, "-", lw=2, label="ego")
        if xs:
            ax.scatter([xs[0]], [ys[0]], c="green", s=40, zorder=3, label="start")
            ax.scatter([xs[-1]], [ys[-1]], c="red", s=40, zorder=3, label="end")
        # Mark first collision/offroad timestep
        for name, series, color in (("overlap", overlaps, "orange"), ("offroad", offroads, "purple")):
            for t, v in enumerate(series):
                if v > 0.5:
                    ax.scatter([xs[t]], [ys[t]], c=color, marker="x", s=80, label=f"{name}@{t}")
                    break
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        title = (
            f"idx={idx} steps={steps} min_d={min_d:.1f} "
            f"ov={hit_overlap} off={hit_offroad} safe={summary['safe']}"
        )
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"traj_{idx:04d}.png", dpi=120)
        plt.close(fig)
        np.savez_compressed(
            out / f"traj_{idx:04d}.npz",
            x=np.asarray(xs),
            y=np.asarray(ys),
            distance_to_goal=np.asarray(dgoals),
            overlap=np.asarray(overlaps),
            offroad=np.asarray(offroads),
        )
        print(f"[traj] {title}", flush=True)
        jax.clear_caches()
        wanted.discard(idx)
        if not wanted:
            break

    (out / "traj_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[traj] wrote {len(summaries)} summaries -> {out}", flush=True)


if __name__ == "__main__":
    main()
