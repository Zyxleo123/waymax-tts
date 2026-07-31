#!/usr/bin/env python3
"""Stochastic SAC init safety diagnostic (Path 2) — same 0/K question as ES.

For each scene in a ScenarioMax tfrecord (e.g. ``es_scenes.tfrecord``), roll the
``repro_sac_v2`` policy **K** times with different RNG seeds (stochastic SAC
actions) and count how many rollouts stay binary-safe under V-Max metrics
(``overlap`` / ``offroad`` never fire). Reports the same bimodal 0-vs-K pattern
we saw for diffusion init.

No EMA in this run — use ``model_final.pkl`` (last ckpt).

Example (GPU node, waymax env)::

    python es_baseline/experiments/diagnose_sac_init_safety.py \\
      --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2 \\
      --path-dataset /zfsauton/scratch/yixiz/ScenarioMaxWaymoES/es_scenes.tfrecord \\
      --out /zfsauton/scratch/yixiz/ScenarioMaxWaymoES/sac_init_safety.json \\
      --k 64 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vmax.scripts.evaluate import utils  # noqa: E402
from vmax.simulator import datasets, make_data_generator  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--path-dataset", required=True,
                   help="ScenarioMax tfrecord (es_scenes.tfrecord)")
    p.add_argument("--model", default=None,
                   help="Explicit .pkl (default: model_final.pkl / latest)")
    p.add_argument("--out", required=True,
                   help="JSON summary path (must be under /zfsauton/scratch/...)")
    p.add_argument("--k", type=int, default=64, help="stochastic rollouts per scene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _require_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    s = str(resolved)
    if not (s.startswith("/zfsauton/scratch/") or s.startswith("/scratch/")):
        raise SystemExit(
            f"Refusing to write large artifacts outside scratch: {resolved}\n"
            f"Pass --out under /zfsauton/scratch/..."
        )
    return resolved


def _rollout_safe(step_fn, reset_fn, scenario, rng_key, max_steps: int) -> dict:
    """One stochastic closed-loop rollout. Safe iff no overlap/offroad ever."""
    rng_key, rkey = jax.random.split(rng_key)
    rkey = jax.random.split(rkey, 1)
    env_tr = reset_fn(scenario, rkey)
    overlap = offroad = False
    steps = 0
    done = bool(np.asarray(env_tr.done).reshape(-1)[0])
    while not done and steps < max_steps:
        m = env_tr.metrics
        if "overlap" in m and float(np.asarray(m["overlap"]).reshape(-1)[0]) > 0.5:
            overlap = True
        if "offroad" in m and float(np.asarray(m["offroad"]).reshape(-1)[0]) > 0.5:
            offroad = True
        rng_key, skey = jax.random.split(rng_key)
        skey = jax.random.split(skey, 1)
        env_tr, _ = step_fn(env_tr, key=skey)
        steps += 1
        done = bool(np.asarray(env_tr.done).reshape(-1)[0])
    # also check final transition metrics
    m = env_tr.metrics
    if "overlap" in m and float(np.asarray(m["overlap"]).reshape(-1)[0]) > 0.5:
        overlap = True
    if "offroad" in m and float(np.asarray(m["offroad"]).reshape(-1)[0]) > 0.5:
        offroad = True
    return {
        "steps": steps,
        "overlap": overlap,
        "offroad": offroad,
        "safe": (not overlap) and (not offroad),
    }


def main() -> None:
    args = _parse()
    out_json = _require_scratch(Path(args.out))
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # V-Max setup_evaluation embeds path_dataset in eval_path; keep a short
    # scratch-local name so it does not nest absolute dataset paths under cwd.
    out_setup = out_dir / "_sac_init_setup"
    out_setup.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    if args.model:
        print(f"NOTE: setup_evaluation selects model_final.pkl from {run_dir}/model/; "
              f"requested --model={args.model}")

    # eval_name must be absolute-under-scratch; use a short leaf so mkdir stays flat.
    eval_name = str(out_setup / "vmax_eval")
    env, step_fn, eval_path, term_keys = utils.setup_evaluation(
        "ai",
        run_dir.name,
        str(run_dir.parent),
        # Use basename-only style via datasets.get_dataset later; pass path as-is
        # but eval_name is scratch-rooted so all writes stay on scratch.
        args.path_dataset,
        eval_name,
        args.max_num_objects,
        False,
        True,
    )
    print(f"termination_keys={term_keys}  eval_path={eval_path}")
    jitted_step = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    gen = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=args.seed,
        repeat=1,
    )

    rng = jax.random.PRNGKey(args.seed)
    scenes = []
    for idx, scenario in enumerate(gen):
        if args.limit is not None and idx >= args.limit:
            break
        safes = 0
        per = []
        for k in range(args.k):
            rng, key = jax.random.split(rng)
            r = _rollout_safe(jitted_step, jitted_reset, scenario, key, args.max_steps)
            per.append(r)
            safes += int(r["safe"])
        rec = {
            "scene_index": idx,
            "k": args.k,
            "num_safe": safes,
            "frac_safe": safes / args.k,
            "n_overlap": sum(1 for r in per if r["overlap"]),
            "n_offroad": sum(1 for r in per if r["offroad"]),
        }
        scenes.append(rec)
        print(f"scene {idx:3d}: safe={safes}/{args.k}  "
              f"overlap={rec['n_overlap']} offroad={rec['n_offroad']}")

    n = len(scenes)
    n0 = sum(1 for s in scenes if s["num_safe"] == 0)
    nK = sum(1 for s in scenes if s["num_safe"] == args.k)
    blob = {
        "args": {**vars(args), "out": str(out_json)},
        "summary": {
            "n_scenes": n,
            "k": args.k,
            "frac_all_zero": n0 / n if n else 0.0,
            "frac_all_safe": nK / n if n else 0.0,
            "mean_num_safe": float(np.mean([s["num_safe"] for s in scenes])) if n else 0.0,
            "n_all_zero": n0,
            "n_all_safe": nK,
        },
        "scenes": scenes,
    }
    out_json.write_text(json.dumps(blob, indent=2))
    print(json.dumps(blob["summary"], indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
