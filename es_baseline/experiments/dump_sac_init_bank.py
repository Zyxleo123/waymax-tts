#!/usr/bin/env python3
"""Dump K=64 stochastic SAC closed-loop ego trajectories for ES sac_safe init.

Rolls ``repro_sac_v2`` on the ScenarioMax ES shard and writes a scratch NPZ
bank keyed by ``es_scenes_record_index`` (same order as ``es_scenes.tfrecord``).

Each traj step is world-frame ``[x, y, yaw, vx, vy]`` at V-Max dt (~0.1s).

Example::

    python es_baseline/experiments/dump_sac_init_bank.py \\
      --run-dir /zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2 \\
      --path-dataset /zfsauton/scratch/yixiz/ScenarioMaxWaymoES/es_scenes.tfrecord \\
      --manifest /zfsauton/scratch/yixiz/ScenarioMaxWaymoES/es_scenes_manifest.json \\
      --out /zfsauton/scratch/yixiz/ScenarioMaxWaymoES/sac_init_bank.npz \\
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
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_VMAX = _REPO / "V-Max"
for p in (_REPO, _VMAX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vmax.scripts.evaluate import utils  # noqa: E402
from vmax.simulator import datasets, make_data_generator, operations  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--path-dataset", required=True)
    p.add_argument("--manifest", required=True,
                   help="es_scenes_manifest.json (maps record index → WOMD shard/idx)")
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True,
                   help="NPZ path under /zfsauton/scratch/...")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dt", type=float, default=0.1,
                   help="V-Max / Waymax sim dt used when rolling (seconds)")
    p.add_argument("--noisy-init", action="store_true", default=False,
                   help="V-Max NoisyInitWrapper: perturbs the initial action per reset, "
                        "so the K rollouts differ. Without it the eval policy is "
                        "deterministic (make_policy(deterministic=True)) and all K "
                        "members come out identical. Note it also moves the reset point "
                        "earlier (init_steps 1 vs 11), which is why start_timestep is "
                        "measured from the env rather than assumed.")
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


def _ego_xyyawvv(state) -> np.ndarray:
    """Current SDC pose as float32 [x, y, yaw, vx, vy] (batch dim optional)."""
    is_sdc = np.asarray(state.object_metadata.is_sdc)
    traj = state.current_sim_trajectory
    if is_sdc.ndim == 2:
        sdc_idx = int(np.argmax(is_sdc[0].astype(np.int32)))
        x = float(np.asarray(traj.x[0, sdc_idx]).reshape(-1)[0])
        y = float(np.asarray(traj.y[0, sdc_idx]).reshape(-1)[0])
        yaw = float(np.asarray(traj.yaw[0, sdc_idx]).reshape(-1)[0])
        vx = float(np.asarray(traj.vel_x[0, sdc_idx]).reshape(-1)[0])
        vy = float(np.asarray(traj.vel_y[0, sdc_idx]).reshape(-1)[0])
    else:
        sdc_idx = int(operations.get_index(state.object_metadata.is_sdc))
        x = float(np.asarray(traj.x[sdc_idx]).reshape(-1)[0])
        y = float(np.asarray(traj.y[sdc_idx]).reshape(-1)[0])
        yaw = float(np.asarray(traj.yaw[sdc_idx]).reshape(-1)[0])
        vx = float(np.asarray(traj.vel_x[sdc_idx]).reshape(-1)[0])
        vy = float(np.asarray(traj.vel_y[sdc_idx]).reshape(-1)[0])
    return np.asarray([x, y, yaw, vx, vy], dtype=np.float32)


def _rollout_traj(step_fn, reset_fn, scenario, rng_key, max_steps: int):
    """Closed-loop rollout; returns poses [T,5], safe, overlap, offroad, length."""
    rng_key, rkey = jax.random.split(rng_key)
    rkey = jax.random.split(rkey, 1)
    env_tr = reset_fn(scenario, rkey)
    start_timestep = int(np.asarray(env_tr.state.timestep).reshape(-1)[0])
    poses = [_ego_xyyawvv(env_tr.state)]
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
        poses.append(_ego_xyyawvv(env_tr.state))
        steps += 1
        done = bool(np.asarray(env_tr.done).reshape(-1)[0])
    m = env_tr.metrics
    if "overlap" in m and float(np.asarray(m["overlap"]).reshape(-1)[0]) > 0.5:
        overlap = True
    if "offroad" in m and float(np.asarray(m["offroad"]).reshape(-1)[0]) > 0.5:
        offroad = True
    traj = np.stack(poses, axis=0).astype(np.float32)
    return {
        "traj": traj,
        "start_timestep": start_timestep,
        "length": int(traj.shape[0]),
        "overlap": overlap,
        "offroad": offroad,
        "safe": (not overlap) and (not offroad),
    }


def _pad_traj(traj: np.ndarray, t_max: int) -> np.ndarray:
    t = traj.shape[0]
    out = np.zeros((t_max, 5), dtype=np.float32)
    n = min(t, t_max)
    out[:n] = traj[:n]
    if t < t_max and t > 0:
        out[t:] = traj[-1]
    return out


def main() -> None:
    args = _parse()
    out_npz = _require_scratch(Path(args.out))
    out_dir = out_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_npz.with_suffix(".json")
    out_setup = out_dir / "_sac_init_setup"
    out_setup.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    by_rec = {}
    for r in manifest["records"]:
        if r.get("es_scenes_record_index") is not None:
            by_rec[int(r["es_scenes_record_index"])] = r

    run_dir = Path(args.run_dir).resolve()
    if args.model:
        print(f"NOTE: setup_evaluation selects model_final.pkl from {run_dir}/model/; "
              f"requested --model={args.model}")

    eval_name = str(out_setup / "vmax_eval_dump")
    env, step_fn, eval_path, term_keys = utils.setup_evaluation(
        "ai",
        run_dir.name,
        str(run_dir.parent),
        args.path_dataset,
        eval_name,
        args.max_num_objects,
        bool(args.noisy_init),
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
        # NOT args.seed: V-Max forwards this as waymax's shuffle_seed, so a non-None
        # value shuffles the dataset while the loop below still stamps each row with
        # manifest[enumerate_index] — that mislabelled every row of the first bank.
        seed=None,
        repeat=1,
    )

    # T_max poses = max_steps + 1 (reset + steps)
    t_max = int(args.max_steps) + 1
    trajs = []
    safes = []
    lengths = []
    scene_meta = []
    start_steps = set()
    rng = jax.random.PRNGKey(args.seed)

    for idx, scenario in enumerate(gen):
        if args.limit is not None and idx >= args.limit:
            break
        man = by_rec.get(idx, {})
        scene_trajs = np.zeros((args.k, t_max, 5), dtype=np.float32)
        scene_safe = np.zeros((args.k,), dtype=np.bool_)
        scene_len = np.zeros((args.k,), dtype=np.int32)
        n_safe = 0
        for k in range(args.k):
            rng, key = jax.random.split(rng)
            r = _rollout_traj(jitted_step, jitted_reset, scenario, key, args.max_steps)
            scene_trajs[k] = _pad_traj(r["traj"], t_max)
            scene_safe[k] = bool(r["safe"])
            scene_len[k] = int(r["length"])
            start_steps.add(int(r["start_timestep"]))
            n_safe += int(r["safe"])
        trajs.append(scene_trajs)
        safes.append(scene_safe)
        lengths.append(scene_len)
        scene_meta.append({
            "es_scenes_record_index": idx,
            "scenario_id": man.get("scenario_id"),
            "tf_example_shard": man.get("tf_example_shard"),
            "tf_example_idx": man.get("tf_example_idx"),
            "num_safe": n_safe,
            "k": args.k,
            "unique_members": int(len(np.unique(
                np.round(scene_trajs[:, :, :2].reshape(args.k, -1), 3), axis=0))),
        })
        print(f"scene {idx:3d}: safe={n_safe}/{args.k}  "
              f"shard={man.get('tf_example_shard')} idx={man.get('tf_example_idx')}  "
              f"id={man.get('scenario_id')}", flush=True)

    traj_arr = np.stack(trajs, axis=0)       # [N,K,T,5]
    safe_arr = np.stack(safes, axis=0)       # [N,K]
    len_arr = np.stack(lengths, axis=0)      # [N,K]
    rec_idx = np.asarray([m["es_scenes_record_index"] for m in scene_meta], dtype=np.int32)

    if len(start_steps) > 1:
        raise SystemExit(f"rollouts reset at differing timesteps {sorted(start_steps)}; "
                         f"the bank needs one time base")
    dup = [m for m in scene_meta if m["unique_members"] < args.k]
    if dup:
        print(f"WARNING: {len(dup)}/{len(scene_meta)} scenes have duplicate members "
              f"(min unique={min(m['unique_members'] for m in dup)} of K={args.k}). "
              f"The eval policy is deterministic — pass --noisy-init for a diverse bank.")

    np.savez_compressed(
        out_npz,
        traj=traj_arr,
        safe=safe_arr,
        length=len_arr,
        es_scenes_record_index=rec_idx,
        dt=np.float32(args.dt),
        layout=np.asarray(["x", "y", "yaw", "vx", "vy"]),
        k=np.int32(args.k),
        t_max=np.int32(t_max),
        seed=np.int32(args.seed),
        start_timestep=np.int32(sorted(start_steps)[0] if start_steps else 0),
        noisy_init=np.bool_(bool(args.noisy_init)),
    )
    n = len(scene_meta)
    n0 = sum(1 for m in scene_meta if m["num_safe"] == 0)
    nK = sum(1 for m in scene_meta if m["num_safe"] == args.k)
    blob = {
        "args": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                 "out": str(out_npz)},
        "summary": {
            "n_scenes": n,
            "k": args.k,
            "t_max": t_max,
            "dt": args.dt,
            "layout": ["x", "y", "yaw", "vx", "vy"],
            "frac_all_zero": n0 / n if n else 0.0,
            "frac_all_safe": nK / n if n else 0.0,
            "mean_num_safe": float(np.mean([m["num_safe"] for m in scene_meta])) if n else 0.0,
            "n_all_zero": n0,
            "n_all_safe": nK,
            "npz": str(out_npz),
        },
        "scenes": scene_meta,
    }
    out_json.write_text(json.dumps(blob, indent=2))
    print(json.dumps(blob["summary"], indent=2))
    print(f"wrote {out_npz} and {out_json}")


if __name__ == "__main__":
    main()
