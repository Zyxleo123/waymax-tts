#!/usr/bin/env python3
"""Smoke test for the online SAC init roller (no diffusion model, no ES).

Checks the three things that can silently be wrong:

  1. the policy loads and produces *different* rollouts (K identical members would
     mean the population has one mode, which is what the offline deterministic bank
     did before ``--noisy-init``),
  2. the rollouts start at the ego pose we handed in (anchor gap ~ 0 m), which is the
     whole point of rolling online, and
  3. splicing an ego history that is *not* the log actually moves the rollout: we
     re-run with the ego displaced laterally and check the trajectories move with it.

Run it on a GPU node via es_baseline/slurm_sac_online_smoke.sbatch.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

try:
    import tensorflow as _tf

    _tf.config.set_visible_devices([], "GPU")
except Exception as _e:  # pragma: no cover
    print(f"[smoke] could not hide GPU from TensorFlow: {_e}")

import jax  # noqa: E402
import numpy as np  # noqa: E402

_ES = Path(__file__).resolve().parents[1]
if str(_ES) not in sys.path:
    sys.path.insert(0, str(_ES))

from experiments.sac_online import OnlineSacInit  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sac_run_dir",
                   default="/zfsauton/scratch/yixiz/waymax_rs/vmax_womd_raw_parity/"
                           "womd_raw_parity_sac_lq")
    p.add_argument("--sac_model", default=None)
    p.add_argument("--tfrecord",
                   default=str(_ES / "repro" / "tf00000" / "training" /
                               "training_tfexample.tfrecord-00000-of-01000"))
    p.add_argument("--scenario_index", type=int, default=71)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--rollout_steps", type=int, default=50)
    p.add_argument("--timesteps", default="10,40")
    p.add_argument("--action_noise", default="0.0",
                   help="comma-separated action-noise levels to sweep in one job, e.g. "
                        "'0,0.05,0.1'. The policy's own entropy is what a bare 0 buys.")
    return p.parse_args()


def main() -> None:
    args = _parse()
    print("jax devices:", jax.devices(), flush=True)

    noise_levels = [float(x) for x in str(args.action_noise).split(",") if x != ""]
    rollers = []
    for noise in noise_levels:
        rollers.append((noise, OnlineSacInit(
            run_dir=args.sac_run_dir,
            tfrecord_path=args.tfrecord,
            model_path=args.sac_model,
            k=args.k,
            rollout_steps=args.rollout_steps,
            action_noise_std=noise,
        )))
    roller = rollers[0][1]
    print(f"model={roller.model_path}  max_num_objects={roller.max_num_objects}  "
          f"init_steps={roller.init_steps}  noise sweep={noise_levels}", flush=True)

    idx = int(args.scenario_index)
    roller.prepare([idx])
    scenario = roller._scenario(idx)
    sdc = int(np.argmax(np.asarray(scenario.object_metadata.is_sdc).astype(np.int32)))
    log = scenario.log_trajectory
    ego_log = np.stack([
        np.asarray(log.x)[sdc], np.asarray(log.y)[sdc], np.asarray(log.yaw)[sdc],
        np.asarray(log.vel_x)[sdc], np.asarray(log.vel_y)[sdc],
    ], axis=-1).astype(np.float32)      # [T, 5]
    print(f"scenario {idx}: {ego_log.shape[0]} logged steps", flush=True)

    failures = []
    for t in [int(x) for x in args.timesteps.split(",")]:
        key = jax.random.PRNGKey(t)
        wall_start = time.perf_counter()
        traj, safe, length = roller.rollout(
            scenario_index=idx, ego_history_t5=ego_log, timestep=t, rng=key)
        wall = time.perf_counter() - wall_start
        anchor_gap = np.linalg.norm(traj[:, 0, :2] - ego_log[t, :2][None, :], axis=1)
        spread = float(np.max(np.std(traj[:, -1, :2], axis=0)))
        uniq = len(np.unique(np.round(traj[:, :, :2].reshape(traj.shape[0], -1), 3), axis=0))
        print(f"\nt={t:3d}  traj={traj.shape}  safe={int(safe.sum())}/{len(safe)}  "
              f"len[min,max]=[{length.min()},{length.max()}]  "
              f"wall={wall:.2f}s (first call includes jit)")
        print(f"        anchor gap max={anchor_gap.max():.4f} m   "
              f"unique members={uniq}/{traj.shape[0]}   endpoint std={spread:.2f} m")

        # How much coverage does the population actually have, and what does buying
        # more of it with action noise cost in safety? This is the knob to set before
        # running the arm: a population that all goes the same place is one candidate.
        if len(rollers) > 1:
            print("        noise  endpoint-spread  lateral-spread  unique  safe")
            for noise, r in rollers:
                if r is roller:
                    traj_n, safe_n = traj, safe
                else:
                    traj_n, safe_n, _ = r.rollout(
                        scenario_index=idx, ego_history_t5=ego_log, timestep=t, rng=key)
                ends = traj_n[:, -1, :2]
                pair = float(np.mean(np.linalg.norm(
                    ends[:, None, :] - ends[None, :, :], axis=-1)))
                yaw0 = float(ego_log[t, 2])
                lat = float(np.std((ends[:, 0] - ego_log[t, 0]) * -np.sin(yaw0)
                                   + (ends[:, 1] - ego_log[t, 1]) * np.cos(yaw0)))
                u = len(np.unique(np.round(traj_n[:, :, :2].reshape(traj_n.shape[0], -1), 3), axis=0))
                print(f"        {noise:5.3f}  {pair:14.2f} m  {lat:12.2f} m  "
                      f"{u:3d}/{traj_n.shape[0]}  {int(np.sum(safe_n)):3d}/{len(safe_n)}")

        if anchor_gap.max() > 1e-3:
            failures.append(f"t={t}: rollouts do not start at the given ego pose "
                            f"(max gap {anchor_gap.max():.3f} m)")
        # Near the end of the episode the rollout is clamped to a couple of steps, and
        # two sampled actions genuinely can land within the 1 mm rounding used above —
        # only a rollout with room to diverge says anything about population diversity.
        if uniq < 2 and traj.shape[1] > 10:
            failures.append(f"t={t}: all {traj.shape[0]} rollouts identical over "
                            f"{traj.shape[1]} steps — the population has a single mode")

        # (3) displace the ego 2 m laterally and check the rollouts follow.
        moved = ego_log.copy()
        yaw = moved[t, 2]
        moved[: t + 1, 0] += 2.0 * -np.sin(yaw)
        moved[: t + 1, 1] += 2.0 * np.cos(yaw)
        traj2, _, _ = roller.rollout(
            scenario_index=idx, ego_history_t5=moved, timestep=t, rng=key)
        shift = float(np.linalg.norm(traj2[:, 0, :2] - traj[:, 0, :2], axis=1).mean())
        print(f"        ego displaced 2.00 m -> rollout start moved {shift:.2f} m")
        if abs(shift - 2.0) > 0.05:
            failures.append(f"t={t}: spliced ego history did not reach the rollout "
                            f"(start moved {shift:.2f} m for a 2.00 m displacement)")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("OK: online SAC roller anchors on the live ego pose and gives a diverse population")


if __name__ == "__main__":
    main()
