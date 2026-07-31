#!/usr/bin/env python3
"""Re-apply the bank's BC-quality gates to an *existing* bank, offline.

The gates (goal truncation + the yaw-rate smoothness veto) now run inside
``rl.bank.Banker`` at harvest time, but banks harvested before they existed hold
full-length, ungated trajectories. This replays the same code over them so you can
see what survives -- and, with ``--write``, produce a gated copy without re-running SAC.

The goal comes from the source tfrecord (the SDC's last valid *logged* xy, V-Max's
own definition), read in file order, which is the identity the bank is keyed on --
so ``--dataset`` must be the same tfrecord the bank was harvested from.

Report only::

    python -m rl.bank_regate --bank-dir .../bank_pretrained_probe/bank

Write a gated copy::

    python -m rl.bank_regate --bank-dir .../bank --out-dir .../bank_gated --write
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "V-Max"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rl.bank import (  # noqa: E402
    MAX_YAW_RATE_RAD_S,
    goal_cut_index,
    max_yaw_rate,
    truncate_at,
)

DEFAULT_BANK = "/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/bank_pretrained_probe/bank"
DEFAULT_DATASET = "/zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures/failures.tfrecord"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank-dir", default=DEFAULT_BANK)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Tfrecord the bank was harvested from.")
    p.add_argument("--out-dir", default=None, help="Where to write gated trajectories (with --write).")
    p.add_argument("--write", action="store_true", help="Write the gated copy; otherwise report only.")
    p.add_argument("--max-yaw-rate", type=float, default=MAX_YAW_RATE_RAD_S)
    p.add_argument("--start-t", type=int, default=11, help="First policy-controlled step (91 - scenario_length).")
    p.add_argument("--max-num-objects", type=int, default=32)
    p.add_argument("--chunk", type=int, default=4, help="Scenes per data-generator batch; keep small, this OOMs.")
    return p.parse_args()


def read_goals(path: str, num_scenarios: int, max_num_objects: int, chunk: int) -> np.ndarray:
    """[N, 2] SDC last-valid logged xy, in file order (seed=None -> no shuffle)."""
    from vmax.simulator import make_data_generator

    gen = make_data_generator(
        path=path,
        max_num_objects=max_num_objects,
        include_sdc_paths=False,
        seed=None,
        batch_dims=(chunk,),
        repeat=None,
    )
    goals = []
    for _ in range(math.ceil(num_scenarios / chunk)):
        batch = next(gen)
        log = batch.log_trajectory
        sdc = np.argmax(np.asarray(batch.object_metadata.is_sdc), axis=-1)
        rows = np.arange(len(sdc))
        xy = np.asarray(log.xy)[rows, sdc]
        valid = np.asarray(log.valid)[rows, sdc]
        last = np.max(np.where(valid, np.arange(valid.shape[1]), -1), axis=1)
        goals.append(xy[rows, np.maximum(last, 0)])
        del batch, log
    return np.concatenate(goals)[:num_scenarios]


def main() -> int:
    args = _parse_args()
    bank_dir = Path(args.bank_dir)
    manifest = json.load(open(bank_dir / "bank_manifest.json"))
    banked = sorted(int(k) for k in manifest["banked"])
    num_scenarios = int(manifest["num_scenarios"])

    goals = read_goals(args.dataset, num_scenarios, args.max_num_objects, args.chunk)

    keys = ("x", "y", "yaw", "vel_x", "vel_y", "valid")
    trajs = {k: [] for k in keys}
    for i in banked:
        d = np.load(bank_dir / "trajectories" / f"scenario_{i:05d}.npz")
        for k in keys:
            trajs[k].append(np.asarray(d[k]))
    traj = {k: np.stack(v) for k, v in trajs.items()}
    goal_b = goals[np.array(banked)]

    dist = np.linalg.norm(np.stack([traj["x"], traj["y"]], -1) - goal_b[:, None, :], axis=-1)
    cut = goal_cut_index(dist, traj["valid"], start_t=args.start_t)
    yr = max_yaw_rate(traj["yaw"], cut, start_t=args.start_t)
    smooth = yr <= args.max_yaw_rate
    gated = truncate_at(traj, cut)

    n = len(banked)
    steps = np.array([manifest["banked"][str(i)]["step"] for i in banked])
    kept_frames = gated["valid"][:, args.start_t:].sum(axis=1)
    orig_frames = traj["valid"][:, args.start_t:].sum(axis=1)
    final_d = dist[np.arange(n), -1]
    cut_d = dist[np.arange(n), cut]

    print(f"bank: {bank_dir}\n  {n} banked / {num_scenarios} scenarios\n")
    print(f"TRUNCATION (cut at closest approach, start_t={args.start_t})")
    print(f"  distance to goal at the cut : p50 {np.median(cut_d):5.2f} m   max {cut_d.max():5.2f} m")
    print(f"  distance to goal at old end : p50 {np.median(final_d):5.2f} m   max {final_d.max():5.2f} m")
    print(f"  frames kept                 : p50 {int(np.median(kept_frames))} of {int(np.median(orig_frames))}"
          f"   ({100 * (1 - kept_frames.sum() / orig_frames.sum()):.0f}% of policy frames dropped)")
    print()
    print(f"SMOOTHNESS GATE (max |yaw rate| <= {args.max_yaw_rate} rad/s)")
    print(f"  max yaw rate: p50 {np.median(yr):5.2f}   p90 {np.percentile(yr, 90):5.2f}   max {yr.max():5.2f} rad/s")
    print(f"  PASS {int(smooth.sum()):3d}/{n}   FAIL {int((~smooth).sum()):3d}/{n}"
          f"   ({100 * (~smooth).mean():.0f}% of the harvest would be refused)")
    print()
    print("  by banked step:")
    for s in sorted(set(steps.tolist())):
        m = steps == s
        print(f"    step {int(s):6d}: {int(m.sum()):3d} banked, {int(smooth[m].sum()):3d} pass, "
              f"{int((~smooth[m]).sum()):3d} fail ({100 * (~smooth[m]).mean():3.0f}%)")

    if args.write:
        out_dir = Path(args.out_dir or (str(bank_dir) + "_gated"))
        traj_dir = out_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        kept = {}
        for j, i in enumerate(banked):
            if not smooth[j]:
                continue
            np.savez_compressed(traj_dir / f"scenario_{i:05d}.npz", **{k: v[j] for k, v in gated.items()})
            kept[str(i)] = dict(manifest["banked"][str(i)])
            kept[str(i)].update(
                path=str(traj_dir / f"scenario_{i:05d}.npz"),
                cut_step=int(cut[j]),
                # Same key/statistic TrajectoryBank.update compares on, so a run resumed
                # against this bank upgrades these entries instead of treating them as
                # quality-less (which would let the next clean rollout overwrite them
                # even if it were worse).
                quality=float(yr[j]),
            )
        payload = {
            "num_scenarios": num_scenarios,
            "num_banked": len(kept),
            "banked": kept,
            "remaining": [i for i in range(num_scenarios) if str(i) not in kept],
            "regated_from": str(bank_dir),
            "gate": {"max_yaw_rate_rad_s": args.max_yaw_rate, "start_t": args.start_t, "truncated": True},
        }
        tmp = out_dir / "bank_manifest.json.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, out_dir / "bank_manifest.json")
        print(f"\nwrote {len(kept)} gated + truncated trajectories -> {out_dir}")
    else:
        print("\n(report only; pass --write to produce a gated copy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
