"""Repair the (row -> scene) labelling of an existing SAC init bank.

``dump_sac_init_bank.py`` passed ``seed=`` to ``make_data_generator``, which V-Max
forwards as waymax's ``shuffle_seed`` — so the generator yielded scenes shuffled
while the loop stamped each row with ``manifest[enumerate_index]``. The rollouts
themselves are fine; only the labels are permuted.

This script re-derives the true label of every row geometrically: a SAC rollout
of scene S starts exactly at S's log ego pose at ``start_timestep`` (verified at
0.00 m). It matches each row's first pose against the log ego poses of the
candidate scenes, requires a unique bijection within ``--tol``, and writes a
corrected copy of the bank (trajectories untouched, ``es_scenes_record_index``
rewritten, ``start_timestep`` recorded).

CPU only — it reads tfrecords and an npz, no model.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import tensorflow as _tf

    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from data.scenario_loader import load_scenario_state_batch_fast
from waymax import config as waymax_config

SCRATCH_PREFIXES = ("/zfsauton/scratch/", "/scratch/")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True, help="sac_init_bank.npz to repair")
    p.add_argument("--manifest", required=True, help="es_scenes_manifest.json")
    p.add_argument("--tfrecord_root", default=str(REPO_ROOT / "repro"),
                   help="dir holding tf<shard>/training/<tfrecord>")
    p.add_argument("--out", default=None,
                   help="output npz (default: <bank stem>_remapped.npz, must be on scratch)")
    p.add_argument("--start_timestep", type=int, default=10,
                   help="scenario timestep that bank index 0 corresponds to")
    p.add_argument("--tol", type=float, default=1.0,
                   help="max metres between a row's first pose and its scene's log ego pose")
    p.add_argument("--max_num_objects", type=int, default=128)
    return p.parse_args()


def _require_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not any(str(resolved).startswith(pfx) for pfx in SCRATCH_PREFIXES):
        raise SystemExit(f"Refusing to write outside scratch: {resolved}")
    return resolved


def _log_ego_xy(tfrecord_path: str, indices: list[int], t: int, max_num_objects: int) -> dict:
    """Log (ground-truth) ego xy at scenario timestep ``t`` for each index."""
    out = {}
    for idx in indices:
        cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=str(tfrecord_path), max_num_objects=int(max_num_objects),
            batch_dims=(1,), shuffle_seed=None)
        state, _ = load_scenario_state_batch_fast(cfg, [idx])
        is_sdc = np.asarray(state.object_metadata.is_sdc)[0]
        sdc = int(np.argmax(is_sdc.astype(np.int32)))
        lt = state.log_trajectory
        out[idx] = np.asarray([float(np.asarray(lt.x)[0, sdc, t]),
                               float(np.asarray(lt.y)[0, sdc, t])], dtype=np.float64)
    return out


def main() -> None:
    args = _parse()
    bank_path = Path(args.bank)
    out_path = _require_scratch(Path(args.out) if args.out
                                else bank_path.with_name(bank_path.stem + "_remapped.npz"))

    blob = np.load(bank_path, allow_pickle=False)
    traj = np.asarray(blob["traj"], dtype=np.float32)
    rec_idx = np.asarray(blob["es_scenes_record_index"], dtype=np.int32)
    manifest = json.loads(Path(args.manifest).read_text())
    by_rec = {int(r["es_scenes_record_index"]): r for r in manifest["records"]
              if r.get("es_scenes_record_index") is not None}

    # candidate scenes = every scene the bank claims to cover
    scenes = []
    for ri in rec_idx.tolist():
        m = by_rec.get(int(ri))
        if m is None:
            raise SystemExit(f"record index {ri} missing from manifest")
        scenes.append((str(m["tf_example_shard"]), int(m["tf_example_idx"]), int(ri)))
    print(f"bank rows: {len(scenes)}  covering shards: "
          f"{sorted({s for s, _, _ in scenes})}")

    # log ego pose per scene at start_timestep
    truth = {}
    for shard in sorted({s for s, _, _ in scenes}):
        idxs = sorted({i for s, i, _ in scenes if s == shard})
        tfdir = Path(args.tfrecord_root) / f"tf{shard}" / "training"
        files = sorted(tfdir.glob("*"))
        if not files:
            raise SystemExit(f"no tfrecord under {tfdir}")
        print(f"  loading {len(idxs)} scenes from {files[0].name} ...", flush=True)
        for idx, xy in _log_ego_xy(str(files[0]), idxs, args.start_timestep,
                                   args.max_num_objects).items():
            truth[(shard, idx)] = xy

    # geometric assignment: row -> scene whose log ego pose matches its first pose
    keys = list(truth)
    assign, report = {}, []
    for row in range(traj.shape[0]):
        p0 = traj[row, 0, 0, :2].astype(np.float64)
        d = {k: float(np.linalg.norm(p0 - truth[k])) for k in keys}
        best = min(d, key=d.get)
        second = sorted(d.values())[1] if len(d) > 1 else float("inf")
        labelled = (scenes[row][0], scenes[row][1])
        assign[row] = best
        report.append({"row": row, "labelled": f"{labelled[0]}/{labelled[1]}",
                       "true": f"{best[0]}/{best[1]}", "dist_m": d[best],
                       "runner_up_m": second, "was_correct": bool(best == labelled)})
        if d[best] > args.tol:
            raise SystemExit(
                f"row {row}: nearest scene {best} is {d[best]:.2f} m away (> --tol {args.tol}); "
                f"refusing to guess")
        if second < 5.0:
            raise SystemExit(
                f"row {row}: ambiguous — runner-up scene only {second:.2f} m away")

    if len(set(assign.values())) != len(assign):
        raise SystemExit("assignment is not a bijection; refusing to write")

    key_to_rec = {(s, i): ri for s, i, ri in scenes}
    new_rec = np.asarray([key_to_rec[assign[row]] for row in range(traj.shape[0])],
                         dtype=np.int32)
    n_fixed = sum(1 for r in report if not r["was_correct"])
    print(f"\nrows relabelled: {n_fixed}/{len(report)}  "
          f"(max match distance {max(r['dist_m'] for r in report):.3f} m)")

    payload = {k: blob[k] for k in blob.files}
    payload["es_scenes_record_index"] = new_rec
    payload["start_timestep"] = np.int32(args.start_timestep)
    np.savez_compressed(out_path, **payload)
    Path(str(out_path) + ".remap.json").write_text(json.dumps(
        {"source_bank": str(bank_path), "start_timestep": args.start_timestep,
         "rows_relabelled": n_fixed, "assignments": report}, indent=2))
    print(f"wrote {out_path}\nwrote {out_path}.remap.json")

    uniq = min(int(len(np.unique(np.round(traj[r].reshape(traj.shape[1], -1), 3), axis=0)))
               for r in range(traj.shape[0]))
    if uniq < traj.shape[1]:
        print(f"\nNOTE: minimum unique members per row is {uniq} of K={traj.shape[1]}. "
              f"Labels are fixed but the bank still has duplicate rollouts — rebuild with "
              f"dump_sac_init_bank.py --noisy_init for real diversity.")


if __name__ == "__main__":
    main()
