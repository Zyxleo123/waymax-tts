"""Load / convert offline SAC traj banks for ES ``sac`` / ``sac_safe`` init."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


SCRATCH_PREFIXES = ("/zfsauton/scratch/", "/scratch/")


def _require_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    s = str(resolved)
    if not any(s.startswith(p) for p in SCRATCH_PREFIXES):
        raise SystemExit(
            f"Refusing to load SAC bank outside scratch: {resolved}\n"
            f"Pass --sac_bank_path under /zfsauton/scratch/..."
        )
    return resolved


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_xy(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Rotate xy by -heading. heading broadcasts over leading dims of xy."""
    c = np.cos(-heading)[..., None]
    s = np.sin(-heading)[..., None]
    x = xy[..., 0]
    y = xy[..., 1]
    xr = x * c[..., 0] - y * s[..., 0]
    yr = x * s[..., 0] + y * c[..., 0]
    return np.stack([xr, yr], axis=-1)


def _interp_traj(traj_t5: np.ndarray, src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Linear interp of [T,5]=x,y,yaw,vx,vy onto query times; yaw unwrapped."""
    t = np.asarray(src_t, dtype=np.float64)
    q = np.asarray(query_t, dtype=np.float64)
    traj = np.asarray(traj_t5, dtype=np.float64)
    if traj.shape[0] == 0:
        return np.zeros((len(q), 5), dtype=np.float32)
    if traj.shape[0] < 2:
        return np.repeat(traj[:1], len(q), axis=0).astype(np.float32)
    yaw = np.unwrap(traj[:, 2])
    out = np.zeros((len(q), 5), dtype=np.float64)
    out[:, 0] = np.interp(q, t, traj[:, 0])
    out[:, 1] = np.interp(q, t, traj[:, 1])
    out[:, 2] = _wrap_to_pi(np.interp(q, t, yaw))
    out[:, 3] = np.interp(q, t, traj[:, 3])
    out[:, 4] = np.interp(q, t, traj[:, 4])
    return out.astype(np.float32)


@dataclass
class SacInitBank:
    """In-memory SAC init bank keyed by (tf_shard, tf_idx)."""
    traj: np.ndarray          # [N,K,T,5] x,y,yaw,vx,vy
    safe: np.ndarray          # [N,K] bool
    length: np.ndarray        # [N,K] int
    dt: float
    key_to_row: Dict[Tuple[str, int], int]
    scenario_ids: Dict[Tuple[str, int], Optional[str]]
    # Scenario timestep that bank index 0 corresponds to. V-Max resets the eval
    # env after the 11-frame history, so the first pose is scenario t=10, not
    # t=0 — verified at 0.00 m against the log ego pose for all 29 rows. Reading
    # the bank as if index 0 were t=0 puts the anchor a full second ahead of the
    # ego (~20 m at speed), so every candidate starts with a jump.
    start_timestep: int = 10

    @property
    def k(self) -> int:
        return int(self.traj.shape[1])

    def get(self, shard: str, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (str(shard).zfill(5) if str(shard).isdigit() else str(shard), int(idx))
        # also try without zfill
        if key not in self.key_to_row:
            key = (str(shard), int(idx))
        if key not in self.key_to_row:
            raise KeyError(
                f"SAC bank missing scene shard={shard} idx={idx}. "
                f"Known keys sample: {list(self.key_to_row)[:5]}..."
            )
        row = self.key_to_row[key]
        return self.traj[row], self.safe[row], self.length[row]


def load_sac_init_bank(bank_path: str | Path, manifest_path: str | Path) -> SacInitBank:
    bank_path = _require_scratch(Path(bank_path))
    manifest_path = Path(manifest_path)
    blob = np.load(bank_path, allow_pickle=False)
    traj = np.asarray(blob["traj"], dtype=np.float32)
    safe = np.asarray(blob["safe"]).astype(bool)
    length = np.asarray(blob["length"], dtype=np.int32)
    dt = float(np.asarray(blob["dt"]).reshape(-1)[0])
    rec_idx = np.asarray(blob["es_scenes_record_index"], dtype=np.int32)

    manifest = json.loads(manifest_path.read_text())
    by_rec = {}
    for r in manifest["records"]:
        if r.get("es_scenes_record_index") is not None:
            by_rec[int(r["es_scenes_record_index"])] = r

    key_to_row: Dict[Tuple[str, int], int] = {}
    scenario_ids: Dict[Tuple[str, int], Optional[str]] = {}
    for row, ri in enumerate(rec_idx.tolist()):
        man = by_rec.get(int(ri))
        if man is None:
            continue
        shard = str(man["tf_example_shard"])
        idx = int(man["tf_example_idx"])
        key = (shard, idx)
        key_to_row[key] = row
        scenario_ids[key] = man.get("scenario_id")

    if not key_to_row:
        raise SystemExit(f"No (shard,idx) keys built from {bank_path} + {manifest_path}")
    start_timestep = (int(np.asarray(blob["start_timestep"]).reshape(-1)[0])
                      if "start_timestep" in blob.files else 10)
    return SacInitBank(
        traj=traj, safe=safe, length=length, dt=dt,
        key_to_row=key_to_row, scenario_ids=scenario_ids,
        start_timestep=start_timestep,
    )


def _nearest_time_on_traj(
    traj_t5: np.ndarray,
    src_t: np.ndarray,
    query_xy: np.ndarray,
    t_nominal: float,
    *,
    search_radius_s: float,
    n_samples: int = 41,
) -> float:
    """Time on ``traj_t5`` whose position is closest to ``query_xy``.

    Searched within ``t_nominal +/- search_radius_s`` (not the whole path) so a
    trajectory that loops back on itself can't lock onto an unrelated crossing.
    """
    lo = max(float(src_t[0]), t_nominal - search_radius_s)
    hi = min(float(src_t[-1]), t_nominal + search_radius_s)
    if hi <= lo:
        return min(max(t_nominal, float(src_t[0])), float(src_t[-1]))
    cand_t = np.linspace(lo, hi, n_samples)
    cand_xy = np.stack(
        [np.interp(cand_t, src_t, traj_t5[:, 0]), np.interp(cand_t, src_t, traj_t5[:, 1])],
        axis=-1,
    )
    d2 = np.sum((cand_xy - np.asarray(query_xy, dtype=np.float64)[None, :]) ** 2, axis=-1)
    return float(cand_t[int(np.argmin(d2))])


def sac_slice_to_ego_norm(
    traj_kt5: np.ndarray,
    lengths_k: np.ndarray,
    *,
    timestep: int,
    origin_xy: np.ndarray,
    anchor_yaw: float,
    predict_horizon: int,
    model_dt: float,
    ego_range: float,
    max_velocity: float,
    bank_dt: float,
    bank_start_timestep: int = 10,
    reanchor_search_s: float = 2.0,
) -> np.ndarray:
    """Convert K world trajs → ego-norm predictions ``[K, H, 5]`` = x,y,vx,vy,yaw.

    Uses absolute world poses from the offline SAC path, re-anchored into the
    *current* ES ego frame (``origin_xy``, ``anchor_yaw``).

    The bank's own clock (``timestep - bank_start_timestep``) only tells you
    the right point on the path if the ego has been following it since
    ``bank_start_timestep``. Once a replan has picked a non-SAC candidate (e.g.
    ES found nothing it scored as safe), the live ego pose drifts off the
    bank's schedule. Re-slicing "the SAC trajectory" at the *elapsed-time*
    point then, when ``postprocess_predictions`` splices the live anchor pose
    onto it and resamples, produces a straight-line cut from "where the ego
    actually is" to "where the schedule says the path is" — a maneuver that
    was never part of the recorded, safety-checked trajectory and can clip
    something the real path never touched. Re-anchoring to the *nearest point
    on the path* (searched in a bounded window around the nominal elapsed
    time, so a self-crossing path can't hijack the search) keeps that splice
    as small as the live divergence actually is, instead of compounding it
    with schedule drift.
    """
    K = traj_kt5.shape[0]
    H = int(predict_horizon)
    origin_xy = np.asarray(origin_xy, dtype=np.float32).reshape(2)
    yaw0 = float(anchor_yaw)
    out = np.zeros((K, H, 5), dtype=np.float32)
    pred_times = (np.arange(H, dtype=np.float64) + 1.0) * float(model_dt)
    # bank index 0 == scenario step ``bank_start_timestep``
    t_nominal = max(0.0, (float(timestep) - float(bank_start_timestep)) * float(bank_dt))

    for k in range(K):
        L = int(lengths_k[k])
        L = max(L, 1)
        traj = traj_kt5[k, :L]  # [L,5] x,y,yaw,vx,vy
        src_t = np.arange(L, dtype=np.float64) * float(bank_dt)
        t0 = (_nearest_time_on_traj(
                  traj, src_t, origin_xy, t_nominal, search_radius_s=reanchor_search_s)
              if L > 1 else t_nominal)
        fut = _interp_traj(traj, src_t, t0 + pred_times)  # [H,5]
        # world → ego-norm (match world_to_ego_normalized channel order)
        xy = fut[:, :2]
        yaw = fut[:, 2]
        vel = fut[:, 3:5]
        pos = _rotate_xy(xy - origin_xy[None, :], np.asarray(yaw0)) / float(ego_range)
        vel_e = _rotate_xy(vel, np.asarray(yaw0)) / float(max_velocity)
        yaw_e = _wrap_to_pi(yaw - yaw0) / np.pi
        out[k] = np.concatenate([pos, vel_e, yaw_e[:, None]], axis=-1).astype(np.float32)
    return out


def select_sac_population(
    traj_kt5: np.ndarray,
    safe_k: np.ndarray,
    lengths_k: np.ndarray,
    *,
    population_size: int,
    sac_safe: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pick K init members from the SAC bank (safe-first if ``sac_safe``)."""
    K_bank = traj_kt5.shape[0]
    K = int(population_size)
    safe_k = np.asarray(safe_k).astype(bool)
    n_safe = int(safe_k.sum())

    if sac_safe and n_safe > 0:
        safe_idx = np.flatnonzero(safe_k)
        if len(safe_idx) >= K:
            idx = safe_idx[:K]
        else:
            rest = np.flatnonzero(~safe_k)
            n_fill = K - len(safe_idx)
            fill = rest[:n_fill] if n_fill > 0 else np.array([], dtype=np.int64)
            idx = np.concatenate([safe_idx, fill])
    else:
        # use bank as-is (including all-zero safe scenes)
        if K_bank >= K:
            idx = np.arange(K, dtype=np.int64)
        else:
            # repeat if somehow undersized
            idx = np.resize(np.arange(K_bank), K).astype(np.int64)

    idx = np.asarray(idx, dtype=np.int64)
    meta = {
        "init_bank_size": int(K_bank),
        "init_bank_num_safe": n_safe,
        "init_bank_num_modes": None,
        "init_selected_num_modes": None,
        "init_selected_num_safe": int(safe_k[idx].sum()),
    }
    return traj_kt5[idx], safe_k[idx], lengths_k[idx], meta
