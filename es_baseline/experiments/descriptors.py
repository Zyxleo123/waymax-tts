"""Behavior-level descriptors + diversity selection for Stage 2.

Represents each candidate trajectory by *behavior* properties (not raw waypoints)
so we can (a) measure how many distinct behavior modes a bank covers and
(b) select a diverse equal-sized population from an oversized mixed bank.

Descriptors (per candidate), computed in the ego's start-heading frame:
  final_long, final_lat            longitudinal / lateral end displacement
  speed_q25/q50/q75/q100           speed at 25/50/75/100% of the horizon
  min_speed, max_speed
  max_brake                        peak deceleration (m/s^2)
  max_lat                          peak lateral excursion (m)
  lane_change                      net signed lateral displacement (mode proxy)
  maneuver_time                    normalized time of peak lateral speed
  route_progress                   goal-distance reduction over the segment (m)
  min_separation                   min distance to any agent (m)  [needs per_step]
  offroad_frac                     fraction of steps off-road     [needs per_step]
  mean_accel, mean_yawrate         comfort statistics

All descriptors are plain functions of the ``[K, T, 5]`` world-frame candidate
tensor (x, y, yaw, vx, vy); the safety-dependent ones optionally read the
``per_step`` dict produced by ``DenseReturnScorer.compute_components``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

DESCRIPTOR_NAMES: List[str] = [
    "final_long", "final_lat", "speed_q25", "speed_q50", "speed_q75", "speed_q100",
    "min_speed", "max_speed", "max_brake", "max_lat", "lane_change", "maneuver_time",
    "route_progress", "min_separation", "offroad_frac", "mean_accel", "mean_yawrate",
]


def compute_descriptors(
    trajectories: np.ndarray,
    dt: float = 0.2,
    goal_xy: Optional[np.ndarray] = None,
    per_step: Optional[Dict[str, np.ndarray]] = None,
    ref_xy: Optional[np.ndarray] = None,
    ref_yaw: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Returns a dict name -> [K] array. ``per_step`` values are [K, T] arrays.

    ``ref_xy`` / ``ref_yaw`` define the SHARED longitudinal/lateral frame. All ES
    candidates start from the same ego pose, so lateral displacement is only
    meaningful in a common frame; pass the ego's current pose. If omitted, falls
    back to each candidate's own first-step pose (fine only when headings agree).
    """
    traj = np.asarray(trajectories, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[-1] != 5:
        raise ValueError(f"trajectories must be [K, T, 5], got {traj.shape}")
    K, T, _ = traj.shape
    xy = traj[:, :, :2]
    if ref_yaw is None:
        yaw0 = traj[:, 0, 2]                               # [K] per-candidate
    else:
        yaw0 = np.full(K, float(ref_yaw))                  # shared reference
    c, s = np.cos(yaw0), np.sin(yaw0)

    origin = xy[:, :1, :] if ref_xy is None else np.asarray(ref_xy, np.float64).reshape(1, 1, 2)
    rel = xy - origin                                      # [K, T, 2] from origin
    long = rel[..., 0] * c[:, None] + rel[..., 1] * s[:, None]   # [K, T]
    lat = -rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]   # [K, T]

    speed = np.linalg.norm(traj[:, :, 3:5], axis=-1)      # [K, T]
    accel = np.diff(speed, axis=1) / dt                   # [K, T-1]
    dyaw = np.arctan2(np.sin(np.diff(traj[:, :, 2], axis=1)),
                      np.cos(np.diff(traj[:, :, 2], axis=1)))
    yawrate = np.abs(dyaw) / dt                           # [K, T-1]
    lat_speed = np.abs(np.diff(lat, axis=1)) / dt         # [K, T-1]

    def q(frac):
        idx = min(T - 1, int(round(frac * (T - 1))))
        return speed[:, idx]

    d = {
        "final_long": long[:, -1],
        "final_lat": lat[:, -1],
        "speed_q25": q(0.25), "speed_q50": q(0.50),
        "speed_q75": q(0.75), "speed_q100": q(1.00),
        "min_speed": speed.min(axis=1),
        "max_speed": speed.max(axis=1),
        "max_brake": np.maximum(0.0, -accel.min(axis=1)),
        "max_lat": np.abs(lat).max(axis=1),
        "lane_change": lat[:, -1],
        "maneuver_time": (lat_speed.argmax(axis=1) / max(1, T - 1)).astype(np.float64),
        "mean_accel": np.abs(accel).mean(axis=1),
        "mean_yawrate": yawrate.mean(axis=1),
    }

    if goal_xy is not None:
        g = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        dgoal = np.linalg.norm(xy - g[None, None, :], axis=-1)   # [K, T]
        d["route_progress"] = dgoal[:, 0] - dgoal[:, -1]
    else:
        d["route_progress"] = np.zeros(K)

    if per_step is not None and "separation" in per_step:
        # per_step separation is a [0,1] margin; recover a monotone descriptor
        d["min_separation"] = np.asarray(per_step["separation"]).min(axis=1)
    else:
        d["min_separation"] = np.zeros(K)
    if per_step is not None and "offroad" in per_step:
        onroad = np.asarray(per_step["offroad"])            # 1 = on-road
        d["offroad_frac"] = 1.0 - onroad.mean(axis=1)
    else:
        d["offroad_frac"] = np.zeros(K)

    return d


def descriptor_matrix(desc: Dict[str, np.ndarray]) -> np.ndarray:
    """Stacks the descriptor dict into a [K, D] matrix in DESCRIPTOR_NAMES order."""
    return np.stack([np.asarray(desc[n], dtype=np.float64) for n in DESCRIPTOR_NAMES], axis=1)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Z-score each descriptor column (robust to zero-variance columns)."""
    m = np.asarray(matrix, dtype=np.float64)
    mu = m.mean(axis=0, keepdims=True)
    sd = m.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (m - mu) / sd


def select_diverse(matrix: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling over normalized descriptors.

    Returns indices of ``n`` trajectories that cover behavior space. Deterministic
    given ``seed`` (only the first seed point is randomized)."""
    m = normalize(matrix)
    K = m.shape[0]
    n = min(int(n), K)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, K))
    chosen = [first]
    dist = np.linalg.norm(m - m[first][None, :], axis=1)
    for _ in range(1, n):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(m - m[nxt][None, :], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def count_modes(matrix: np.ndarray, n_bins: int = 3) -> int:
    """Coarse behavior-mode count: number of occupied cells when each normalized
    descriptor is bucketed into ``n_bins``. A cheap coverage proxy for the
    no-ES diagnostic (bank size M, 2M, 4M, 8M)."""
    m = normalize(matrix)
    edges = np.linspace(-2.0, 2.0, n_bins - 1)
    codes = np.stack([np.digitize(m[:, j], edges) for j in range(m.shape[1])], axis=1)
    return int(len({tuple(row) for row in codes}))
