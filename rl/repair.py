"""Post-hoc repair of banked pseudo-GT trajectories.

Why post-hoc rather than in the reward
--------------------------------------
Measured 2026-07-17: a `yaw_rate_penalty` at -0.5 costs ~16 points of `reached_rate` and
moves `stored_frac_over_limit` from ~0.92 to 0.927 -- i.e. all cost, no benefit. At -2.0 it
halves goal-reaching (0.66 -> 0.30) and *still* does not fix the chatter. The shimmy is a
period-2 limit cycle in the policy's steering; it is far cheaper to remove it from the
artifact than to train it out of the policy.

What the shimmy is
------------------
Yaw flips +-25 deg every frame around a stable mean while the body tracks a smooth path
(banked scenario_00197: peak yaw rate 4.4 rad/s vs the human's 1.09 ceiling over the whole
218-scene failure set). Waymax's `InvertibleBicycleModel` forces `vel = speed * (cos yaw,
sin yaw)`, so the oscillation is *also* baked into the path as a +-0.3 m lateral wobble --
which is why repairing yaw alone can only go so far, and why the path-space repairs below
exist.

Strategies
----------
* ``yaw_refit``   -- yaw[t] = heading of p[t-k] -> p[t+k]. The obvious repair: keeps the
                     path exactly, re-derives heading from it.
* ``yaw_savgol``  -- savgol-filter the unwrapped yaw. The chatter sits at the Nyquist
                     frequency of the 10 Hz sim, which is precisely what a savgol window
                     erases (V-Max's own ComfortMetric reads a 4.49 rad/s chatterer as
                     0.06 rad/s through this filter -- normally a blind spot, here a tool).
* ``path_savgol`` -- savgol the path itself, then re-derive yaw and speed from it. The only
                     one that can remove the wobble the bicycle model baked into position.

All of them re-derive ``vel_x``/``vel_y`` from the repaired yaw and speed, because the
diffusion target carries x, y, vel_x, vel_y AND yaw -- leaving velocity pointing along the
old chattering heading would hand the model a state that contradicts itself.

Repairing yaw rotates the SDC's bounding box (up to ~12 deg), so a repaired trajectory is
NOT certified clean by the original rollout's metrics: re-score offroad/overlap with
``rl.evaluate_pseudo_gt`` before distilling.
"""

from __future__ import annotations

import numpy as np

SIM_DT_S = 0.1
# Below this displacement the direction of p[t+k] - p[t-k] is sensor-grade noise, not a
# heading: a parked car's refit flips 180 deg between frames and reads 31.4 rad/s. The
# human log does this too, so it is a property of the estimator, not of the policy.
MIN_DISP_M = 0.20


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _unwrap(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw, axis=-1)


def _savgol(x_bt: np.ndarray, window: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """Minimal savgol along the last axis; edges shrink the window instead of padding.

    scipy is not a dependency of this repo's runtime env, and `savgol_filter_jax` lives
    behind V-Max's imports -- this keeps `rl.repair` importable from a bare numpy session.
    """
    n = x_bt.shape[-1]
    half = window // 2
    out = np.empty_like(x_bt, dtype=np.float64)
    for t in range(n):
        a, b = max(t - half, 0), min(t + half, n - 1)
        idx = np.arange(a, b + 1)
        order = min(polyorder, len(idx) - 1)
        # Fit in a window-local time base so the design matrix stays well conditioned.
        coef = np.polynomial.polynomial.polyfit(idx - t, x_bt[..., a:b + 1].T, order).T
        if deriv == 0:
            out[..., t] = coef[..., 0]
        else:
            out[..., t] = coef[..., 1] if order >= 1 else 0.0
    return out


def _speed_of(traj: dict[str, np.ndarray]) -> np.ndarray:
    return np.hypot(np.asarray(traj["vel_x"], float), np.asarray(traj["vel_y"], float))


def _with_velocity_from(traj: dict[str, np.ndarray], yaw: np.ndarray, speed: np.ndarray) -> dict[str, np.ndarray]:
    """Rebuild the trajectory with `yaw`, and velocity made collinear with it.

    Collinearity is what the bicycle model enforces every step anyway, so this is the
    state the simulator would have produced had the policy steered smoothly.
    """
    out = dict(traj)
    out["yaw"] = yaw.astype(np.float32)
    out["vel_x"] = (speed * np.cos(yaw)).astype(np.float32)
    out["vel_y"] = (speed * np.sin(yaw)).astype(np.float32)
    return out


def repair_yaw_refit(traj, k: int = 2, min_disp: float = MIN_DISP_M, start_t: int = 0):
    """yaw[t] = heading from p[t-k] to p[t+k]; hold the last heading when barely moving."""
    xy = np.stack([np.asarray(traj["x"], float), np.asarray(traj["y"], float)], -1)
    yaw = np.asarray(traj["yaw"], float).copy()
    n = xy.shape[1]
    for b in range(xy.shape[0]):
        last = yaw[b, max(start_t - 1, 0)]
        for t in range(start_t, n):
            a, c = max(t - k, start_t), min(t + k, n - 1)
            d = xy[b, c] - xy[b, a]
            if np.hypot(d[0], d[1]) >= min_disp:
                last = np.arctan2(d[1], d[0])
            yaw[b, t] = last
    return _with_velocity_from(traj, yaw, _speed_of(traj))


def repair_yaw_savgol(traj, window: int = 5, polyorder: int = 2, start_t: int = 0):
    """Low-pass the yaw itself. The chatter is at Nyquist; the filter removes exactly it."""
    yaw = np.asarray(traj["yaw"], float).copy()
    sm = _wrap(_savgol(_unwrap(yaw[:, start_t:]), window, polyorder))
    yaw[:, start_t:] = sm
    return _with_velocity_from(traj, yaw, _speed_of(traj))


def repair_path_savgol(traj, window: int = 5, polyorder: int = 2, k: int = 1,
                       min_disp: float = MIN_DISP_M, start_t: int = 0):
    """Smooth the PATH, then re-derive yaw and speed from it.

    The only repair that reaches the wobble the bicycle model wrote into position. Moves
    the SDC (a little), so the position deviation is worth checking before trusting it.
    """
    out = dict(traj)
    x = np.asarray(traj["x"], float).copy()
    y = np.asarray(traj["y"], float).copy()
    x[:, start_t:] = _savgol(x[:, start_t:], window, polyorder)
    y[:, start_t:] = _savgol(y[:, start_t:], window, polyorder)
    out["x"], out["y"] = x.astype(np.float32), y.astype(np.float32)

    # Heading from the smoothed path...
    out = repair_yaw_refit(out, k=k, min_disp=min_disp, start_t=start_t)
    # ...and speed from it too, so positions, heading and velocity all tell one story.
    d = np.diff(np.stack([x, y], -1), axis=1)
    seg = np.hypot(d[..., 0], d[..., 1]) / SIM_DT_S
    speed = np.concatenate([seg[:, :1], seg], axis=1)
    return _with_velocity_from(out, np.asarray(out["yaw"], float), speed)


REPAIRS = {
    "yaw_refit_k1": lambda t, s: repair_yaw_refit(t, k=1, start_t=s),
    "yaw_refit_k2": lambda t, s: repair_yaw_refit(t, k=2, start_t=s),
    "yaw_savgol_w5": lambda t, s: repair_yaw_savgol(t, window=5, start_t=s),
    "yaw_savgol_w7": lambda t, s: repair_yaw_savgol(t, window=7, start_t=s),
    "path_savgol_w5": lambda t, s: repair_path_savgol(t, window=5, start_t=s),
    "path_savgol_w7": lambda t, s: repair_path_savgol(t, window=7, start_t=s),
}
