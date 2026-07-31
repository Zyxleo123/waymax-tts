"""Refit the SDC log trajectory so the bicycle dynamics can reproduce it.

Motivation (see ``rl/CONTEXT.md`` -- "BREAKING FINDING"): V-Max / Waymax drive
the ego through :class:`waymax.dynamics.InvertibleBicycleModel`, while every
non-ego object is *log-replayed* (placed exactly on its recorded pose each
step). Raw WOMD logs are **not** kinematically consistent with the bicycle
model:

* the recorded velocity vector ``(vel_x, vel_y)`` is not collinear with the
  recorded heading ``yaw`` (sideslip / sensor noise), but the forward model
  *forces* ``vel = speed * (cos yaw, sin yaw)`` every step, and
* the analytic steering inverse ``delta_yaw / (speed*dt + ...)`` is
  ill-conditioned at low speed, so near-parked cars get garbage / clipped
  steering.

Re-simulating the human log through the bicycle inverse therefore *drifts* the
ego off its logged path (bimodal: half track to <3 cm, half diverge several
metres). Once it drifts it hits the frozen log-replayed agents the human
avoided by exact timing, so the reward signal is polluted by "ghost"
collisions / offroads that **no** policy can avoid. SAC then converges to
"coast" and BC clones a crashing pipeline.

This module fixes the artifact *at the source* -- the approach ScenarioMax /
nuPlan take: re-derive the SDC's heading and speed from its own logged
positions so the trajectory is a sequence of "move along heading toward the
next waypoint" segments. Such a trajectory is (to first order) an exact orbit
of the bicycle integrator, so:

* the analytic inverse produces small, in-range, well-conditioned actions,
* forward-integrating those actions reproduces the logged positions, and
* closed-loop expert replay stays locked to the path -> the ghost collisions
  disappear and the reward becomes learnable.

Only the **SDC** is touched. Every other object keeps its raw log (it is
log-replayed verbatim, so refitting it would change the ground-truth scene).
Positions ``(x, y, z)`` and the validity mask are preserved exactly -- only
``yaw``, ``vel_x``, ``vel_y`` are rewritten -- so the goal (the SDC's last
valid logged xy) and the scene geometry are unchanged.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.utils import geometry

# Below this per-step displacement the object is treated as stationary and its
# *original* yaw is kept (the bicycle model cannot steer below ~0.6 m/s anyway --
# see ``_SPEED_LIMIT`` in waymax.dynamics.bicycle_model). 0.6 m/s * 0.1 s.
_DEFAULT_MOVE_EPS_M = 0.06


def refit_trajectory_fields(
    traj: datatypes.Trajectory,
    *,
    dt: float = 0.1,
    move_eps_m: float = _DEFAULT_MOVE_EPS_M,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute kinematically-consistent ``(yaw, vel_x, vel_y)`` for every object.

    The refit makes the velocity collinear with the direction of travel and the
    speed equal to the inter-step displacement over ``dt``. For (near-)stationary
    steps the original yaw is preserved and the speed is taken from the original
    logged velocity magnitude (so a parked car keeps its box orientation).

    Args:
        traj: Trajectory of shape ``(..., num_objects, num_timesteps)``.
        dt: Simulator timestep (s).
        move_eps_m: Per-step displacement below which a step is "stationary".

    Returns:
        ``(yaw, vel_x, vel_y)`` arrays, each shaped like ``traj.x``.
    """
    x = traj.x
    y = traj.y
    valid = traj.valid

    # Forward differences along the time axis (last axis).
    dx = x[..., 1:] - x[..., :-1]
    dy = y[..., 1:] - y[..., :-1]
    pair_valid = valid[..., 1:] & valid[..., :-1]
    dist = jnp.hypot(dx, dy)

    # Step ``t`` is assigned the heading/speed of the displacement [t, t+1]; the
    # final step has no successor, so it inherits the previous segment's values.
    head = jnp.arctan2(dy, dx)
    speed = dist / dt
    moving = (dist > move_eps_m) & pair_valid

    head = jnp.concatenate([head, head[..., -1:]], axis=-1)
    speed = jnp.concatenate([speed, speed[..., -1:]], axis=-1)
    moving = jnp.concatenate([moving, moving[..., -1:]], axis=-1)

    orig_speed = jnp.hypot(traj.vel_x, traj.vel_y)

    new_yaw = jnp.where(moving, geometry.wrap_yaws(head), traj.yaw)
    new_speed = jnp.where(moving, speed, orig_speed)
    new_vel_x = new_speed * jnp.cos(new_yaw)
    new_vel_y = new_speed * jnp.sin(new_yaw)

    # Never touch invalid steps.
    new_yaw = jnp.where(valid, new_yaw, traj.yaw)
    new_vel_x = jnp.where(valid, new_vel_x, traj.vel_x)
    new_vel_y = jnp.where(valid, new_vel_y, traj.vel_y)
    return new_yaw, new_vel_x, new_vel_y


def refit_sdc_trajectory(
    traj: datatypes.Trajectory,
    is_sdc: jax.Array,
    *,
    dt: float = 0.1,
    move_eps_m: float = _DEFAULT_MOVE_EPS_M,
) -> datatypes.Trajectory:
    """Return ``traj`` with the SDC's ``yaw``/``vel`` made bicycle-consistent.

    Args:
        traj: Trajectory of shape ``(..., num_objects, num_timesteps)``.
        is_sdc: Boolean mask of shape ``(..., num_objects)`` selecting the SDC.
        dt: Simulator timestep (s).
        move_eps_m: Per-step displacement below which a step is "stationary".

    Returns:
        A new Trajectory; non-SDC objects and ``(x, y, z, valid)`` are untouched.
    """
    new_yaw, new_vel_x, new_vel_y = refit_trajectory_fields(
        traj, dt=dt, move_eps_m=move_eps_m
    )
    sdc = is_sdc[..., None]  # broadcast over the time axis
    return traj.replace(
        yaw=jnp.where(sdc, new_yaw, traj.yaw),
        vel_x=jnp.where(sdc, new_vel_x, traj.vel_x),
        vel_y=jnp.where(sdc, new_vel_y, traj.vel_y),
    )


def refit_state_sdc_log(
    state: Any,
    *,
    dt: float = 0.1,
    move_eps_m: float = _DEFAULT_MOVE_EPS_M,
) -> Any:
    """Refit the SDC log (and sim) trajectory of a (possibly batched) state.

    Both ``log_trajectory`` and ``sim_trajectory`` are refit so the warmup region
    the env copies log->sim on reset stays consistent. Other objects are
    log-replayed verbatim and are left untouched.

    Args:
        state: A Waymax ``SimulatorState`` (any leading batch dims are fine).
        dt: Simulator timestep (s).
        move_eps_m: Per-step displacement below which a step is "stationary".

    Returns:
        A new ``SimulatorState`` with the SDC log/sim trajectory refit.
    """
    is_sdc = state.object_metadata.is_sdc
    new_log = refit_sdc_trajectory(
        state.log_trajectory, is_sdc, dt=dt, move_eps_m=move_eps_m
    )
    new_sim = refit_sdc_trajectory(
        state.sim_trajectory, is_sdc, dt=dt, move_eps_m=move_eps_m
    )
    return state.replace(log_trajectory=new_log, sim_trajectory=new_sim)


def refit_state_batch_host(state: Any, **kwargs: Any) -> Any:
    """Eagerly refit a host-resident (NumPy) stacked ``[N, ...]`` state.

    Convenience wrapper for the data loaders, which cache an unbatched host
    pytree. Runs the (jit-able) refit once and returns the result; the caller is
    responsible for moving it to device.
    """
    return jax.jit(lambda s: refit_state_sdc_log(s, **kwargs))(state)
