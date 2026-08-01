# Copyright 2025 Valeo.


"""Operations for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.datatypes import route


# Distance (meters) below which the SDC is considered to have reached its goal.
# Matches simulation.evaluation_utils.check_goal_reaching (failure-case evaluation).
GOAL_THRESHOLD_M: float = 2.0


def get_index(x: jnp.ndarray, k: int = 1, squeeze: bool = True) -> jnp.ndarray:
    """Get the index of the maximum value in an array.

    Args:
        x: Input array.
        k: Number of top values to return.
        squeeze: Whether to squeeze the output.

    Returns:
        The index of the maximum value.

    """
    if k == 1:
        idx = jnp.argmax(x, keepdims=not squeeze)
    else:
        idx = jax.lax.top_k(x, k)[1]

        if squeeze:
            return idx.squeeze()

    return idx


def select_longest_sdc_path_id(sdc_paths: route.Paths) -> int:
    """Select the index of the longest SDC path that covers all trajectory points.

    Args:
        sdc_paths: Paths object with route information.

    Returns:
        Index of the longest SDC path.

    """
    # (1, num_paths, 1)
    on_route = sdc_paths.on_route
    # (1, num_paths, num_points_per_path)
    on_route = jnp.repeat(on_route, sdc_paths.num_points_per_path, axis=-1)
    # (1, num_paths, num_points_per_path)
    mask = jnp.logical_and(on_route, sdc_paths.valid)
    longest_path_idx = jnp.argmax(jnp.sum(mask, axis=-1))

    return longest_path_idx


def get_sdc_goal_xy(state: datatypes.SimulatorState, sdc_idx: jnp.ndarray) -> jnp.ndarray:
    """Return the SDC's goal position: its last valid logged (x, y) position.

    The goal point comes natively from the logged (expert) trajectory in the
    ScenarioMax/Waymax data: it is the final valid position the SDC reaches in the
    recording. This is the single source of truth shared by the goal-reaching
    reward (``wrappers.reward``) and the goal-reaching metric (``BraxWrapper.metrics``).

    Args:
        state: Current simulator state.
        sdc_idx: Index of the SDC in the object dimension.

    Returns:
        The goal (x, y) position as an array of shape [2].
    """
    log_xy = state.log_trajectory.xy[sdc_idx]  # [num_timesteps, 2]
    log_valid = state.log_trajectory.valid[sdc_idx]  # [num_timesteps]

    step_indices = jnp.arange(log_valid.shape[0])
    last_valid_idx = jnp.max(jnp.where(log_valid, step_indices, -1))
    last_valid_idx = jnp.maximum(last_valid_idx, 0)

    return log_xy[last_valid_idx]  # [2]


def get_sdc_pose_xy_yaw(state: datatypes.SimulatorState, sdc_idx: jnp.ndarray) -> tuple[jax.Array, jax.Array]:
    """Return the SDC's current simulated (x, y) position and yaw.

    Args:
        state: Current simulator state.
        sdc_idx: Index of the SDC in the object dimension.

    Returns:
        A tuple of (xy [2], yaw scalar).
    """
    traj = state.current_sim_trajectory

    xy = jnp.stack([traj.x[sdc_idx].squeeze(), traj.y[sdc_idx].squeeze()])
    yaw = traj.yaw[sdc_idx].squeeze()

    return xy, yaw


def get_distance_to_goal(state: datatypes.SimulatorState) -> jax.Array:
    """Return the Euclidean distance (meters) from the SDC to its goal.

    Args:
        state: Current simulator state.

    Returns:
        Scalar distance in meters.
    """
    sdc_idx = get_index(state.object_metadata.is_sdc)
    goal_xy = get_sdc_goal_xy(state, sdc_idx)
    current_xy, _ = get_sdc_pose_xy_yaw(state, sdc_idx)

    return jnp.linalg.norm(current_xy - goal_xy)


def get_initial_distance_to_goal(state: datatypes.SimulatorState) -> jax.Array:
    """Return the distance (meters) from the SDC's *initial* position to its goal.

    "Initial" is the first simulated step of the episode (``init_steps - 1``, i.e. the
    last logged step before the policy takes over). This is a per-scenario constant,
    used to normalize goal features and the goal potential so that a 10 m parking
    maneuver and a 100 m highway run are on the same scale.

    Args:
        state: Current simulator state.

    Returns:
        Scalar distance in meters, floored at 1.0 to stay safe under division.
    """
    sdc_idx = get_index(state.object_metadata.is_sdc)
    goal_xy = get_sdc_goal_xy(state, sdc_idx)

    # The sim trajectory is filled with the logged history up to the episode start,
    # so the earliest valid simulated step is the SDC's position at t=0 of the episode.
    sim_xy = state.sim_trajectory.xy[sdc_idx]  # [num_timesteps, 2]
    sim_valid = state.sim_trajectory.valid[sdc_idx]  # [num_timesteps]

    step_indices = jnp.arange(sim_valid.shape[0])
    first_valid_idx = jnp.min(jnp.where(sim_valid, step_indices, sim_valid.shape[0] - 1))

    initial_distance = jnp.linalg.norm(sim_xy[first_valid_idx] - goal_xy)

    return jnp.maximum(initial_distance, 1.0)


def get_remaining_time(state: datatypes.SimulatorState, dt: float = 0.1) -> jax.Array:
    """Return the time (seconds) left in the episode before the SDC runs out of horizon.

    The goal is the SDC's position at the *final* logged step, so "reaching the goal"
    is really "cover the expert's displacement within the expert's time budget". The
    policy therefore needs the remaining time to decide whether to speed up: without
    it the success condition depends on state the policy cannot observe.

    Args:
        state: Current simulator state.
        dt: Simulation timestep in seconds.

    Returns:
        Scalar time in seconds, floored at ``dt`` to stay safe under division.
    """
    return jnp.maximum(state.remaining_timesteps.astype(jnp.float32) * dt, dt)


def is_within_goal(state: datatypes.SimulatorState, goal_threshold: float = GOAL_THRESHOLD_M) -> jax.Array:
    """Return whether the SDC's *current* simulated position is within its goal radius.

    Args:
        state: Current simulator state.
        goal_threshold: Distance (in meters) below which the goal is reached.

    Returns:
        Boolean scalar: True if the SDC is currently within ``goal_threshold`` of its goal.
    """
    return get_distance_to_goal(state) < goal_threshold
