# Copyright 2025 Valeo.

"""Reward functions for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from vmax.simulator import constants, metrics, operations
from vmax.simulator.wrappers.base import Wrapper

# Clearance (m) to the nearest road edge at which `offroad_margin` starts to
# bite. Scenarios harvested as offroad failures cross the edge by only ~0.1-0.2 m
# while the human log keeps as little as ~0.3 m clearance, so the ramp has to be
# short enough that expert-like driving still scores ~0.
OFFROAD_MARGIN_M = 0.3


class RewardLinearWrapper(Wrapper):
    """Wraps the environment to compute rewards using a linear combination of functions."""

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        """Initialize the reward linear wrapper.

        Args:
            env: The environment to wrap.
            reward_config: Configuration for reward computation.

        """
        super().__init__(env)
        self._reward_config = reward_config

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Combine rewards provided from the reward functions.
        The reward is computed as a linear combination of the individual rewards,
        weighted by their respective coefficients.

        Args:
            state: Current simulator state.

        Returns:
            Combined reward value.
        """
        reward = 0.0

        for reward_name, reward_weigth in self._reward_config.items():
            reward_fn = _get_reward_fn(reward_name)
            reward += reward_fn(state) * reward_weigth

        return jnp.array(reward, dtype=jnp.float32)


class RewardCustomWrapper(Wrapper):
    """Wraps the environment to compute rewards using a custom function."""

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Compute a custom reward.

        This is a placeholder function, use it as you wish o7

        Args:
            state: Current simulator state.
            action: Action taken.

        Returns:
            Reward value.

        """
        # Implement your custom reward logic here
        return jnp.array(0.0)


def _get_reward_fn(reward_name: str) -> callable:
    """Retrieve a reward function by its name.

    Args:
        reward_name: Name identifier of the reward function.

    Returns:
        Callable reward function.

    """
    reward_dict = {
        "log_div_clip": _compute_log_divergence_clip_reward,
        "log_div": _compute_log_divergence_reward,
        "overlap": _compute_overlap_reward,
        "offroad": _compute_offroad_reward,
        "offroad_margin": _compute_offroad_margin_reward,
        "off_route": _compute_off_route_reward,
        "below_ttc": _compute_below_ttc_reward,
        "red_light": _compute_red_light_reward,
        "comfort": _compute_comfort_reward,
        "yaw_rate_penalty": _compute_yaw_rate_penalty_reward,
        "overspeed": _compute_overspeed_limit_reward,
        "driving_direction": _compute_driving_direction_reward,
        "lane_deviation": _compute_deviate_lane_reward,
        "progression": _compute_making_progress_reward,
        "reached_goal": _compute_reached_goal_reward,
        "reached_goal_once": _compute_reached_goal_once_reward,
        "goal_progress": _compute_goal_progress_reward,
    }

    if reward_name not in reward_dict:
        raise ValueError(f"Reward function {reward_name} not implemented.")

    return reward_dict[reward_name]


# Penalty based rewards


def _compute_overlap_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing overlaps between the SDC and other objects.

    Args:
        state: Current simulator state.

    Returns:
        True if overlap detected, False otherwise.

    """
    overlap = waymax_metrics.OverlapMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree_util.tree_map(lambda x: x[sdc_idx], overlap)

    return sdc_overlap == 1.0


def _compute_offroad_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing when the SDC drives off the road.

    Args:
        state: Current simulator state.

    Returns:
        True if off-road detected, False otherwise.

    """
    offroad = waymax_metrics.OffroadMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree_util.tree_map(lambda x: x[sdc_idx], offroad)

    return sdc_offroad == 1.0


def _compute_offroad_margin_reward(state: datatypes.SimulatorState) -> jax.Array:
    """Continuous version of :func:`_compute_offroad_reward`.

    ``offroad`` is a step function: it pays nothing to keep 0.3 m of clearance
    rather than 0.01 m, so the policy gets no gradient until it has already
    crossed the edge — too late to steer away. This returns the *penetration*
    of the SDC bounding box toward the nearest road edge, ramped over
    ``OFFROAD_MARGIN_M``:

        0.0  clearance >= OFFROAD_MARGIN_M  (comfortably on-road)
        1.0  bbox exactly on the edge
        2.0  bbox >= OFFROAD_MARGIN_M past the edge (clipped)

    Sign convention matches the other penalty rewards: positive magnitude, to be
    applied with a negative weight in ``reward_config``.
    """
    from waymax.metrics.roadgraph import compute_signed_distance_to_nearest_road_edge_point

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    current = datatypes.dynamic_slice(state.sim_trajectory, state.timestep, 1, -1)

    # (num_objects, num_corners=4, 2) -> SDC only, plus z to disambiguate
    # overpasses (same trick as waymax's is_offroad).
    corners = jnp.squeeze(current.bbox_corners, axis=-3)[sdc_idx]
    z = jnp.ones_like(corners[..., 0:1]) * current.z[sdc_idx, :][jnp.newaxis, :]
    corners = jnp.concatenate((corners, z), axis=-1)

    # >0 means that corner is past the road edge; take the worst corner.
    distance = jnp.max(compute_signed_distance_to_nearest_road_edge_point(corners, state.roadgraph_points))

    return jnp.clip((distance + OFFROAD_MARGIN_M) / OFFROAD_MARGIN_M, 0.0, 2.0)


def _compute_red_light_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing red light violations.

    Args:
        state: Current simulator state.

    Returns:
        True if red light violation detected, False otherwise.

    """
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value

    return has_runned_red_light


def _compute_overspeed_limit_reward(state: datatypes.SimulatorState, threshold: float = 2.23) -> bool:
    """Compute a reward based on speed-limit adherence.

    Returns True if the SDC's speed exceeds the road's speed limit by more than
    2.23 m/s (approximately 5 mph).

    Args:
        state: Current simulator state.

    Returns:
        True if speed limit is exceeded by more than 2.23 m/s, False otherwise.

    """
    speed_limit = metrics.infer_speed_limit_from_simulator_state(state)
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()

    return ego_speed > speed_limit + threshold


def _compute_below_ttc_reward(state: datatypes.SimulatorState, threshold: float = 1.5) -> bool:
    """Compute a reward based on time-to-collision with other objects.

    The time-to-collision (TTC) is a measure of how long it would take for the SDC to
    collide with another object if both continue on their current trajectories.

    Args:
        state: Current simulator state.
        threshold: Minimum safe time-to-collision in seconds. Default is 1.5s.

    Returns:
        True if TTC is below threshold (unsafe), False otherwise (safe).

    """
    ttc = metrics.TimeToCollisionMetric().compute(state).value

    return ttc < threshold


def _compute_off_route_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing deviation from the planned route.

    Returns True if the SDC has deviated from its planned route. A deviation occurs
    when the SDC's position has a non-zero distance to the nearest point on the route.

    Args:
        state: Current simulator state.

    Returns:
        True if off-route detected, False otherwise.
    """
    # Waymax offroute's score: 0 if you're on-route, distance to route if you're offroute
    off_route_score = metrics.OffRouteMetric().compute(state).value

    return off_route_score > 0


def _compute_driving_direction_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward for driving direction compliance.

    This reward is based on the driving direction compliance metric, which evaluates
    whether the SDC is following the intended driving direction.

    Args:
        state: Current simulator state.

    Returns:
        True if driving direction is compliant, False otherwise.

    """
    driving_direction_metric = metrics.DrivingDirectionComplianceMetric().compute(state).value

    return driving_direction_metric > 0


def _compute_deviate_lane_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing lane deviations.

    This reward is based on the lane deviation metric, which evaluates how much the SDC
    deviates from its intended lane.

    Args:
        state: Current simulator state.

    Returns:
        True if lane deviation is detected, False otherwise.

    """
    on_multiple_lanes_metric = metrics.OnMultipleLanesMetric().compute(state).value

    return on_multiple_lanes_metric > 0


# Reward based rewards


def _compute_log_divergence_clip_reward(state: datatypes.SimulatorState, threshold: float = 0.3) -> bool:
    """Compute reward based on whether log divergence exceeds a threshold.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.
        threshold: Maximum acceptable divergence (in log space) before considering it a deviation.
                  Default is 0.3, higher values are more permissive.

    Returns:
        True if divergence exceeds threshold, False otherwise.
    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence < threshold


def _compute_making_progress_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward promoting forward progress along the route.

    Compares the current progression metric with the previous timestep to determine
    if the SDC is making forward progress along its intended route.

    Args:
        state: Current simulator state.

    Returns:
        True if the SDC made forward progress, False otherwise.
    """
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value

    return current_progression > previous_progression


def _get_sdc_goal_xy(state: datatypes.SimulatorState, sdc_idx: jax.Array) -> jax.Array:
    """Return the SDC's goal position: its last valid logged (x, y) position.

    Thin wrapper around ``operations.get_sdc_goal_xy`` (shared with the goal-reaching
    metric) so the reward and the logged metric always use the same goal definition.
    """
    return operations.get_sdc_goal_xy(state, sdc_idx)


def _compute_reached_goal_reward(
    state: datatypes.SimulatorState, goal_threshold: float = operations.GOAL_THRESHOLD_M
) -> bool:
    """Compute a *per-step* (dense) goal-reaching reward.

    The goal is the SDC's destination in the scenario (its last valid logged
    position). The reward fires (returns True) on **every** step for which the
    SDC's current simulated position is within ``goal_threshold`` meters of that
    goal. It therefore rewards both reaching and *staying at* the destination,
    encouraging the policy to actually reach the goal rather than only making
    incremental progress along the route.

    Args:
        state: Current simulator state.
        goal_threshold: Distance (in meters) below which the goal is considered reached.
            Defaults to the shared ``operations.GOAL_THRESHOLD_M`` (matches evaluation).

    Returns:
        True if the SDC is currently within ``goal_threshold`` of its goal, False otherwise.
    """
    return operations.is_within_goal(state, goal_threshold)


def _compute_reached_goal_once_reward(
    state: datatypes.SimulatorState, goal_threshold: float = operations.GOAL_THRESHOLD_M
) -> bool:
    """Compute a *one-time* (sparse) goal-reaching reward.

    Identical goal definition to ``_compute_reached_goal_reward`` but the reward
    fires only on the **first** step at which the SDC enters the goal radius, and
    never again for the rest of the episode. This mimics keeping a ``goal_reached``
    flag, but is implemented statelessly (the reward function only receives the
    ``SimulatorState``): the state already carries the full simulated trajectory
    history, so "first time" is detected as "in the goal radius now, but not at any
    earlier simulated step". This is robust to the SDC leaving and re-entering the
    radius (it still only pays out once).

    Args:
        state: Current simulator state.
        goal_threshold: Distance (in meters) below which the goal is considered reached.
            Defaults to the shared ``operations.GOAL_THRESHOLD_M`` (matches evaluation).

    Returns:
        True only on the first step the SDC reaches its goal, False otherwise.
    """
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    goal_xy = _get_sdc_goal_xy(state, sdc_idx)

    # Distance to goal at every simulated step of the SDC's trajectory.
    sim_xy = state.sim_trajectory.xy[sdc_idx]  # [num_timesteps, 2]
    sim_valid = state.sim_trajectory.valid[sdc_idx]  # [num_timesteps]
    distances = jnp.linalg.norm(sim_xy - goal_xy, axis=-1)  # [num_timesteps]

    # Only steps that have actually been simulated (valid) count. Future steps are
    # marked invalid by Waymax, so they never contribute.
    within_goal = (distances < goal_threshold) & sim_valid  # [num_timesteps]

    step_indices = jnp.arange(within_goal.shape[0])
    timestep = state.timestep

    reached_now = within_goal[timestep]
    reached_before = jnp.any(within_goal & (step_indices < timestep))

    return reached_now & jnp.logical_not(reached_before)


# Linar reward functions


def _compute_log_divergence_reward(state: datatypes.SimulatorState) -> float:
    """Compute reward based on log divergence.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.

    Returns:
        Log divergence value.

    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence


def _compute_goal_progress_reward(state: datatypes.SimulatorState) -> float:
    """Compute a *dense* shaping reward for driving towards the SDC's goal.

    Unlike ``reached_goal``/``reached_goal_once`` which only look at whether the
    SDC is currently *within* the goal radius (pure proximity), this reward is
    based on the SDC's *motion relative to the goal*: it rewards closing the
    distance to the goal from one step to the next, and penalizes moving away
    from it, regardless of how far the SDC still is from its destination. This
    directly encourages driving towards the goal at every step, not just
    arriving at/staying near it.

    Implemented as the reduction in distance-to-goal between the previous and
    current simulated step: ``previous_distance_to_goal - current_distance_to_goal``.
    Positive values mean the SDC moved closer to the goal on this step, negative
    values mean it moved further away, and the magnitude scales with how much
    ground was gained/lost (roughly the SDC's speed component directed at the
    goal, multiplied by the simulation timestep).

    Args:
        state: Current simulator state.

    Returns:
        The signed change in distance to the goal (meters); positive means the
        SDC made progress towards the goal on this step. Returns 0.0 on the
        first simulated step, since there is no previous position to compare.

    """
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    goal_xy = _get_sdc_goal_xy(state, sdc_idx)

    current_x = state.current_sim_trajectory.x[sdc_idx].squeeze()
    current_y = state.current_sim_trajectory.y[sdc_idx].squeeze()
    current_xy = jnp.stack([current_x, current_y])
    current_distance = jnp.linalg.norm(current_xy - goal_xy)

    sim_xy = state.sim_trajectory.xy[sdc_idx]  # [num_timesteps, 2]
    previous_timestep = jnp.maximum(state.timestep - 1, 0)
    previous_distance = jnp.linalg.norm(sim_xy[previous_timestep] - goal_xy)

    progress = previous_distance - current_distance

    return jnp.where(state.timestep <= 0, 0.0, progress)


def _compute_comfort_reward(state: datatypes.SimulatorState) -> float:
    """Compute a comfort-related reward.

    This reward is based on the comfort metric, which evaluates the smoothness of the SDC's
    trajectory. A higher comfort metric indicates a smoother and more comfortable ride.

    NOTE: this is a *bonus* in (0, 1], not a penalty -- a parked car scores ~1.0, so a large
    weight pays the policy to idle. It is also blind to single-frame yaw chatter (see
    `_compute_yaw_rate_penalty_reward`). Prefer `yaw_rate_penalty` for that failure mode.

    Args:
        state: Current simulator state.

    Returns:
        Comfort metric value.

    """
    comfort_metric_reward_2 = metrics.ComfortMetric().compute_reward(state).value

    return comfort_metric_reward_2


# The comfort limit ComfortMetric uses for yaw rate, reused here so the penalty and the
# metric agree on what "too fast" means. rl/bank.py gates banked trajectories on the same
# number, measured with the same raw finite difference.
_YAW_RATE_LIMIT_RAD_S = 0.95
_COMFORT_WINDOW_STEPS = 10


def _compute_yaw_rate_penalty_reward(state: datatypes.SimulatorState) -> float:
    """Penalty in (-1, 0] for *raw* yaw rate above the comfort limit.

    Why this exists rather than `comfort` (all measured 2026-07-16 on banked
    scenario_00197, a real chattering rollout whose yaw flips +-25 deg every frame):

    * `comfort`'s own yaw-rate term is **blind** to it. `ComfortMetric` savgol-filters yaw
      (window 5, polyorder 2) before differentiating, and a period-2 oscillation sits at
      the Nyquist frequency of the 10 Hz sim, so the filter erases it: raw peak
      **4.49 rad/s** reads as **0.06 rad/s**, scoring `value_yaw_rate` = **1.000**.
    * The *aggregate* comfort score still catches it (**0.357**) via its lateral- and
      yaw-acceleration terms -- so `comfort` is not useless here, just indirect.
    * But `comfort` is a **bonus in (0, 1], not a penalty**: banked scenario_00200, whose
      SDC is parked for the whole scene, scores a perfect **1.000**. At weight w over an
      80-step episode that is +80w for doing nothing, against `reached_goal_once`'s
      one-off 5.0.

    This term is instead 0 while compliant and saturates toward -1 as the excess grows, so
    a parked car earns nothing and a chatterer pays ~-0.97/step. Bounded on purpose: an
    unbounded `-(peak - limit)` would be worth -3.5 per step on a chatterer and swamp every
    other term.

    Calibrated against the human log on the 218-scene failure set: the expert never exceeds
    1.09 rad/s under this statistic (p50 0.35, mean per-step penalty 0.0000 -- 1 scene in
    218 is taxed at all), while the policy's median peak is 3.07 rad/s. The limit is well
    clear of anything a human does and only bites the pathology.

    `rl/bank.py` gates banked trajectories on the same raw statistic and the same limit.

    Args:
        state: Current simulator state.

    Returns:
        Penalty value in (-1, 0].

    """
    past_traj = datatypes.dynamic_slice(
        state.sim_trajectory,
        state.timestep - (_COMFORT_WINDOW_STEPS - 1),
        _COMFORT_WINDOW_STEPS,
        -1,
    )
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    yaw = past_traj.yaw[sdc_idx]  # [_COMFORT_WINDOW_STEPS]
    valid = past_traj.valid[sdc_idx]

    d_yaw = jnp.diff(yaw)
    d_yaw = jnp.arctan2(jnp.sin(d_yaw), jnp.cos(d_yaw))  # wrap to (-pi, pi]
    # A difference is only meaningful when both of its endpoints are real.
    pair_valid = valid[:-1] & valid[1:]
    yaw_rate = jnp.abs(d_yaw) / constants.TIME_DELTA
    peak = jnp.max(jnp.where(pair_valid, yaw_rate, 0.0))

    excess = jnp.maximum(0.0, peak - _YAW_RATE_LIMIT_RAD_S)
    return -(1.0 - jnp.exp(-excess))
