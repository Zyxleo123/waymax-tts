# Copyright 2025 Valeo.

"""Tests for the goal-reaching rewards added to the V-Max linear reward system.

Run (needs the ``waymax`` conda env)::

    pytest V-Max/tests/test_reward_goal.py -q

Covers both goal-reaching variants:

* ``reached_goal``       - per-step (dense) bonus while within the goal radius.
* ``reached_goal_once``  - one-time (sparse) bonus on the first step the goal is
  reached; implemented statelessly from the simulated trajectory history.

The goal is the SDC's destination, defined as the SDC's last valid logged
position.
"""

from __future__ import annotations

import os
import types

# These are lightweight logic tests; run on CPU to avoid GPU contention flakiness.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from waymax import datatypes

from vmax.simulator import operations
from vmax.simulator.wrappers import reward


GOAL_THRESHOLD_M = operations.GOAL_THRESHOLD_M  # shared default used by the reward functions


def _make_trajectory(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> datatypes.Trajectory:
    """Build a minimal waymax Trajectory of shape [num_objects, num_timesteps]."""
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    zeros = jnp.zeros_like(x)
    ones = jnp.ones_like(x)
    return datatypes.Trajectory(
        x=x,
        y=y,
        z=zeros,
        vel_x=zeros,
        vel_y=zeros,
        yaw=zeros,
        valid=valid,
        timestamp_micros=jnp.zeros_like(x, dtype=jnp.int32),
        length=ones,
        width=ones,
        height=ones,
    )


def _make_state(
    *,
    is_sdc: np.ndarray,
    log_x: np.ndarray,
    log_y: np.ndarray,
    log_valid: np.ndarray,
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    sim_valid: np.ndarray,
    timestep: int,
):
    """Build a duck-typed simulator state exposing only what the rewards read.

    Shapes: all *_x/*_y/*_valid are [num_objects, num_timesteps]. ``current_sim_trajectory``
    is derived as the slice of the sim trajectory at ``timestep`` (matching Waymax).
    """
    log_trajectory = _make_trajectory(log_x, log_y, log_valid)
    sim_trajectory = _make_trajectory(sim_x, sim_y, sim_valid)
    current_sim_trajectory = _make_trajectory(
        np.asarray(sim_x)[:, timestep : timestep + 1],
        np.asarray(sim_y)[:, timestep : timestep + 1],
        np.asarray(sim_valid)[:, timestep : timestep + 1],
    )
    object_metadata = types.SimpleNamespace(is_sdc=jnp.asarray(is_sdc, dtype=jnp.bool_))
    return types.SimpleNamespace(
        object_metadata=object_metadata,
        log_trajectory=log_trajectory,
        sim_trajectory=sim_trajectory,
        current_sim_trajectory=current_sim_trajectory,
        timestep=jnp.int32(timestep),
    )


def _single_step_state(*, goal_x, goal_y, current_x, current_y, is_sdc=None):
    """Convenience for per-step tests: goal defined by the log, one sim step."""
    if is_sdc is None:
        is_sdc = np.array([True])
    n = np.asarray(is_sdc).shape[0]
    log_x = np.asarray(goal_x, dtype=np.float32).reshape(n, -1)
    log_y = np.asarray(goal_y, dtype=np.float32).reshape(n, -1)
    log_valid = np.ones_like(log_x, dtype=bool)
    sim_x = np.asarray(current_x, dtype=np.float32).reshape(n, 1)
    sim_y = np.asarray(current_y, dtype=np.float32).reshape(n, 1)
    sim_valid = np.ones_like(sim_x, dtype=bool)
    return _make_state(
        is_sdc=is_sdc,
        log_x=log_x,
        log_y=log_y,
        log_valid=log_valid,
        sim_x=sim_x,
        sim_y=sim_y,
        sim_valid=sim_valid,
        timestep=0,
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_rewards_are_registered():
    assert reward._get_reward_fn("reached_goal") is reward._compute_reached_goal_reward
    assert reward._get_reward_fn("reached_goal_once") is reward._compute_reached_goal_once_reward
    assert reward._get_reward_fn("goal_progress") is reward._compute_goal_progress_reward


# --------------------------------------------------------------------------- #
# Per-step reward: reached_goal
# --------------------------------------------------------------------------- #


def test_per_step_fires_when_at_goal():
    state = _single_step_state(goal_x=[[0.0, 5.0, 10.0]], goal_y=[[0.0, 0.0, 0.0]], current_x=[10.0], current_y=[0.0])
    assert bool(reward._compute_reached_goal_reward(state)) is True


def test_per_step_does_not_fire_when_far():
    state = _single_step_state(goal_x=[[0.0, 5.0, 10.0]], goal_y=[[0.0, 0.0, 0.0]], current_x=[0.0], current_y=[0.0])
    assert bool(reward._compute_reached_goal_reward(state)) is False


def test_per_step_threshold_boundary():
    inside = _single_step_state(
        goal_x=[[10.0]], goal_y=[[0.0]], current_x=[10.0 - (GOAL_THRESHOLD_M - 0.1)], current_y=[0.0]
    )
    outside = _single_step_state(
        goal_x=[[10.0]], goal_y=[[0.0]], current_x=[10.0 - (GOAL_THRESHOLD_M + 0.1)], current_y=[0.0]
    )
    assert bool(reward._compute_reached_goal_reward(inside)) is True
    assert bool(reward._compute_reached_goal_reward(outside)) is False


def test_per_step_fires_every_step_while_camping():
    # Dense variant should keep firing while the SDC stays within the radius.
    sim_x = np.array([[0.0, 5.0, 10.0, 10.0, 10.0]])
    sim_y = np.zeros((1, 5))
    sim_valid = np.ones((1, 5), dtype=bool)
    for t in (3, 4):
        state = _make_state(
            is_sdc=np.array([True]),
            log_x=np.array([[0.0, 10.0]]),
            log_y=np.array([[0.0, 0.0]]),
            log_valid=np.array([[True, True]]),
            sim_x=sim_x,
            sim_y=sim_y,
            sim_valid=sim_valid,
            timestep=t,
        )
        assert bool(reward._compute_reached_goal_reward(state)) is True


def test_goal_uses_last_valid_not_last_timestep():
    # Trailing invalid (padded) step must be ignored; goal is (5, 0).
    at_goal = _make_state(
        is_sdc=np.array([True]),
        log_x=np.array([[0.0, 5.0, 999.0]]),
        log_y=np.array([[0.0, 0.0, 999.0]]),
        log_valid=np.array([[True, True, False]]),
        sim_x=np.array([[5.0]]),
        sim_y=np.array([[0.0]]),
        sim_valid=np.array([[True]]),
        timestep=0,
    )
    at_pad = _make_state(
        is_sdc=np.array([True]),
        log_x=np.array([[0.0, 5.0, 999.0]]),
        log_y=np.array([[0.0, 0.0, 999.0]]),
        log_valid=np.array([[True, True, False]]),
        sim_x=np.array([[999.0]]),
        sim_y=np.array([[999.0]]),
        sim_valid=np.array([[True]]),
        timestep=0,
    )
    assert bool(reward._compute_reached_goal_reward(at_goal)) is True
    assert bool(reward._compute_reached_goal_reward(at_pad)) is False


def test_per_step_uses_correct_sdc_among_multiple_objects():
    is_sdc = np.array([False, True])
    log_x = np.array([[50.0, 50.0], [0.0, 20.0]])  # other agent; SDC goal = (20, 0)
    log_y = np.zeros((2, 2))
    log_valid = np.ones((2, 2), dtype=bool)

    reached = _single_step_state_multi(is_sdc, log_x, log_y, log_valid, current=np.array([[50.0], [20.0]]))
    not_reached = _single_step_state_multi(is_sdc, log_x, log_y, log_valid, current=np.array([[50.0], [0.0]]))
    assert bool(reward._compute_reached_goal_reward(reached)) is True
    assert bool(reward._compute_reached_goal_reward(not_reached)) is False


def _single_step_state_multi(is_sdc, log_x, log_y, log_valid, current):
    return _make_state(
        is_sdc=is_sdc,
        log_x=log_x,
        log_y=log_y,
        log_valid=log_valid,
        sim_x=current[:, :1],
        sim_y=np.zeros_like(current[:, :1]),
        sim_valid=np.ones_like(current[:, :1], dtype=bool),
        timestep=0,
    )


# --------------------------------------------------------------------------- #
# One-time reward: reached_goal_once
# --------------------------------------------------------------------------- #


def _approach_state(timestep, sim_x, *, goal=10.0):
    sim_x = np.asarray(sim_x, dtype=np.float32).reshape(1, -1)
    n_t = sim_x.shape[1]
    return _make_state(
        is_sdc=np.array([True]),
        log_x=np.array([[0.0, goal]]),
        log_y=np.array([[0.0, 0.0]]),
        log_valid=np.array([[True, True]]),
        sim_x=sim_x,
        sim_y=np.zeros((1, n_t)),
        sim_valid=np.ones((1, n_t), dtype=bool),
        timestep=timestep,
    )


def test_once_does_not_fire_before_reaching():
    # distances: [10,6,3] -> within(<3): [F,F,F]. Never reached yet.
    sim_x = [0.0, 4.0, 7.0]
    for t in (0, 1, 2):
        assert bool(reward._compute_reached_goal_once_reward(_approach_state(t, sim_x))) is False


def test_once_fires_on_first_entry():
    # distances: [10,6,3,1,0] -> within: [F,F,F,T,T]; first entry at step 3.
    sim_x = [0.0, 4.0, 7.0, 9.0, 10.0]
    assert bool(reward._compute_reached_goal_once_reward(_approach_state(3, sim_x))) is True


def test_once_does_not_fire_while_camping():
    # Same trajectory; at step 4 (already reached at step 3) it must NOT fire again.
    sim_x = [0.0, 4.0, 7.0, 9.0, 10.0]
    assert bool(reward._compute_reached_goal_once_reward(_approach_state(4, sim_x))) is False


def test_once_only_first_entry_when_leaving_and_reentering():
    # within: step2 in, step3 out, step5 in again. Only step2 pays out.
    # distances to goal=10: [10, 8, 1, 5, 8, 0.5]
    sim_x = [0.0, 2.0, 9.0, 5.0, 2.0, 9.5]
    assert bool(reward._compute_reached_goal_once_reward(_approach_state(2, sim_x))) is True  # first entry
    assert bool(reward._compute_reached_goal_once_reward(_approach_state(5, sim_x))) is False  # re-entry


def test_once_uses_correct_sdc_among_multiple_objects():
    is_sdc = np.array([False, True])
    # Other agent (row 0) permanently at the SDC's goal; SDC (row 1) approaches.
    goal = 20.0
    log_x = np.array([[goal, goal], [0.0, goal]])
    log_y = np.zeros((2, 2))
    log_valid = np.ones((2, 2), dtype=bool)
    # SDC sim x approaches goal, first within radius at step 2.
    sim_x = np.array([[goal, goal, goal], [0.0, 10.0, 20.0]])
    sim_y = np.zeros((2, 3))
    sim_valid = np.ones((2, 3), dtype=bool)

    state_first = _make_state(
        is_sdc=is_sdc, log_x=log_x, log_y=log_y, log_valid=log_valid,
        sim_x=sim_x, sim_y=sim_y, sim_valid=sim_valid, timestep=2,
    )
    state_before = _make_state(
        is_sdc=is_sdc, log_x=log_x, log_y=log_y, log_valid=log_valid,
        sim_x=sim_x, sim_y=sim_y, sim_valid=sim_valid, timestep=1,
    )
    assert bool(reward._compute_reached_goal_once_reward(state_first)) is True
    assert bool(reward._compute_reached_goal_once_reward(state_before)) is False


# --------------------------------------------------------------------------- #
# Dense shaping reward: goal_progress (driving *towards* the goal)
# --------------------------------------------------------------------------- #


def test_progress_positive_when_closing_distance():
    # SDC moves from x=0 to x=4, goal at x=10: distance 10 -> 6, progress = +4.
    state = _approach_state(1, sim_x=[0.0, 4.0], goal=10.0)
    assert float(reward._compute_goal_progress_reward(state)) == pytest.approx(4.0)


def test_progress_negative_when_moving_away():
    # SDC moves from x=5 to x=2, goal at x=10: distance 5 -> 8, progress = -3.
    state = _approach_state(1, sim_x=[5.0, 2.0], goal=10.0)
    assert float(reward._compute_goal_progress_reward(state)) == pytest.approx(-3.0)


def test_progress_zero_when_stationary_or_perpendicular():
    # No movement at all: distance unchanged, progress = 0.
    state = _approach_state(1, sim_x=[3.0, 3.0], goal=10.0)
    assert float(reward._compute_goal_progress_reward(state)) == pytest.approx(0.0)


def test_progress_zero_on_first_step():
    # No previous step to compare against yet.
    state = _approach_state(0, sim_x=[0.0, 4.0], goal=10.0)
    assert float(reward._compute_goal_progress_reward(state)) == pytest.approx(0.0)


def test_progress_uses_correct_sdc_among_multiple_objects():
    is_sdc = np.array([False, True])
    log_x = np.array([[50.0, 50.0], [0.0, 20.0]])  # other agent stationary; SDC goal = (20, 0)
    log_y = np.zeros((2, 2))
    log_valid = np.ones((2, 2), dtype=bool)
    sim_x = np.array([[50.0, 50.0], [0.0, 6.0]])  # SDC moves 0 -> 6, distance 20 -> 14
    sim_y = np.zeros((2, 2))
    sim_valid = np.ones((2, 2), dtype=bool)

    state = _make_state(
        is_sdc=is_sdc, log_x=log_x, log_y=log_y, log_valid=log_valid,
        sim_x=sim_x, sim_y=sim_y, sim_valid=sim_valid, timestep=1,
    )
    assert float(reward._compute_goal_progress_reward(state)) == pytest.approx(6.0)


def test_progress_is_jittable():
    log_x = jnp.array([[0.0, 10.0]])
    log_y = jnp.array([[0.0, 0.0]])
    log_valid = jnp.array([[True, True]])
    sim_x = jnp.array([[0.0, 4.0, 9.0]])
    sim_y = jnp.zeros((1, 3))
    sim_valid = jnp.array([[True, True, True]])

    def build(timestep):
        return types.SimpleNamespace(
            object_metadata=types.SimpleNamespace(is_sdc=jnp.array([True])),
            log_trajectory=_make_trajectory(log_x, log_y, log_valid),
            sim_trajectory=_make_trajectory(sim_x, sim_y, sim_valid),
            current_sim_trajectory=_make_trajectory(
                jax.lax.dynamic_slice_in_dim(sim_x, timestep, 1, axis=1),
                jax.lax.dynamic_slice_in_dim(sim_y, timestep, 1, axis=1),
                jax.lax.dynamic_slice_in_dim(sim_valid, timestep, 1, axis=1),
            ),
            timestep=timestep,
        )

    progress = jax.jit(lambda t: reward._compute_goal_progress_reward(build(t)))

    assert float(progress(jnp.int32(1))) == pytest.approx(4.0)  # 0->4, dist 10->6
    assert float(progress(jnp.int32(2))) == pytest.approx(5.0)  # 4->9, dist 6->1


# --------------------------------------------------------------------------- #
# jit-ability (both run inside the training jit/vmap)
# --------------------------------------------------------------------------- #


def test_both_rewards_are_jittable():
    log_x = jnp.array([[0.0, 10.0]])
    log_y = jnp.array([[0.0, 0.0]])
    log_valid = jnp.array([[True, True]])
    sim_x = jnp.array([[0.0, 7.0, 9.0, 10.0]])
    sim_y = jnp.zeros((1, 4))
    sim_valid = jnp.array([[True, True, True, True]])

    def build(timestep):
        return types.SimpleNamespace(
            object_metadata=types.SimpleNamespace(is_sdc=jnp.array([True])),
            log_trajectory=_make_trajectory(log_x, log_y, log_valid),
            sim_trajectory=_make_trajectory(sim_x, sim_y, sim_valid),
            current_sim_trajectory=_make_trajectory(
                jax.lax.dynamic_slice_in_dim(sim_x, timestep, 1, axis=1),
                jax.lax.dynamic_slice_in_dim(sim_y, timestep, 1, axis=1),
                jax.lax.dynamic_slice_in_dim(sim_valid, timestep, 1, axis=1),
            ),
            timestep=timestep,
        )

    per_step = jax.jit(lambda t: reward._compute_reached_goal_reward(build(t)))
    once = jax.jit(lambda t: reward._compute_reached_goal_once_reward(build(t)))

    assert bool(per_step(jnp.int32(3))) is True  # at goal
    assert bool(per_step(jnp.int32(0))) is False  # far
    assert bool(once(jnp.int32(2))) is True  # first entry (dist at step2 = 1 < threshold)
    assert bool(once(jnp.int32(3))) is False  # already reached


# --------------------------------------------------------------------------- #
# End-to-end through the real RewardLinearWrapper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reward_name,current_x,expected",
    [
        ("reached_goal", 10.0, 2.0),
        ("reached_goal", 0.0, 0.0),
        ("reached_goal_once", 10.0, 2.0),
        ("reached_goal_once", 0.0, 0.0),
    ],
)
def test_reward_linear_wrapper_end_to_end(reward_name, current_x, expected):
    wrapper = reward.RewardLinearWrapper.__new__(reward.RewardLinearWrapper)
    wrapper._reward_config = {reward_name: 2.0}
    state = _single_step_state(goal_x=[[0.0, 10.0]], goal_y=[[0.0, 0.0]], current_x=[current_x], current_y=[0.0])
    assert pytest.approx(float(wrapper.reward(state, None)), abs=1e-6) == expected
