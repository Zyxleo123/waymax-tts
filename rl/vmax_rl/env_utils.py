"""Helpers to tweak the V-Max environment after construction.

V-Max's ``make_env_for_{training,evaluation}`` build a Waymax
``PlanningAgentEnvironment`` that controls only the SDC and **log-replays** every
other agent. That is fine for closed-loop expert replay (BC), but **policy**
rollouts deviate from the log: frozen log-replayed agents then produce timing
ghost collisions that corrupt the SAC reward (see ``rl/CONTEXT.md``).

Default training/eval now uses the **policy-friendly sim** preset:

* :func:`attach_idm_sim_agents` — reactive IDM non-ego agents (yield to the SDC).
* :func:`apply_policy_friendly_reward_config` — don't terminate on overlap/offroad;
  downweight those penalties so progression can dominate.

Opt back into legacy log-replay + harsh penalties with ``--log-replay-agents``.
"""

from __future__ import annotations

from rl.vmax_rl import compat  # noqa: F401

# Preset for learnable policy rollouts (SAC / PPO on failure cases).
POLICY_FRIENDLY_TERMINATION_KEYS = ["run_red_light"]
POLICY_FRIENDLY_REWARD_WEIGHTS = {"overlap": -0.25, "offroad": -0.25}


def apply_policy_friendly_reward_config(config: dict) -> None:
    """Patch ``termination_keys`` + collision/offroad weights for policy RL."""
    config["termination_keys"] = list(POLICY_FRIENDLY_TERMINATION_KEYS)
    reward_cfg = config["reward_config"]
    for key, weight in POLICY_FRIENDLY_REWARD_WEIGHTS.items():
        reward_cfg[key] = weight


def _unwrap_to_base(env):
    """Walk the V-Max wrapper chain down to the base PlanningAgentEnvironment."""
    base = env
    # Wrappers expose the wrapped env as ``.env``; the base env does not.
    while hasattr(base, "env"):
        base = base.env
    return base


def attach_idm_sim_agents(env, *, desired_vel: float = 30.0):
    """Make all non-SDC objects reactive (IDM) instead of log-replayed.

    Mutates the underlying ``PlanningAgentEnvironment`` in place (sets its
    ``_sim_agent_actors`` / ``_sim_agent_params``) and returns ``env`` for
    chaining. Must be called before the env is used in ``reset``/``step`` (so the
    per-episode sim-agent actor states get initialised on reset).

    Args:
        env: A V-Max env from ``make_env_for_{training,evaluation}``.
        desired_vel: IDM free-road desired speed (m/s).
    """
    from waymax.agents import IDMRoutePolicy

    idm = IDMRoutePolicy(
        is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
        desired_vel=desired_vel,
    )
    base = _unwrap_to_base(env)
    base._sim_agent_actors = (idm,)
    base._sim_agent_params = (None,)
    return env
