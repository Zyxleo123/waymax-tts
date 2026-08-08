"""A gymnasium.Env that wraps the Waymax PlanningAgentEnvironment for PPO.

The ego (SDC) is controlled with one of two action spaces, selected via
``action_space_type``:

* ``"bicycle"`` (default): an :class:`InvertibleBicycleModel`
  (acceleration, steering) action in ``[-1, 1]``. Motion is kinematically
  feasible by construction.
* ``"delta"``: a :class:`DeltaLocal` (next-position) action ``(dx, dy, dyaw)``
  expressed in the ego frame. The policy still emits values in ``[-1, 1]``;
  they are scaled to physical limits (``delta_max_dx``/``dy``/``dyaw``) before
  being applied. This mirrors the position-based control used by the diffusion
  planner pipeline, at the cost of the bicycle model's feasibility guarantee.

Observations are a compact, fixed-size, SDC-centric vector. The reward is dense
goal-progress plus a success bonus minus collision/offroad penalties and a small
action-magnitude penalty.

Waymax runs on the GPU via JAX; the SB3 policy network is tiny and runs on CPU,
so there is no GPU contention. All per-step heavy lifting (step / metrics /
observation) is JIT-compiled once and reused across scenarios (shapes are fixed
by ``max_num_objects`` and the WOMD horizon).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from rl import obs_layout
from rl.obs_layout import DEFAULT_OBS_BLOCKS, obs_layout_size
from rl.scenario_source import ScenarioSource

# The observation's block structure (entity counts, history depth, feature dims,
# trailing valid bit) lives in ``rl.obs_layout``, aligned to V-Max's
# ``repro_sac_v2`` observation_config, and is shared with the torch encoders in
# ``rl.encoders``. See that module for the reference config.
_OBS_CLIP = 10.0

# Default physical limits for the "delta" (next-position) action space, applied
# per simulator step. The policy outputs values in [-1, 1] that are scaled to
# these ranges before being passed to the DeltaLocal dynamics model.
_DELTA_MAX_DX = 6.0     # meters (forward/back in ego frame)
_DELTA_MAX_DY = 6.0     # meters (left/right in ego frame)
_DELTA_MAX_DYAW = float(np.pi)  # radians

_VALID_ACTION_SPACES = ("bicycle", "delta")

# Preset for learnable policy rollouts (SAC / PPO on failure cases).
# Mirrors ``rl/vmax_rl/env_utils.POLICY_FRIENDLY_*``.
POLICY_FRIENDLY_COLLISION = -0.25
POLICY_FRIENDLY_OFFROAD = -0.25


def attach_idm_sim_agents(env: waymax_env.PlanningAgentEnvironment, *, desired_vel: float = 30.0):
    """Make all non-SDC objects reactive (IDM) instead of log-replayed.

    Must be called before the first ``reset`` so per-episode sim-agent actor
    state is initialised correctly.
    """
    from waymax.agents import IDMRoutePolicy

    idm = IDMRoutePolicy(
        is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
        desired_vel=desired_vel,
    )
    env._sim_agent_actors = (idm,)
    env._sim_agent_params = (None,)
    return env


@dataclasses.dataclass
class RewardConfig:
    """Weights for the shaped reward."""

    progress: float = 1.0       # reward per meter of progress (straight-line or route)
    step_penalty: float = 0.0   # constant per-step penalty
    action_penalty: float = 0.01  # penalty on sum(action^2)
    # Collision / offroad are charged at most ONCE per episode (on the first
    # violating step), not on every violating step. Integrating them made
    # standing still the safe play: a policy that never moves scores ~0, while
    # any attempt to drive risked -5/step for as long as the episode ran.
    collision: float = -10.0    # one-time penalty; ends the episode by default
    offroad: float = -5.0       # one-time penalty
    goal_bonus: float = 10.0    # one-time bonus; ends the episode
    goal_threshold_m: float = 3.0
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = False
    # Route-following reward. When True, "progress" is measured as advancement
    # along the expert log path (which is on-road and collision-free), and the
    # policy is penalized for deviating laterally from it. This discourages the
    # straight-line "beeline to the goal" behavior that cuts corners off-road and
    # drives through other agents. When False, progress is straight-line distance
    # reduction to the final goal point (original behavior).
    route_reward: bool = False
    lateral_penalty: float = 0.5  # reward per meter of lateral deviation from route
    # V-Max scores route adherence with a *bounded indicator* (`off_route: -0.2`
    # charged once per step past a threshold), not an unbounded per-metre cost.
    # The per-metre form lets the penalty dominate the whole return when the
    # policy is far from the route -- which is exactly what happened on the first
    # SB3 run (return was ~= the integrated lateral penalty). Set
    # ``off_route_threshold_m`` to switch to the V-Max form.
    off_route_threshold_m: float | None = None
    off_route_penalty: float = -0.2
    # V-Max's `progression` is likewise a *bounded* indicator: +w on any step
    # where route arclength increased, whether by 0.1 m or 1 m. The per-metre
    # form here scales with speed, so at 1-2 m/step it pays 5-10x V-Max's rate
    # and pays it for driving fast rather than for making progress. Set True for
    # V-Max parity.
    progression_indicator: bool = False


def observation_dim() -> int:
    return obs_layout_size(DEFAULT_OBS_BLOCKS)


def _wrap_to_pi(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _top_k_padded(score: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """``jax.lax.top_k`` that tolerates fewer candidates than ``k``.

    ``lax.top_k`` requires ``k <= score.shape[-1]``, but a scene can legitimately
    hold fewer entities than an observation block has slots (a scene with 3
    traffic lights, or ``max_num_objects`` below the agents block size). Pad the
    score with a ``-inf`` sentinel so the block always yields ``k`` rows; the
    padded rows score below the ``-1e8`` selection cutoff, so callers mark them
    invalid and zero them out. Returned indices are clamped into range so the
    subsequent gathers stay valid.
    """
    n = score.shape[-1]
    if n >= k:
        return jax.lax.top_k(score, k)
    padded = jnp.concatenate([score, jnp.full((k - n,), -jnp.inf, score.dtype)])
    values, indices = jax.lax.top_k(padded, k)
    return values, jnp.minimum(indices, n - 1)


class WaymaxGymEnv(gym.Env):
    """Single-agent Waymax planning environment as a gymnasium.Env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_source: ScenarioSource,
        *,
        reward_config: RewardConfig | None = None,
        max_episode_steps: int = 80,
        sequential: bool = False,
        seed: int | None = None,
        action_space_type: str = "bicycle",
        delta_max_dx: float = _DELTA_MAX_DX,
        delta_max_dy: float = _DELTA_MAX_DY,
        delta_max_dyaw: float = _DELTA_MAX_DYAW,
        reactive_agents: bool = False,
        idm_desired_vel: float = 30.0,
    ):
        super().__init__()
        if action_space_type not in _VALID_ACTION_SPACES:
            raise ValueError(
                f"action_space_type must be one of {_VALID_ACTION_SPACES}, "
                f"got {action_space_type!r}."
            )
        self._source = scenario_source
        self._reward = reward_config or RewardConfig()
        self._max_episode_steps_cap = int(max_episode_steps)
        self._sequential = bool(sequential)
        self._seq_cursor = 0
        self._np_rng = np.random.default_rng(seed)

        # Match the env to the actual object count of the loaded scenarios.
        max_num_objects = int(self._source.num_objects)

        # Select the dynamics model / action space. The policy always emits a
        # vector in [-1, 1]; for "delta" we scale it to physical (dx, dy, dyaw).
        self._action_space_type = action_space_type
        if action_space_type == "bicycle":
            self._dynamics = waymax_dynamics.InvertibleBicycleModel(normalize_actions=True)
            self._action_dim = 2
            self._action_scale: np.ndarray | None = None
        else:  # "delta"
            self._dynamics = waymax_dynamics.DeltaLocal(
                max_dx=float(delta_max_dx),
                max_dy=float(delta_max_dy),
                max_dyaw=float(delta_max_dyaw),
            )
            self._action_dim = 3
            self._action_scale = np.array(
                [delta_max_dx, delta_max_dy, delta_max_dyaw], dtype=np.float32
            )
        env_cfg = waymax_config.EnvironmentConfig(
            max_num_objects=max_num_objects,
            controlled_object=waymax_config.ObjectType.SDC,
            compute_reward=False,  # reward is computed in this wrapper instead
            metrics=waymax_config.MetricsConfig(metrics_to_run=("overlap", "offroad")),
        )
        self._env = waymax_env.PlanningAgentEnvironment(
            dynamics_model=self._dynamics,
            config=env_cfg,
        )
        if reactive_agents:
            attach_idm_sim_agents(self._env, desired_vel=idm_desired_vel)

        # JIT the hot paths once; shapes are constant across scenarios.
        env_obj = self._env

        def _metric_values(state):
            m = env_obj.metrics(state)
            return m["overlap"].value, m["offroad"].value

        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)
        self._jit_metrics = jax.jit(_metric_values)
        self._jit_obs = jax.jit(_compute_observation)

        obs_dim = observation_dim()
        self.observation_space = spaces.Box(
            low=-_OBS_CLIP, high=_OBS_CLIP, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32
        )

        # Per-episode state.
        self._state: Any = None
        self._goal_xy: jax.Array | None = None
        self._prev_dist: float = 0.0
        self._route_xy: np.ndarray | None = None
        self._route_s: np.ndarray | None = None
        self._prev_s: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = self._max_episode_steps_cap
        # Whether the one-time collision / offroad penalties have been paid this
        # episode. info["collision"] / info["offroad"] stay per-step (the
        # callbacks OR them over the episode themselves).
        self._charged_collision: bool = False
        self._charged_offroad: bool = False

    # ------------------------------------------------------------------ #
    def _next_scenario_index(self) -> int:
        if self._sequential:
            idx = self._seq_cursor % len(self._source)
            self._seq_cursor += 1
            return idx
        return self._source.sample_index(self._np_rng)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        scen_np = self._source.get(self._next_scenario_index())
        spec_dict = (
            dataclasses.asdict(self._source.spec(self._seq_cursor - 1))
            if self._sequential
            else {}
        )
        return self._reset_from_scenario(scen_np, spec_dict)

    def _reset_from_scenario(self, scen_np: Any, spec_dict: dict) -> tuple[np.ndarray, dict]:
        """Shared reset body: takes an already-fetched host scenario and drives the
        simulator, goal, route and episode bookkeeping to a fresh episode start.

        Factored out so subclasses that fetch scenarios differently -- by index
        (this class), by sampling a mix (``MixedWaymaxGymEnv`` in bc_core.py), or
        by pulling the next one off an infinite stream (``StreamingWaymaxGymEnv``
        below) -- don't each reimplement this bookkeeping.
        """
        scen = jax.tree_util.tree_map(jnp.asarray, scen_np)
        self._state = self._jit_reset(scen)
        self._goal_xy = jnp.asarray(_compute_goal_xy(scen_np), dtype=jnp.float32)

        # Expert log path (on-road, collision-free) used for the route reward.
        self._route_xy, self._route_s = _compute_route(scen_np)

        obs, goal_dist, ego_xy = self._jit_obs(self._state, self._goal_xy)
        self._prev_dist = float(goal_dist)
        self._prev_s, init_lateral = _project_to_route(
            np.asarray(ego_xy), self._route_xy, self._route_s
        )
        self._step_count = 0
        self._charged_collision = False
        self._charged_offroad = False

        num_timesteps = int(np.asarray(scen_np.log_trajectory.x).shape[-1])
        init_steps = self._env.config.init_steps
        self._max_steps = min(self._max_episode_steps_cap, num_timesteps - init_steps)

        info = {
            "goal_dist": self._prev_dist,
            "lateral_deviation_m": float(init_lateral),
            "scenario": spec_dict,
        }
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self._action_dim)
        # The bicycle model normalizes [-1, 1] internally; DeltaLocal expects
        # physical units, so scale the normalized action to its limits here.
        if self._action_scale is not None:
            data = np.clip(action, -1.0, 1.0) * self._action_scale
        else:
            data = action
        wx_action = datatypes.Action(
            data=jnp.asarray(data, dtype=jnp.float32),
            valid=jnp.ones((1,), dtype=jnp.bool_),
        )
        self._state = self._jit_step(self._state, wx_action)

        obs, goal_dist, ego_xy = self._jit_obs(self._state, self._goal_xy)
        overlap, offroad = self._jit_metrics(self._state)

        goal_dist = float(goal_dist)
        collision = float(overlap) > 0.5
        off = float(offroad) > 0.5
        reached = goal_dist <= self._reward.goal_threshold_m

        rc = self._reward
        s, lateral = _project_to_route(
            np.asarray(ego_xy), self._route_xy, self._route_s, prev_s=self._prev_s
        )
        if rc.route_reward:
            if rc.progression_indicator:
                # V-Max's `progression`: a bounded +w whenever route progression
                # increased at all, regardless of by how much.
                reward = rc.progress * float(s > self._prev_s)
            else:
                reward = rc.progress * (s - self._prev_s)
            if rc.off_route_threshold_m is None:
                reward -= rc.lateral_penalty * lateral
            else:
                # V-Max form: a bounded per-step indicator, so the route term can
                # never swamp progression the way the per-metre cost does.
                reward += rc.off_route_penalty * float(lateral > rc.off_route_threshold_m)
            self._prev_s = s
        else:
            reward = rc.progress * (self._prev_dist - goal_dist)
        reward -= rc.step_penalty
        reward -= rc.action_penalty * float(np.sum(action ** 2))

        terminated = False
        if collision:
            if not self._charged_collision:
                reward += rc.collision
                self._charged_collision = True
            terminated = terminated or rc.terminate_on_collision
        if off:
            if not self._charged_offroad:
                reward += rc.offroad
                self._charged_offroad = True
            terminated = terminated or rc.terminate_on_offroad
        if reached:
            reward += rc.goal_bonus
            terminated = True

        self._prev_dist = goal_dist
        self._step_count += 1
        truncated = (self._step_count >= self._max_steps) and not terminated

        info = {
            "goal_dist": goal_dist,
            "collision": collision,
            "offroad": off,
            "reached": reached,
            "lateral_deviation_m": float(lateral),
            "is_success": bool(reached),
        }
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Trajectory export (for distilling RL rollouts into the diffusion model)
    # ------------------------------------------------------------------ #
    @property
    def sim_state(self) -> Any:
        """The current (post-step) Waymax ``SimulatorState`` for this episode."""
        return self._state

    def simulated_log_state(self) -> Any:
        """Returns the episode's ``SimulatorState`` with the ego's *logged*
        trajectory overwritten by the *simulated* (policy-controlled) trajectory
        for the steps that were actually rolled out.

        The result is a drop-in target for ``data.preprocess`` /
        ``train.train_diffusion``: the diffusion model reads its ego trajectory
        target from ``log_trajectory``, so this makes it learn the RL policy's
        behaviour instead of the human log. Only the ego's pose/velocity are
        replaced; timestamps, validity, and all other objects are left intact so
        the trajectory stays well-formed (monotonic timestamps, valid masks).
        """
        if self._state is None:
            raise RuntimeError("Call reset()/step() before simulated_log_state().")
        return _splice_sdc_sim_into_log(self._state)

    def ego_trajectories(self) -> dict[str, Any]:
        """Returns the ego's rolled-out (policy) trajectory and the logged
        (expert) trajectory plus the goal, as host NumPy lists, for the current
        episode. Intended for offline inspection of what the policy actually did.

        ``sim_*`` arrays cover only the steps the policy actually drove (valid
        steps up to the current ``timestep``); ``log_*`` is the full expert log.
        """
        if self._state is None:
            raise RuntimeError("Call reset()/step() before ego_trajectories().")
        state = self._state
        is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
        ego = int(np.argmax(is_sdc))
        sim = state.sim_trajectory
        log = state.log_trajectory
        t = int(np.asarray(state.timestep))

        sim_valid = np.asarray(sim.valid)[ego].astype(bool)
        steps = np.arange(sim_valid.shape[0])
        simulated = (steps <= t) & sim_valid

        def ego_row(field: Any) -> np.ndarray:
            return np.asarray(field)[ego]

        goal = np.asarray(self._goal_xy, dtype=np.float32)
        return {
            "timestep": t,
            "goal_xy": [float(goal[0]), float(goal[1])],
            "sim_x": ego_row(sim.x)[simulated].astype(float).tolist(),
            "sim_y": ego_row(sim.y)[simulated].astype(float).tolist(),
            "sim_yaw": ego_row(sim.yaw)[simulated].astype(float).tolist(),
            "sim_vel_x": ego_row(sim.vel_x)[simulated].astype(float).tolist(),
            "sim_vel_y": ego_row(sim.vel_y)[simulated].astype(float).tolist(),
            "log_x": ego_row(log.x).astype(float).tolist(),
            "log_y": ego_row(log.y).astype(float).tolist(),
            "log_yaw": ego_row(log.yaw).astype(float).tolist(),
            "log_valid": ego_row(log.valid).astype(bool).tolist(),
        }


class StreamingWaymaxGymEnv(WaymaxGymEnv):
    """A :class:`WaymaxGymEnv` that resets from an infinite scenario stream.

    ``WaymaxGymEnv`` resets by index into a :class:`~rl.scenario_source.ScenarioSource`,
    which caches every scenario it might be asked for as a host NumPy pytree --
    fine for a few hundred failure cases, but the full WOMD training split is
    ~500k scenarios and would not fit in host RAM. Pass a
    :class:`~rl.scenario_source.StreamingScenarioSource` (built from
    ``rl.scenario_source.make_expert_scenario_generator`` over the *whole*
    training glob, not just the non-failure shards BC uses) and each reset pulls
    the next scenario off the stream instead.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)  # seeds self.np_random; skip the index-based base reset
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        scen_np = self._source.sample_scenario(self._np_rng)
        return self._reset_from_scenario(scen_np, spec_dict={})


def _splice_sdc_sim_into_log(state: Any) -> Any:
    """Overwrites the SDC's logged pose/velocity with its simulated values for
    the steps that were simulated (``sim_trajectory.valid`` up to ``timestep``).
    Unbatched state (prefix shape ``()``) expected.
    """
    is_sdc = state.object_metadata.is_sdc  # [N]
    ego = jnp.argmax(is_sdc.astype(jnp.int32))
    sim = state.sim_trajectory
    log = state.log_trajectory

    num_t = log.x.shape[-1]
    step_idx = jnp.arange(num_t)
    # Steps that were genuinely simulated (and are valid) for the ego.
    simulated = (step_idx <= state.timestep) & sim.valid[ego]  # [T]

    def blend(sim_field: jax.Array, log_field: jax.Array) -> jax.Array:
        new_ego = jnp.where(simulated, sim_field[ego], log_field[ego])
        return log_field.at[ego].set(new_ego)

    new_log = log.replace(
        x=blend(sim.x, log.x),
        y=blend(sim.y, log.y),
        yaw=blend(sim.yaw, log.yaw),
        vel_x=blend(sim.vel_x, log.vel_x),
        vel_y=blend(sim.vel_y, log.vel_y),
    )
    return state.replace(log_trajectory=new_log)


def _compute_route(scen_np: Any) -> tuple[np.ndarray, np.ndarray]:
    """Returns the ego's logged path as ``(route_xy[M, 2], cum_arclen[M])``.

    The expert log is on-road and collision-free, so advancing along it is a
    feasible proxy objective. Only valid logged steps are used, in time order.
    """
    is_sdc = np.asarray(scen_np.object_metadata.is_sdc).astype(bool)
    ego = int(np.argmax(is_sdc))
    x = np.asarray(scen_np.log_trajectory.x)[ego]
    y = np.asarray(scen_np.log_trajectory.y)[ego]
    valid = np.asarray(scen_np.log_trajectory.valid)[ego].astype(bool)
    xy = np.stack([x[valid], y[valid]], axis=-1).astype(np.float32)
    if xy.shape[0] == 0:
        xy = np.zeros((1, 2), dtype=np.float32)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1) if xy.shape[0] > 1 else np.zeros((0,))
    cum_s = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    return xy, cum_s


def _project_to_route(
    ego_xy: np.ndarray,
    route_xy: np.ndarray,
    cum_s: np.ndarray,
    prev_s: float | None = None,
    back_window_m: float = 5.0,
    fwd_window_m: float = 25.0,
) -> tuple[float, float]:
    """Projects ``ego_xy`` onto the route polyline.

    Returns ``(arclength_at_projection, lateral_distance)``.

    The nearest segment is searched *locally* around ``prev_s`` rather than
    globally. A global argmin makes the projection teleport wherever the route
    passes near itself -- at an intersection the ego re-approaches, or a path
    that doubles back -- and since the progression reward is ``s - prev_s``, a
    teleport forward pays out metres of progress the ego never drove (and a
    teleport backward charges a penalty it never earned). Restricting the search
    to ``[prev_s - back_window_m, prev_s + fwd_window_m]`` keeps the projection
    on the branch the ego is actually travelling. ``prev_s=None`` (episode
    reset) still searches globally, which is what we want with no history.
    """
    if route_xy.shape[0] < 2:
        d = float(np.linalg.norm(ego_xy - route_xy[0]))
        return 0.0, d
    a = route_xy[:-1]
    b = route_xy[1:]
    ab = b - a
    ab_sq = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-6)
    t = np.einsum("ij,ij->i", ego_xy - a, ab) / ab_sq
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dists = np.linalg.norm(proj - ego_xy, axis=1)

    if prev_s is not None:
        # A segment is a candidate if it overlaps the window at all, so the
        # ego is never boxed out of the segment it is standing on.
        in_window = (cum_s[1:] >= prev_s - back_window_m) & (
            cum_s[:-1] <= prev_s + fwd_window_m
        )
        if in_window.any():
            dists = np.where(in_window, dists, np.inf)

    k = int(np.argmin(dists))
    s = float(cum_s[k] + t[k] * (cum_s[k + 1] - cum_s[k]))
    return s, float(dists[k])


def _compute_goal_xy(scen_np: Any) -> np.ndarray:
    """Goal = ego's last valid logged (x, y) position (matches goal_reaching.py)."""
    is_sdc = np.asarray(scen_np.object_metadata.is_sdc).astype(bool)
    if is_sdc.ndim != 1:
        raise ValueError(
            f"Expected unbatched scenario (is_sdc shape {is_sdc.shape}); "
            "got a batched state — slice batch dim before calling _compute_goal_xy."
        )
    ego = int(np.argmax(is_sdc))
    x = np.asarray(scen_np.log_trajectory.x)[ego]
    y = np.asarray(scen_np.log_trajectory.y)[ego]
    valid = np.asarray(scen_np.log_trajectory.valid)[ego].astype(bool)
    valid_t = np.flatnonzero(valid)
    t = int(valid_t[-1]) if valid_t.size > 0 else 0
    return np.array([x[t], y[t]], dtype=np.float32)


def _one_hot_valid(idx: jax.Array, num_classes: int) -> jax.Array:
    """One-hot over ``num_classes``, clamping out-of-range ids into range."""
    return jax.nn.one_hot(jnp.clip(idx, 0, num_classes - 1), num_classes)


def _one_hot_tl_state(idx: jax.Array, num_classes: int) -> jax.Array:
    """One-hot over traffic-light states 1..``num_classes``, dropping UNKNOWN.

    Mirrors V-Max: ``one_hot(state, num_classes + 1)[..., 1:]``. Waymax state 0
    is UNKNOWN and must map to an all-zero row, not to its own channel -- a plain
    ``one_hot(state, 8)`` both gives UNKNOWN a channel and shifts states 1..7 down
    by one, aliasing state 8 onto state 7 once clipped.
    """
    return jax.nn.one_hot(idx, num_classes + 1)[..., 1:]


def _past_window(traj: Any, timestep: jax.Array, num_steps: int):
    """Slice ``num_steps`` trajectory frames ending at (and including) ``timestep``.

    ``jax.lax.dynamic_slice`` clamps the start index into range, so the early
    part of an episode simply repeats the earliest available frames rather than
    reading out of bounds.
    """
    start = timestep - (num_steps - 1)
    return datatypes.dynamic_slice(traj, start, num_steps, axis=-1)


def _compute_observation(state: Any, goal_xy: jax.Array):
    """Builds the V-Max-aligned, SDC-centric observation vector (JIT-compiled).

    Blocks, in order, matching ``rl.obs_layout.DEFAULT_OBS_BLOCKS``: ``sdc``,
    ``agents``, ``roadgraph``, ``traffic_lights``, ``path_target``, ``goal``.
    Feature sets and sizes mirror V-Max's ``repro_sac_v2`` observation_config
    (see ``rl/obs_layout.py``), computed straight from raw WOMD.

    Everything is expressed in the ego frame at the current timestep. Rows for
    padded/invalid entities are zeroed and carry a 0 validity bit, so the
    encoder can mask them.

    Returns (obs[obs_dim] float32, goal_distance scalar float32,
    ego_xy[2] float32 world position).
    """
    is_sdc = state.object_metadata.is_sdc  # [N]
    ego_idx = jnp.argmax(is_sdc.astype(jnp.int32))
    P = obs_layout.OBS_PAST_NUM_STEPS

    # --- ego pose at the current step defines the frame ---------------------- #
    cur = datatypes.dynamic_slice(state.sim_trajectory, state.timestep, 1, axis=-1)
    ex, ey = cur.x[ego_idx, 0], cur.y[ego_idx, 0]
    eyaw = cur.yaw[ego_idx, 0]
    ch, sh = jnp.cos(eyaw), jnp.sin(eyaw)

    def to_ego(px, py):
        dx, dy = px - ex, py - ey
        return ch * dx + sh * dy, -sh * dx + ch * dy

    def rot(px, py):
        return ch * px + sh * py, -sh * px + ch * py

    def norm_xy(v):
        return jnp.clip(v, -obs_layout.MAX_METERS, obs_layout.MAX_METERS) / obs_layout.MAX_METERS

    # --- object history: [N, P] ------------------------------------------- #
    hist = _past_window(state.sim_trajectory, state.timestep, P)
    hx, hy = to_ego(hist.x, hist.y)
    hvx, hvy = rot(hist.vel_x, hist.vel_y)
    hyaw = _wrap_to_pi(hist.yaw - eyaw)
    hvalid = hist.valid

    def object_rows(sel_idx: jax.Array, sel_ok: jax.Array) -> jax.Array:
        """Assemble [K, P, 8] rows (7 features + valid) for the selected objects."""
        valid = hvalid[sel_idx] & sel_ok[:, None]
        feats = jnp.stack(
            [
                norm_xy(hx[sel_idx]),
                norm_xy(hy[sel_idx]),
                jnp.clip(hvx[sel_idx], -obs_layout.MAX_SPEED, obs_layout.MAX_SPEED) / obs_layout.MAX_SPEED,
                jnp.clip(hvy[sel_idx], -obs_layout.MAX_SPEED, obs_layout.MAX_SPEED) / obs_layout.MAX_SPEED,
                hyaw[sel_idx],
                hist.length[sel_idx] / obs_layout.MAX_METERS,
                hist.width[sel_idx] / obs_layout.MAX_METERS,
                valid.astype(jnp.float32),
            ],
            axis=-1,
        )
        return jnp.where(valid[..., None], feats, 0.0)

    # sdc block: the ego's own history (always valid).
    sdc_block = object_rows(ego_idx[None], jnp.ones((1,), dtype=bool)).reshape(-1)

    # agents block: K nearest valid non-ego objects, ranked at the current step.
    ox_now, oy_now = hx[:, -1], hy[:, -1]
    other_valid = hvalid[:, -1] & (~is_sdc)
    dist_now = jnp.sqrt(ox_now ** 2 + oy_now ** 2)
    score = jnp.where(other_valid, -dist_now, -jnp.inf)
    top_vals, top_idx = _top_k_padded(score, obs_layout.NUM_CLOSEST_OBJECTS)
    agents_block = object_rows(top_idx, jnp.isfinite(top_vals)).reshape(-1)

    # --- roadgraph block --------------------------------------------------- #
    # Road edges only (what the offroad metric is computed against), inside a
    # front-biased ego box, decimated by `interval`, then top-k nearest.
    rg = state.roadgraph_points
    rx, ry = to_ego(rg.x, rg.y)
    rdx, rdy = rot(rg.dir_x, rg.dir_y)

    is_edge = jnp.zeros_like(rg.valid)
    for t in obs_layout.ROADGRAPH_ELEMENT_TYPES:
        is_edge = is_edge | (rg.types == t)
    in_box = (
        (rx <= obs_layout.METERS_BOX_FRONT)
        & (rx >= -obs_layout.METERS_BOX_BACK)
        & (ry <= obs_layout.METERS_BOX_LEFT)
        & (ry >= -obs_layout.METERS_BOX_RIGHT)
    )
    keep_stride = (jnp.arange(rg.x.shape[-1]) % obs_layout.ROADGRAPH_INTERVAL) == 0
    rvalid = rg.valid & is_edge & in_box & keep_stride

    rdist = jnp.sqrt(rx ** 2 + ry ** 2)
    rscore = jnp.where(rvalid, -rdist, -jnp.inf)
    r_top_vals, r_top_idx = _top_k_padded(rscore, obs_layout.ROADGRAPH_TOP_K)
    rsel = jnp.isfinite(r_top_vals)
    rg_rows = jnp.stack(
        [
            norm_xy(rx[r_top_idx]),
            norm_xy(ry[r_top_idx]),
            rdx[r_top_idx],
            rdy[r_top_idx],
            rsel.astype(jnp.float32),
        ],
        axis=-1,
    )
    roadgraph_block = jnp.where(rsel[:, None], rg_rows, 0.0).reshape(-1)

    # --- traffic lights block (K nearest, with history) -------------------- #
    tl_hist = _past_window(state.log_traffic_light, state.timestep, P)
    tlx, tly = to_ego(tl_hist.x, tl_hist.y)          # [L, P]
    tl_valid_all = tl_hist.valid
    tl_dist_now = jnp.sqrt(tlx[:, -1] ** 2 + tly[:, -1] ** 2)
    tl_score = jnp.where(tl_valid_all[:, -1], -tl_dist_now, -jnp.inf)
    tl_top_vals, tl_top_idx = _top_k_padded(tl_score, obs_layout.NUM_CLOSEST_TRAFFIC_LIGHTS)
    tl_ok = jnp.isfinite(tl_top_vals)
    tl_valid = tl_valid_all[tl_top_idx] & tl_ok[:, None]
    tl_rows = jnp.concatenate(
        [
            norm_xy(tlx[tl_top_idx])[..., None],
            norm_xy(tly[tl_top_idx])[..., None],
            _one_hot_tl_state(tl_hist.state[tl_top_idx], obs_layout.NUM_TL_STATES),
            tl_valid.astype(jnp.float32)[..., None],
        ],
        axis=-1,
    )
    traffic_lights_block = jnp.where(tl_valid[..., None], tl_rows, 0.0).reshape(-1)

    # --- path_target block (the SDC route) --------------------------------- #
    # V-Max takes the longest on-route path and samples `num_points` every
    # `points_gap`. This is the block the route reward is defined against, so
    # without it the policy is scored on something it cannot see.
    path_block = _path_target_features(state, to_ego, norm_xy)

    # --- goal block (ours; V-Max's repro_sac_v2 has no goal features) ------- #
    gx, gy = to_ego(goal_xy[0], goal_xy[1])
    goal_dist = jnp.sqrt(gx ** 2 + gy ** 2)
    gth = jnp.arctan2(gy, gx)
    goal_block = jnp.concatenate(
        [
            jnp.stack(
                [
                    norm_xy(gx),
                    norm_xy(gy),
                    jnp.clip(goal_dist / obs_layout.MAX_METERS, 0.0, 1.0),
                    jnp.cos(gth),
                    jnp.sin(gth),
                ]
            ),
            jnp.ones((1,)),
        ]
    )

    obs = jnp.concatenate(
        [
            sdc_block,
            agents_block,
            roadgraph_block,
            traffic_lights_block,
            path_block,
            goal_block,
        ]
    )
    obs = jnp.clip(obs, -_OBS_CLIP, _OBS_CLIP)
    ego_xy = jnp.stack([ex, ey]).astype(jnp.float32)
    return obs.astype(jnp.float32), goal_dist.astype(jnp.float32), ego_xy


def _path_target_features(state: Any, to_ego, norm_xy) -> jax.Array:
    """``num_points`` ego-frame route points, sampled every ``points_gap``.

    Mirrors V-Max's ``VecFeaturesExtractor._build_target_features``: pick the
    on-route SDC path with the most valid points, find the path point nearest the
    ego, then sample every ``points_gap``th point *ahead of that one*. The window
    has to move with the ego -- sampling absolute indices from the start of the
    path means the targets fall behind the vehicle as it drives, so the policy
    loses the upcoming route exactly while being rewarded for advancing along it.
    Emits zeros when the scenario was loaded without SDC paths
    (``include_sdc_paths=False``), so the observation stays a fixed width.
    """
    n_pts = obs_layout.PATH_TARGET_NUM_POINTS
    gap = obs_layout.PATH_TARGET_POINTS_GAP

    paths = getattr(state, "sdc_paths", None)
    if paths is None:
        return jnp.zeros((n_pts * 2,), dtype=jnp.float32)

    # on_route is per-path; broadcast it over that path's points.
    mask = paths.valid & paths.on_route  # [num_paths, num_points]
    best = jnp.argmax(jnp.sum(mask, axis=-1))
    px, py = paths.x[best], paths.y[best]
    pvalid = mask[best]

    # Anchor the window on the valid path point closest to the ego.
    all_ex, all_ey = to_ego(px, py)
    all_d2 = all_ex ** 2 + all_ey ** 2
    cur = jnp.argmin(jnp.where(pvalid, all_d2, jnp.inf))

    last = px.shape[-1] - 1
    idx = cur + jnp.arange(1, n_pts + 1) * gap
    # Past the end of the path there is no route left to point at; clamping would
    # emit the final point `n_pts` times and read as "the route stops here".
    in_range = idx <= last
    idx = jnp.minimum(idx, last)

    ex_, ey_ = all_ex[idx], all_ey[idx]
    sel = pvalid[idx] & in_range
    rows = jnp.stack([norm_xy(ex_), norm_xy(ey_)], axis=-1)
    return jnp.where(sel[:, None], rows, 0.0).reshape(-1)
