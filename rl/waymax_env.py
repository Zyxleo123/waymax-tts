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

from rl.scenario_source import ScenarioSource

# Observation layout constants.
_K_AGENTS = 8          # nearest other agents included in the observation
_K_ROADGRAPH = 20      # nearest roadgraph points included in the observation
_POS_NORM = 50.0       # meters
_VEL_NORM = 20.0       # m/s
_SPEED_NORM = 20.0     # m/s
_DIST_NORM = 50.0      # meters
_RG_RANGE = 50.0       # meters; roadgraph points beyond this are ignored
_OBS_CLIP = 10.0

# Default physical limits for the "delta" (next-position) action space, applied
# per simulator step. The policy outputs values in [-1, 1] that are scaled to
# these ranges before being passed to the DeltaLocal dynamics model.
_DELTA_MAX_DX = 6.0     # meters (forward/back in ego frame)
_DELTA_MAX_DY = 6.0     # meters (left/right in ego frame)
_DELTA_MAX_DYAW = float(np.pi)  # radians

_VALID_ACTION_SPACES = ("bicycle", "delta")


@dataclasses.dataclass
class RewardConfig:
    """Weights for the shaped reward."""

    progress: float = 1.0       # reward per meter of progress (straight-line or route)
    step_penalty: float = 0.0   # constant per-step penalty
    action_penalty: float = 0.01  # penalty on sum(action^2)
    collision: float = -10.0    # one-time penalty; ends the episode
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


def observation_dim() -> int:
    return 2 + 5 + _K_AGENTS * 7 + _K_ROADGRAPH * 2


def _wrap_to_pi(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


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
        scen = jax.tree_util.tree_map(jnp.asarray, scen_np)
        self._state = self._jit_reset(scen)
        self._goal_xy = jnp.asarray(_compute_goal_xy(scen_np), dtype=jnp.float32)

        # Expert log path (on-road, collision-free) used for the route reward.
        self._route_xy, self._route_s = _compute_route(scen_np)

        obs, goal_dist, ego_xy = self._jit_obs(self._state, self._goal_xy)
        self._prev_dist = float(goal_dist)
        self._prev_s, _ = _project_to_route(np.asarray(ego_xy), self._route_xy, self._route_s)
        self._step_count = 0

        num_timesteps = int(np.asarray(scen_np.log_trajectory.x).shape[-1])
        init_steps = self._env.config.init_steps
        self._max_steps = min(self._max_episode_steps_cap, num_timesteps - init_steps)

        info = {
            "goal_dist": self._prev_dist,
            "scenario": dataclasses.asdict(self._source.spec(self._seq_cursor - 1))
            if self._sequential
            else {},
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
        if rc.route_reward:
            # Progress = advancement along the expert path; penalize deviation.
            s, lateral = _project_to_route(
                np.asarray(ego_xy), self._route_xy, self._route_s
            )
            reward = rc.progress * (s - self._prev_s)
            reward -= rc.lateral_penalty * lateral
            self._prev_s = s
        else:
            reward = rc.progress * (self._prev_dist - goal_dist)
        reward -= rc.step_penalty
        reward -= rc.action_penalty * float(np.sum(action ** 2))

        terminated = False
        if collision:
            reward += rc.collision
            terminated = terminated or rc.terminate_on_collision
        if off:
            reward += rc.offroad
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
    ego_xy: np.ndarray, route_xy: np.ndarray, cum_s: np.ndarray
) -> tuple[float, float]:
    """Projects ``ego_xy`` onto the route polyline.

    Returns ``(arclength_at_projection, lateral_distance)``.
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
    k = int(np.argmin(dists))
    s = float(cum_s[k] + t[k] * (cum_s[k + 1] - cum_s[k]))
    return s, float(dists[k])


def _compute_goal_xy(scen_np: Any) -> np.ndarray:
    """Goal = ego's last valid logged (x, y) position (matches goal_reaching.py)."""
    is_sdc = np.asarray(scen_np.object_metadata.is_sdc).astype(bool)
    ego = int(np.argmax(is_sdc))
    x = np.asarray(scen_np.log_trajectory.x)[ego]
    y = np.asarray(scen_np.log_trajectory.y)[ego]
    valid = np.asarray(scen_np.log_trajectory.valid)[ego].astype(bool)
    valid_t = np.flatnonzero(valid)
    t = int(valid_t[-1]) if valid_t.size > 0 else 0
    return np.array([x[t], y[t]], dtype=np.float32)


def _compute_observation(state: Any, goal_xy: jax.Array):
    """Builds a compact SDC-centric observation vector (JIT-compiled).

    Returns (obs[obs_dim] float32, goal_distance scalar float32,
    ego_xy[2] float32 world position).
    """
    is_sdc = state.object_metadata.is_sdc  # [N]
    ego_idx = jnp.argmax(is_sdc.astype(jnp.int32))

    cur = datatypes.dynamic_slice(state.sim_trajectory, state.timestep, 1, axis=-1)
    x = cur.x[:, 0]
    y = cur.y[:, 0]
    yaw = cur.yaw[:, 0]
    vx = cur.vel_x[:, 0]
    vy = cur.vel_y[:, 0]
    valid = cur.valid[:, 0]

    ex, ey, eyaw = x[ego_idx], y[ego_idx], yaw[ego_idx]
    evx, evy = vx[ego_idx], vy[ego_idx]
    ch, sh = jnp.cos(eyaw), jnp.sin(eyaw)

    def to_ego(px, py):
        dx, dy = px - ex, py - ey
        return ch * dx + sh * dy, -sh * dx + ch * dy

    def rot(px, py):
        return ch * px + sh * py, -sh * px + ch * py

    # Ego block.
    ego_speed = jnp.sqrt(evx ** 2 + evy ** 2)
    num_t = jnp.asarray(state.log_trajectory.num_timesteps, dtype=jnp.float32)
    t_frac = state.timestep.astype(jnp.float32) / jnp.maximum(num_t, 1.0)
    ego_block = jnp.stack([ego_speed / _SPEED_NORM, t_frac])

    # Goal block.
    gx, gy = to_ego(goal_xy[0], goal_xy[1])
    goal_dist = jnp.sqrt(gx ** 2 + gy ** 2)
    gth = jnp.arctan2(gy, gx)
    goal_block = jnp.stack(
        [gx / _POS_NORM, gy / _POS_NORM, goal_dist / _DIST_NORM, jnp.cos(gth), jnp.sin(gth)]
    )

    # Other agents block (K nearest valid non-ego objects).
    ox, oy = to_ego(x, y)
    ovx, ovy = rot(vx, vy)
    oyaw = _wrap_to_pi(yaw - eyaw)
    other_valid = valid & (~is_sdc)
    dist = jnp.sqrt(ox ** 2 + oy ** 2)
    score = jnp.where(other_valid, -dist, -1e9)
    top_vals, top_idx = jax.lax.top_k(score, _K_AGENTS)
    sel = top_vals > -1e8
    agent_feat = jnp.stack(
        [
            ox[top_idx] / _POS_NORM,
            oy[top_idx] / _POS_NORM,
            ovx[top_idx] / _VEL_NORM,
            ovy[top_idx] / _VEL_NORM,
            jnp.cos(oyaw[top_idx]),
            jnp.sin(oyaw[top_idx]),
            sel.astype(jnp.float32),
        ],
        axis=-1,
    )
    agent_feat = jnp.where(sel[:, None], agent_feat, 0.0).reshape(-1)

    # Roadgraph block (K nearest valid points within range).
    rg = state.roadgraph_points
    rx, ry = to_ego(rg.x, rg.y)
    rdist = jnp.sqrt(rx ** 2 + ry ** 2)
    rvalid = rg.valid & (rdist < _RG_RANGE)
    rscore = jnp.where(rvalid, -rdist, -1e9)
    r_top_vals, r_top_idx = jax.lax.top_k(rscore, _K_ROADGRAPH)
    rsel = r_top_vals > -1e8
    rg_feat = jnp.stack([rx[r_top_idx] / _POS_NORM, ry[r_top_idx] / _POS_NORM], axis=-1)
    rg_feat = jnp.where(rsel[:, None], rg_feat, 0.0).reshape(-1)

    obs = jnp.concatenate([ego_block, goal_block, agent_feat, rg_feat])
    obs = jnp.clip(obs, -_OBS_CLIP, _OBS_CLIP)
    ego_xy = jnp.stack([ex, ey]).astype(jnp.float32)
    return obs.astype(jnp.float32), goal_dist.astype(jnp.float32), ego_xy
