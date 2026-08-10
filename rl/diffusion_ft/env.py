"""``DiffusionWaymaxEnv``: the one trustworthy batched environment (Phase 1).

The outer action of this MDP is *one sampled diffusion trajectory*. A step:

1. builds features from the current *simulated* ego state (executed motion, not
   the human log -- see :func:`_splice_sim_into_log_batched`),
2. samples a trajectory from the policy (or accepts one passed in, for
   best-of-N / ES search),
3. postprocesses it to world coordinates,
4. executes only the configured prefix through ``StateDynamics`` (the backend
   that matches the state-trajectory action representation and the existing
   diffusion evaluation),
5. returns the batched simulator reward (from :mod:`rl.diffusion_ft.reward`, the
   one shared reward function), the next state, termination flags, and the
   per-step reward components.

Design invariants (Phase 1):

* **``log_trajectory`` is immutable.** Executed motion lives in
  ``sim_trajectory`` (Waymax writes it there as we step ``StateDynamics``). The
  goal and the expert route are read from the untouched log.
* **Real goal timestep per scene.** No ``goal_step_override=90``; the goal step
  is each scene's last valid ego log step (:func:`infer_goals_from_sim_state`).
* **Batched == single.** Every quantity is computed over the batch axis with the
  same recurrence a single scene would use, so a permuted batch permutes the
  outputs identically (Gate 1).

This env owns the policy for on-policy collection, but ``step`` also accepts an
externally supplied ``trajectory_world_bt5`` so the search baselines
(best-of-N, ES) can score their own candidates through the identical executor
and reward.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from data.postprocess import postprocess_predictions
from data.preprocess import preprocess_simulator_state
from simulation.evaluation_utils import infer_goals_from_sim_state

from rl.diffusion_ft.checkpoint import DiffusionCheckpoint
from rl.diffusion_ft import reward as reward_mod
from rl.diffusion_ft.reward import RewardConfig, RewardState


# --------------------------------------------------------------------------- #
# Batched helpers
# --------------------------------------------------------------------------- #
def _ego_indices_b(state: Any) -> jax.Array:
    """One SDC index per world, ``[B]``."""
    is_sdc_bn = jnp.asarray(state.object_metadata.is_sdc).astype(jnp.int32)
    return jnp.argmax(is_sdc_bn, axis=1)


def _splice_sim_into_log_batched(state: Any) -> Any:
    """Batched analog of ``rl.waymax_env._splice_sdc_sim_into_log``.

    Returns a *scratch* copy of ``state`` in which the ego's ``log_trajectory``
    rows are overwritten by its ``sim_trajectory`` for every already-simulated,
    valid step (``step <= timestep``). This is used **only** to feed the
    preprocessor the executed history; the caller's real state keeps its
    ``log_trajectory`` untouched.
    """
    is_sdc_bn = jnp.asarray(state.object_metadata.is_sdc)  # [B, N]
    ego_b = _ego_indices_b(state)                          # [B]
    sim = state.sim_trajectory
    log = state.log_trajectory
    num_t = log.x.shape[-1]
    step_idx = jnp.arange(num_t)                           # [T]
    # simulated & valid for the ego, per world: [B, T]
    ego_onehot_bn = jax.nn.one_hot(ego_b, is_sdc_bn.shape[1], dtype=bool)  # [B, N]

    def blend(sim_f: jax.Array, log_f: jax.Array) -> jax.Array:
        # ego row (per world): [B, T]
        sim_ego_bt = jnp.take_along_axis(sim_f, ego_b[:, None, None], axis=1)[:, 0]
        log_ego_bt = jnp.take_along_axis(log_f, ego_b[:, None, None], axis=1)[:, 0]
        sim_valid_bt = jnp.take_along_axis(
            sim.valid, ego_b[:, None, None], axis=1
        )[:, 0]
        use_bt = (step_idx[None, :] <= state.timestep) & sim_valid_bt
        new_ego_bt = jnp.where(use_bt, sim_ego_bt, log_ego_bt)  # [B, T]
        # scatter the new ego row back into [B, N, T]
        return jnp.where(ego_onehot_bn[:, :, None], new_ego_bt[:, None, :], log_f)

    new_log = log.replace(
        x=blend(sim.x, log.x),
        y=blend(sim.y, log.y),
        yaw=blend(sim.yaw, log.yaw),
        vel_x=blend(sim.vel_x, log.vel_x),
        vel_y=blend(sim.vel_y, log.vel_y),
    )
    return state.replace(log_trajectory=new_log)


def _ego_pose_now_b(state: Any) -> jax.Array:
    """Ego ``[x, y]`` at the current ``sim_trajectory`` timestep, ``[B, 2]``."""
    ego_b = _ego_indices_b(state)
    cur = datatypes.dynamic_slice(state.sim_trajectory, state.timestep, 1, axis=-1)
    x_b = jnp.take_along_axis(cur.x[..., 0], ego_b[:, None], axis=1)[:, 0]
    y_b = jnp.take_along_axis(cur.y[..., 0], ego_b[:, None], axis=1)[:, 0]
    return jnp.stack([x_b, y_b], axis=-1)


# --------------------------------------------------------------------------- #
@dataclass
class StepOutput:
    """One outer (replan) step's result."""

    features: dict[str, jax.Array]          # observation for the next replan
    reward_b: jax.Array                     # [B] summed over executed world steps
    terminated_b: jax.Array                 # [B]
    truncated_b: jax.Array                  # [B]
    per_step_reward_bk: jax.Array           # [B, prefix] atomic world-step rewards
    components: dict[str, jax.Array]        # summed reward components, each [B]
    trajectory_world_bt5: jax.Array         # the executed candidate (world coords)


class DiffusionWaymaxEnv:
    """Batched diffusion-policy environment; outer action = one trajectory."""

    def __init__(
        self,
        ckpt: DiffusionCheckpoint,
        *,
        reward_config: RewardConfig | None = None,
        max_num_objects: int = 128,
        start_timestep: int | None = None,
        replan_interval_steps: int = 10,
        use_ema: bool = False,
        seed: int = 0,
        max_replans: int | None = None,
    ):
        self._ckpt = ckpt
        self._cfg = ckpt.preprocess_cfg
        self._reward_cfg = reward_config or RewardConfig()
        self._replan = int(replan_interval_steps)
        # Optional cap on the number of outer replan steps per episode (bounds
        # cost for CPU smoke tests). None == run to the scenario horizon.
        self._max_replans = None if max_replans is None else int(max_replans)
        self._max_num_objects = int(max_num_objects)
        self._rng = jax.random.PRNGKey(int(seed))

        self._policy = ckpt.merge(use_ema=use_ema)

        env_cfg = waymax_config.EnvironmentConfig(
            max_num_objects=self._max_num_objects,
            controlled_object=waymax_config.ObjectType.SDC,
            compute_reward=False,
            metrics=waymax_config.MetricsConfig(metrics_to_run=("overlap", "offroad")),
        )
        self._env = waymax_env.PlanningAgentEnvironment(
            dynamics_model=waymax_dynamics.StateDynamics(),
            config=env_cfg,
        )
        self._start_timestep = (
            int(start_timestep) if start_timestep is not None
            else int(self._env.config.init_steps)
        )

        env_obj = self._env

        def _metric_values(state):
            m = env_obj.metrics(state)
            return m["overlap"].value, m["offroad"].value

        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)
        self._jit_metrics = jax.jit(_metric_values)
        self._jit_condition = jax.jit(self._policy.compute_condition)

        # Per-episode batched state.
        self._state: Any = None
        self._goal_xy_b2: jax.Array | None = None
        self._goal_t_b: jax.Array | None = None
        self._reward_state: RewardState | None = None
        self._batch_size: int = 0

    # ------------------------------------------------------------------ #
    @property
    def sim_state(self) -> Any:
        return self._state

    @property
    def timestep(self) -> int:
        # The JAX simulator state is the single source of truth for the current
        # timestep. Tracking a separate Python counter drifted from it (Waymax
        # reset sets state.timestep = init_steps - 1, not init_steps), which
        # anchored every replan observation one step ahead of the executed motion.
        return int(self._state.timestep)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # ------------------------------------------------------------------ #
    def reset(self, scenarios: Any, *, seed: int | None = None) -> dict[str, jax.Array]:
        """Reset from a batched (prefix ``[B]``) Waymax scenario pytree.

        ``scenarios`` is a device/host pytree with a leading batch axis (stack B
        unbatched scenarios, e.g. from :meth:`build_batched_scenarios`).
        Returns the first observation (preprocessed feature dict).
        """
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        scen = jax.tree_util.tree_map(jnp.asarray, scenarios)
        self._state = self._jit_reset(scen)
        self._batch_size = int(jnp.asarray(scen.log_trajectory.x).shape[0])

        goal_xy_b2, goal_t_b, _ = infer_goals_from_sim_state(self._state)
        self._goal_xy_b2 = jnp.asarray(goal_xy_b2, dtype=jnp.float32)
        self._goal_t_b = jnp.asarray(goal_t_b, dtype=jnp.int32)

        init_goal_dist_b = jnp.linalg.norm(
            _ego_pose_now_b(self._state) - self._goal_xy_b2, axis=-1
        )
        # Route reward is off by default; init_s is 0 and unused unless enabled.
        self._reward_state = reward_mod.init_reward_state(
            init_goal_dist_b=init_goal_dist_b,
            init_s_b=jnp.zeros_like(init_goal_dist_b),
        )
        return self._build_features()

    # ------------------------------------------------------------------ #
    def _build_features(self) -> dict[str, jax.Array]:
        """Preprocess the current *executed* state into policy features.

        Uses a scratch state whose ego log rows are the executed sim motion (so
        the preprocessor observes what the policy actually drove), anchored at
        the current timestep, with each scene's real goal step and goal xy.
        """
        scratch = _splice_sim_into_log_batched(self._state)
        self._rng, key = jax.random.split(self._rng)
        pre_batch, _ = preprocess_simulator_state(
            scratch,
            key,
            self._cfg,
            anchor_step_override=self.timestep,
            goal_step_override=self._goal_t_b,   # array: real per-scene goal step
            goal_xy_override=self._goal_xy_b2,
        )
        features = dict(pre_batch.features)
        bsz = self._batch_size
        features["inst_features"] = jnp.zeros((bsz, self._cfg.inst_dim), dtype=jnp.float32)
        features["inst_valid"] = jnp.zeros((bsz,), dtype=jnp.float32)
        features["subgoal_xy"] = jnp.zeros((bsz, 2), dtype=jnp.float32)
        features["subgoal_valid"] = jnp.zeros((bsz,), dtype=jnp.float32)
        self._last_pre_batch = pre_batch
        return features

    # ------------------------------------------------------------------ #
    def sample_trajectory(
        self, features: dict[str, jax.Array], *, rng: jax.Array | None = None
    ) -> jax.Array:
        """Sample one trajectory per world and postprocess to world coords.

        Returns ``trajectory_world_bt5`` of shape ``[B, world_steps, 5]``
        (``x, y, yaw, vel_x, vel_y``), aligned to ``world_t_seconds`` with the
        anchor (current) state as its first element.
        """
        if rng is None:
            self._rng, rng = jax.random.split(self._rng)
        cond_bf, inst_cond_bf = self._jit_condition(features)
        norm_btd = self._policy.sample_from_condition(
            cond_bf, inst_cond_bf, features["inst_valid"], rng=rng
        )
        post = postprocess_predictions(norm_btd, self._last_pre_batch.aux, self._cfg)
        return jnp.asarray(post["trajectory_world_world_dt"], dtype=jnp.float32)

    # ------------------------------------------------------------------ #
    # Glue for external samplers (DPPO / search): expose the frozen condition
    # and the norm->world postprocess so a trainer can sample trajectories with
    # its own (frozen-early / trainable-late) denoiser and feed them to step().
    # ------------------------------------------------------------------ #
    def compute_condition(self, features: dict[str, jax.Array]):
        """Frozen scene/instruction condition ``(cond1, cond2)`` for ``features``.

        The scene tokenizer + instruction/subgoal encoders are frozen, so this is
        the fixed conditioning the DPPO-trainable denoiser is applied on top of.
        """
        return self._jit_condition(features)

    def norm_to_world(self, norm_btd: jax.Array) -> jax.Array:
        """Postprocess a normalized ``[B, T, 5]`` trajectory to world ``[B, W, 5]``
        using the current replan's aux (call after :meth:`_build_features` /
        :meth:`reset`, i.e. once per replan)."""
        post = postprocess_predictions(jnp.asarray(norm_btd), self._last_pre_batch.aux, self._cfg)
        return jnp.asarray(post["trajectory_world_world_dt"], dtype=jnp.float32)

    @property
    def policy(self):
        """The frozen policy (scene tokenizer + base denoiser)."""
        return self._policy

    # ------------------------------------------------------------------ #
    def _execute_prefix(self, trajectory_world_bt5: jax.Array) -> StepOutput:
        """Execute the first ``replan`` world steps of a world-coord trajectory.

        ``trajectory_world_bt5[:, 0]`` is the anchor (current) state; steps
        ``1..replan`` are applied as ``StateDynamics`` actions, advancing the
        simulated trajectory one world step at a time and accumulating reward.
        """
        traj = jnp.asarray(trajectory_world_bt5, dtype=jnp.float32)
        world_steps = int(traj.shape[1])
        bsz = self._batch_size
        # Number of world steps we can still execute: bounded by the prefix, the
        # candidate length, and the scenario horizon.
        horizon = int(jnp.asarray(self._state.log_trajectory.x).shape[-1])
        max_by_horizon = horizon - 1 - self.timestep
        n_exec = int(min(self._replan, world_steps - 1, max_by_horizon))

        per_step = []
        comp_accum: dict[str, jax.Array] = {}
        term_any_b = jnp.zeros((bsz,), dtype=bool)

        for k in range(1, n_exec + 1):
            pose_b5 = traj[:, k, :]  # target [x, y, yaw, vel_x, vel_y]
            # env.step -> PlanningAgentDynamics.compute_update tiles BOTH data and
            # valid over the object axis with the same `x[..., newaxis, :]`, so
            # valid must carry a trailing axis (``[B, 1]``) to tile consistently
            # with data (``[B, 5]``); a bare ``[B]`` mis-tiles to ``[num_objects, B]``.
            action = datatypes.Action(
                data=pose_b5,
                valid=jnp.ones((bsz, 1), dtype=jnp.bool_),
            )
            self._state = self._jit_step(self._state, action)

            overlap_b, offroad_b = self._jit_metrics(self._state)
            # Single-agent current-step metrics are per world; normalize any
            # trailing singleton (``[B,1]``) to ``[B]`` so they broadcast cleanly
            # against goal_dist_b in the reward.
            overlap_b = jnp.reshape(overlap_b, (bsz,))
            offroad_b = jnp.reshape(offroad_b, (bsz,))
            goal_dist_b = jnp.linalg.norm(
                _ego_pose_now_b(self._state) - self._goal_xy_b2, axis=-1
            )
            reward_b, self._reward_state, term_b, comps = reward_mod.reward_step(
                self._reward_cfg,
                self._reward_state,
                goal_dist_b=goal_dist_b,
                overlap_b=overlap_b,
                offroad_b=offroad_b,
            )
            per_step.append(reward_b)
            term_any_b = term_any_b | term_b
            for key, val in comps.items():
                if key.endswith("_flag"):
                    comp_accum[key] = comp_accum.get(key, jnp.zeros((bsz,), bool)) | val
                else:
                    comp_accum[key] = comp_accum.get(key, jnp.zeros((bsz,))) + val

        if per_step:
            per_step_bk = jnp.stack(per_step, axis=-1)  # [B, n_exec]
            reward_sum_b = jnp.sum(per_step_bk, axis=-1)
        else:
            per_step_bk = jnp.zeros((bsz, 0), dtype=jnp.float32)
            reward_sum_b = jnp.zeros((bsz,), dtype=jnp.float32)

        # Truncate any world that ran out of horizon (or hit the prefix cap at
        # the scenario end) without terminating.
        out_of_horizon = (self.timestep >= horizon - 1)
        truncated_b = jnp.full((bsz,), bool(out_of_horizon)) & (~self._reward_state.done)

        next_features = self._build_features()
        return StepOutput(
            features=next_features,
            reward_b=reward_sum_b,
            terminated_b=term_any_b,
            truncated_b=truncated_b,
            per_step_reward_bk=per_step_bk,
            components=comp_accum,
            trajectory_world_bt5=traj,
        )

    # ------------------------------------------------------------------ #
    def step(
        self,
        *,
        trajectory_world_bt5: jax.Array | None = None,
        features: dict[str, jax.Array] | None = None,
        rng: jax.Array | None = None,
    ) -> StepOutput:
        """One outer replan step.

        If ``trajectory_world_bt5`` is given (best-of-N / ES), it is executed
        directly. Otherwise a trajectory is sampled from the current policy using
        ``features`` (defaults to freshly built features for the current state).
        """
        if trajectory_world_bt5 is None:
            if features is None:
                features = self._build_features()
            trajectory_world_bt5 = self.sample_trajectory(features, rng=rng)
        return self._execute_prefix(trajectory_world_bt5)

    # ------------------------------------------------------------------ #
    def ego_log_world_trajectory(self) -> jax.Array:
        """The ego's *logged* world trajectory ``[B, T, 5]`` (for expert replay).

        Layout matches a sampled candidate: ``x, y, yaw, vel_x, vel_y``. Executing
        this from ``start_timestep`` reproduces the human demonstration and must
        yield clean metrics (Gate 1 expert-replay test).
        """
        ego_b = _ego_indices_b(self._state)
        log = self._state.log_trajectory

        def ego_row(field: jax.Array) -> jax.Array:
            return jnp.take_along_axis(field, ego_b[:, None, None], axis=1)[:, 0]

        return jnp.stack(
            [ego_row(log.x), ego_row(log.y), ego_row(log.yaw),
             ego_row(log.vel_x), ego_row(log.vel_y)],
            axis=-1,
        )

    # ------------------------------------------------------------------ #
    def expert_replay_episode(self) -> dict[str, Any]:
        """Roll the whole episode by feeding the ego's *logged* poses as actions.

        Deterministic (no sampling): at each replan the candidate is the log
        window starting at the current timestep, so the ego tracks the human
        demonstration exactly. Used by the Gate 1 expert-replay / parity /
        permutation tests. Returns aggregated metrics, per-step rewards, and the
        episode return.
        """
        log_world_bt5 = self.ego_log_world_trajectory()  # [B, T, 5]
        horizon = int(log_world_bt5.shape[1])
        bsz = self._batch_size

        per_step_chunks: list[jax.Array] = []
        outer_reward_b = jnp.zeros((bsz,), dtype=jnp.float32)
        overlap_any_b = jnp.zeros((bsz,), dtype=bool)
        offroad_any_b = jnp.zeros((bsz,), dtype=bool)
        reached_any_b = jnp.zeros((bsz,), dtype=bool)

        n_replans = 0
        while True:
            t = self.timestep
            if t >= horizon - 1:
                break
            if self._max_replans is not None and n_replans >= self._max_replans:
                break
            n_replans += 1
            window = log_world_bt5[:, t : t + self._replan + 1, :]
            if int(window.shape[1]) < 2:
                break
            out = self._execute_prefix(window)
            per_step_chunks.append(out.per_step_reward_bk)
            outer_reward_b = outer_reward_b + out.reward_b
            overlap_any_b = overlap_any_b | out.components.get(
                "collision_flag", jnp.zeros((bsz,), bool)
            )
            offroad_any_b = offroad_any_b | out.components.get(
                "offroad_flag", jnp.zeros((bsz,), bool)
            )
            reached_any_b = reached_any_b | out.components.get(
                "reached_flag", jnp.zeros((bsz,), bool)
            )
            if bool(jnp.all(self._reward_state.done)):
                break

        per_step_bk = (
            jnp.concatenate(per_step_chunks, axis=-1)
            if per_step_chunks
            else jnp.zeros((bsz, 0), dtype=jnp.float32)
        )
        return {
            "per_step_reward_bk": per_step_bk,
            "episode_return_b": jnp.sum(per_step_bk, axis=-1),
            "outer_reward_b": outer_reward_b,
            "overlap_any_b": overlap_any_b,
            "offroad_any_b": offroad_any_b,
            "reached_any_b": reached_any_b,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_batched_scenarios(source: Any, indices: list[int]) -> Any:
        """Stack ``len(indices)`` unbatched scenarios from a ScenarioSource into
        a leading-batch-axis pytree suitable for :meth:`reset`.
        """
        scens = [source.get(int(i)) for i in indices]
        return jax.tree_util.tree_map(
            lambda *leaves: jnp.stack([jnp.asarray(x) for x in leaves], axis=0), *scens
        )
