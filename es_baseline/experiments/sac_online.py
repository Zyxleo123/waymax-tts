"""Online SAC initialization: roll the V-Max policy from the *current* ES state.

The offline path (``dump_sac_init_bank.py`` + ``sac_bank.py``) rolls K closed-loop
SAC episodes once per scene, from the env's own reset point (scenario step 10), and
every ES replan afterwards re-slices that same frozen bundle. That is only the
policy's answer to "what would you do from the state you *started* in". Once ES has
executed a few non-SAC candidates the ego is somewhere the offline rollouts never
visited, and the bank has to be re-anchored onto the live pose by nearest-point
search — a splice that invents a maneuver the policy never proposed.

This module removes the mismatch: at every replan step the ego's executed history is
written into a clean V-Max simulator state, and K stochastic rollouts are launched
from *that* state. Index 0 of each returned trajectory is the ego's current pose, so
the ES anchor gap is zero by construction and no re-anchoring is needed.

Two things are deliberately kept separate from the ES state:

  * The scenario is loaded fresh from the tfrecord rather than reusing the ES
    ``sim_state``. ES rewrites the ego's *log* trajectory in place with each accepted
    plan, and V-Max reads the log for the goal, the route and the expert; feeding it
    back would move the goal as the search progresses.
  * The executed ego history goes into ``sim_trajectory`` only, which is exactly
    where a real closed-loop rollout would have put it.

The object budget, observation config, termination keys and network come from the
training run's own ``.hydra/config.yaml``, so the policy sees the observation it was
trained on.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
_ES = _REPO / "es_baseline"
_VMAX = _REPO / "V-Max"
for _p in (_ES, _VMAX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from waymax import config as waymax_config  # noqa: E402
from waymax import datatypes, dynamics  # noqa: E402


# Step index of the checkpoint the online arm is meant to run: the best eval score of
# the womd_raw_parity_sac_lq run, saved as model_best.pkl. V-Max's own
# ``get_model_path`` would pick the highest-numbered file instead, which is a later,
# worse checkpoint — hence the explicit default here.
DEFAULT_MODEL_FILE = "model_best.pkl"

POSE_LAYOUT = ("x", "y", "yaw", "vel_x", "vel_y")


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _algorithm_modules(algorithm: str):
    """``(make_inference_fn, make_networks)`` for a V-Max algorithm name.

    A local copy of ``vmax.scripts.evaluate.utils.get_algorithm_modules``: importing
    that module drags in the evaluation script's plotting stack (seaborn, mediapy,
    pandas), none of which this process has or needs.
    """
    algorithm = algorithm.lower()
    if algorithm == "sac":
        from vmax.agents.learning.reinforcement.sac import sac_factory as factory
    elif algorithm == "bc":
        from vmax.agents.learning.imitation.bc import bc_factory as factory
    elif algorithm == "bc_sac":
        from vmax.agents.learning.hybrid.bc_sac import bc_sac_factory as factory
    elif algorithm == "ppo":
        from vmax.agents.learning.reinforcement.ppo import ppo_factory as factory
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    return factory.make_inference_fn, factory.make_networks


def _load_params(path: str):
    """Unpickle a V-Max checkpoint, remapping pre-rename module paths."""
    import io
    import pickle

    from etils import epath

    class _ModuleCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("vmax.learning.algorithms"):
                new_module = module.replace(
                    "vmax.learning.algorithms.rl", "vmax.agents.learning.reinforcement")
                try:
                    __import__(new_module)
                    return getattr(sys.modules[new_module], name)
                except (ImportError, AttributeError) as exc:
                    print(f"[sac_online] checkpoint module remap failed: {exc}")
            return super().find_class(module, name)

    with epath.Path(path).open("rb") as fin:
        return _ModuleCompatUnpickler(io.BytesIO(fin.read())).load()


def _load_scenarios_unbatched(tfrecord_path: str, indices, ds_cfg):
    """Load the given records as *unbatched* simulator states, one scan of the shard.

    Deliberately not ``data.scenario_loader.load_scenario_state_batch_fast``: that
    one stacks the serialized records before preprocessing, and waymax applies
    ``max_num_objects`` as ``v[:max_num_objects]`` on the leading axis — which is the
    batch axis once records are stacked. With a batch smaller than the object budget
    it silently truncates nothing, so every state comes back with the parser's default
    128 objects and the 64-object V-Max env rejects it. Preprocessing one record at a
    time puts the object axis first again, exactly like V-Max's own data generator.
    """
    import tensorflow as tf
    from waymax import dataloader
    from waymax.dataloader import womd_factories

    wanted = sorted({int(i) for i in indices})
    remaining = set(wanted)
    out = {}
    for raw_idx, serialized in enumerate(tf.data.TFRecordDataset([tfrecord_path])):
        if raw_idx not in remaining:
            continue
        processed = dataloader.preprocess_serialized_womd_data(serialized, ds_cfg)
        out[raw_idx] = womd_factories.simulator_state_from_womd_dict(
            processed, include_sdc_paths=ds_cfg.include_sdc_paths)
        remaining.discard(raw_idx)
        if not remaining:
            break
    if remaining:
        raise ValueError(
            f"scenario indices {sorted(remaining)} are out of range for {tfrecord_path}")
    return out


class OnlineSacInit:
    """K stochastic V-Max SAC rollouts launched from an arbitrary ES state."""

    def __init__(
        self,
        run_dir: str | Path,
        tfrecord_path: str | Path,
        *,
        model_path: Optional[str | Path] = None,
        k: int = 64,
        rollout_steps: int = 50,
        deterministic: bool = False,
        action_noise_std: float = 0.0,
        max_num_objects: Optional[int] = None,
        dt: float = 0.1,
    ) -> None:
        from vmax.simulator import make_env_for_evaluation, wrappers

        self.run_dir = Path(run_dir).resolve()
        self.tfrecord_path = str(tfrecord_path)
        self.k = int(k)
        self.rollout_steps = int(rollout_steps)
        self.deterministic = bool(deterministic)
        self.action_noise_std = float(action_noise_std)
        self.dt = float(dt)

        config_path = self.run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} missing — the online SAC init needs the training run's "
                f"hydra config to rebuild the observation and network it was trained with")
        cfg = yaml.safe_load(config_path.read_text())
        # Flatten exactly as vmax.scripts.evaluate.utils.setup_evaluation does: the
        # network builders read these top-level keys. `network.encoder` (not
        # `algorithm.network.encoder`) is the encoder that was actually trained.
        cfg["encoder"] = cfg["network"]["encoder"]
        cfg["policy"] = cfg["algorithm"]["network"]["policy"]
        cfg["value"] = cfg["algorithm"]["network"]["value"]
        cfg["unflatten_config"] = cfg["observation_config"]
        cfg["action_distribution"] = cfg["algorithm"]["network"]["action_distribution"]
        self.config = cfg

        self.max_num_objects = int(max_num_objects or cfg["max_num_objects"])
        self.env = make_env_for_evaluation(
            max_num_objects=self.max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=not bool(cfg["waymo_dataset"]),
            observation_type=cfg["observation_type"],
            observation_config=cfg["observation_config"],
            termination_keys=cfg["termination_keys"],
            noisy_init=False,
        )

        self._brax = self.env
        while not isinstance(self._brax, wrappers.BraxWrapper):
            self._brax = self._brax.env
        # Everything below BraxWrapper: the plain waymax env plus obs/reward wrappers.
        # Its reset is the one that fills the sim trajectory from the log.
        self._inner = self._brax.env
        self.init_steps = int(self.env.get_wrapper_attr("config").init_steps)

        self.model_path = Path(model_path) if model_path else self.run_dir / "model" / DEFAULT_MODEL_FILE
        if not self.model_path.exists():
            raise FileNotFoundError(f"SAC checkpoint not found: {self.model_path}")

        algorithm = cfg["algorithm"]["name"]
        make_inference_fn, build_network = _algorithm_modules(algorithm)
        network = build_network(
            observation_size=self.env.observation_spec(),
            action_size=self.env.action_spec().data.shape[0],
            unflatten_fn=self.env.get_wrapper_attr("features_extractor").unflatten_features,
            learning_rate=cfg["algorithm"]["learning_rate"],
            network_config=cfg,
        )
        params = _load_params(str(self.model_path))
        self.policy_fn = make_inference_fn(network)(params.policy, deterministic=self.deterministic)

        self._scenarios: Dict[int, object] = {}
        self._rollout_cache: Dict[int, callable] = {}
        self._num_timesteps: Optional[int] = None

    # ---- scenarios ---------------------------------------------------------
    def prepare(self, scenario_indices) -> None:
        """Load (and cache) clean V-Max scenarios for the given tfrecord indices."""
        missing = sorted({int(i) for i in scenario_indices} - set(self._scenarios))
        if not missing:
            return
        # WOD_1_3_1_TRAINING already carries the SDC-route layout the checkpoint was
        # trained with (include_sdc_paths, 45 paths x 800 points == num_sdc_paths /
        # num_points_per_sdc_path in the run's hydra config).
        ds_cfg = dataclasses.replace(
            waymax_config.WOD_1_3_1_TRAINING,
            path=self.tfrecord_path,
            max_num_objects=self.max_num_objects,
            batch_dims=(),
            shuffle_seed=None,
        )
        loaded = _load_scenarios_unbatched(self.tfrecord_path, missing, ds_cfg)
        for scen_idx, state in loaded.items():
            num_objects = int(state.log_trajectory.x.shape[0])
            if num_objects != self.max_num_objects:
                raise RuntimeError(
                    f"scenario {scen_idx} loaded with {num_objects} objects, env expects "
                    f"{self.max_num_objects}")
            self._scenarios[int(scen_idx)] = state
            self._num_timesteps = int(state.log_trajectory.x.shape[-1])

    def _scenario(self, scenario_index: int):
        idx = int(scenario_index)
        if idx not in self._scenarios:
            self.prepare([idx])
        return self._scenarios[idx]

    # ---- state surgery -----------------------------------------------------
    def _state_at(self, scenario, ego_history_t5: np.ndarray, timestep: int):
        """Clean scenario advanced to ``timestep`` with the ES ego history spliced in."""
        state = self._inner.reset(scenario)
        reset_step = self.init_steps - 1
        if int(timestep) < reset_step:
            raise ValueError(
                f"online SAC init cannot start before the env's own reset step "
                f"{reset_step} (got timestep={timestep})")
        n_advance = int(timestep) - reset_step
        if n_advance > 0:
            state = datatypes.update_state_by_log(state, n_advance)

        sdc = int(np.argmax(np.asarray(state.object_metadata.is_sdc).astype(np.int32)))
        upto = int(timestep) + 1
        hist = np.asarray(ego_history_t5, dtype=np.float32)[:upto]
        if hist.shape[0] < upto:
            raise ValueError(
                f"ego history has {hist.shape[0]} steps, need {upto} to reach timestep {timestep}")

        sim = state.sim_trajectory
        updates = {}
        for i, name in enumerate(POSE_LAYOUT):
            arr = np.array(getattr(sim, name), copy=True)
            col = _wrap_to_pi(hist[:, i]) if name == "yaw" else hist[:, i]
            arr[sdc, :upto] = col
            updates[name] = jnp.asarray(arr)
        return state.replace(sim_trajectory=sim.replace(**updates))

    # ---- rollout -----------------------------------------------------------
    @staticmethod
    def _ego_pose(state) -> jax.Array:
        from vmax.simulator import operations as vmax_operations

        idx = vmax_operations.get_index(state.object_metadata.is_sdc)
        traj = state.current_sim_trajectory
        return jnp.stack([
            traj.x[idx, 0], traj.y[idx, 0], traj.yaw[idx, 0],
            traj.vel_x[idx, 0], traj.vel_y[idx, 0],
        ])

    def _transition_from_state(self, state):
        """EnvTransition for a state we placed ourselves.

        Mirrors ``BraxWrapper.reset`` minus the reset itself, which would rewind the
        simulation to the env's init step and throw away the ES history.
        """
        from vmax.simulator import wrappers

        obs = self._brax.observe(state)
        shape = state.shape + self._brax.discount_spec().shape
        return wrappers.EnvTransition(
            state=state,
            observation=obs,
            reward=jnp.zeros(shape, dtype=jnp.float32),
            done=jnp.zeros(shape, dtype=jnp.bool_),
            flag=jnp.ones(shape, dtype=jnp.bool_),
            metrics=self._brax.metrics(state),
            info={
                "steps": jnp.zeros(state.shape, dtype=jnp.int32),
                "rewards": jnp.zeros(state.shape, dtype=jnp.float32),
                "truncation": jnp.zeros(state.shape, dtype=jnp.bool_),
                "scenario_id": jnp.zeros(state.shape, dtype=jnp.int32),
            },
        )

    def _get_rollout_fn(self, steps: int):
        if steps in self._rollout_cache:
            return self._rollout_cache[steps]

        from vmax.agents.pipeline.inference import _create_valid_action

        def rollout(state_k, key):
            transition = jax.vmap(self._transition_from_state)(state_k)
            pose0 = jax.vmap(self._ego_pose)(transition.state)

            def body(carry, step_key):
                tr = carry
                k_act, k_noise = jax.random.split(step_key)
                actions, _ = self.policy_fn(tr.observation, k_act)
                if self.action_noise_std > 0.0:
                    actions = jnp.clip(
                        actions + self.action_noise_std * jax.random.normal(k_noise, actions.shape),
                        -1.0, 1.0)
                tr = self.env.step(tr, _create_valid_action(actions))
                out = (
                    jax.vmap(self._ego_pose)(tr.state),
                    tr.metrics["overlap"],
                    tr.metrics["offroad"],
                    tr.done,
                )
                return tr, out

            _, ys = jax.lax.scan(body, transition, jax.random.split(key, steps))
            return pose0, ys

        fn = jax.jit(rollout)
        self._rollout_cache[steps] = fn
        return fn

    def rollout(
        self,
        *,
        scenario_index: int,
        ego_history_t5: np.ndarray,
        timestep: int,
        rng: jax.Array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(traj [K,T,5] world x,y,yaw,vx,vy, safe [K], length [K])``.

        ``traj[:, 0]`` is the ego's pose at ``timestep`` — i.e. the bank's time base is
        the current step, not the episode start.
        """
        scenario = self._scenario(scenario_index)
        state_t = self._state_at(scenario, ego_history_t5, timestep)

        num_timesteps = int(self._num_timesteps or scenario.log_trajectory.x.shape[-1])
        # Never step past the end of the logged episode: the other agents are on log
        # playback and have nothing left to play.
        steps = int(min(self.rollout_steps, max(num_timesteps - 1 - int(timestep), 0)))
        if steps == 0:
            pose = np.asarray(self._ego_pose(state_t), dtype=np.float32)
            traj = np.repeat(pose[None, None, :], self.k, axis=0)
            return traj, np.ones((self.k,), dtype=bool), np.ones((self.k,), dtype=np.int32)

        state_k = jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (self.k,) + jnp.asarray(x).shape),
            state_t,
        )
        pose0, (poses, overlap, offroad, done) = self._get_rollout_fn(steps)(state_k, rng)

        pose0 = np.asarray(pose0, dtype=np.float32)                 # [K, 5]
        poses = np.asarray(poses, dtype=np.float32)                 # [steps, K, 5]
        traj = np.concatenate([pose0[None], poses], axis=0).transpose(1, 0, 2)  # [K, steps+1, 5]

        done = np.asarray(done).reshape(steps, self.k).astype(bool)
        overlap = np.asarray(overlap).reshape(steps, self.k) > 0.5
        offroad = np.asarray(offroad).reshape(steps, self.k) > 0.5

        # A step is "recorded" up to and including the one that ended the episode; the
        # fixed-length scan keeps running afterwards, and those states are meaningless.
        ended_before = np.concatenate(
            [np.zeros((1, self.k), dtype=bool), np.cumsum(done, axis=0)[:-1] > 0], axis=0)
        recorded = ~ended_before                                    # [steps, K]
        length = 1 + recorded.sum(axis=0).astype(np.int32)          # + the pose at `timestep`
        unsafe = ((overlap | offroad) & recorded).any(axis=0)
        return traj, ~unsafe, length
