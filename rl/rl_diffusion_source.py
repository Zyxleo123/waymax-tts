"""Bridge: turn RL-policy rollouts into training data for the diffusion model.

The diffusion model (``train/train_diffusion.py``) reads its ego-trajectory
*target* from ``SimulatorState.log_trajectory`` via
``data.preprocess.preprocess_simulator_state``. By default that target is the
human WOMD log. This module rolls out a trained RL policy (or a random policy,
for smoke tests) in :class:`~rl.waymax_env.WaymaxGymEnv`, splices the resulting
ego trajectory back into ``log_trajectory`` (see
:func:`rl.waymax_env._splice_sdc_sim_into_log`), and yields batched
``SimulatorState`` objects in exactly the same format Waymax's
``dataloader.simulator_state_generator`` produces.

That means the diffusion training loop can consume RL trajectories with no other
changes: it will learn to imitate the policy instead of the human log.

Usage
-----
As a generator (e.g. inside a training loop)::

    from rl.scenario_source import ScenarioSource
    from rl.rl_diffusion_source import RLDiffusionSource

    source = ScenarioSource.from_tfrecord(path, indices, max_num_objects=32)
    rl_source = RLDiffusionSource(source, model_path="runs/ppo_waymax/ppo_waymax.zip")
    for sim_state in rl_source.generator(batch_size=8):
        ...  # feed sim_state to preprocess_simulator_state / the diffusion model

Smoke test (rollout -> splice -> preprocess), no trained model required::

    python -m rl.rl_diffusion_source \
        --tfrecord /path/to/training_tfexample.tfrecord-00000-of-01000 \
        --num-scenarios 4 --batch-size 4 --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv


class RLDiffusionSource:
    """Rolls out an RL policy and yields diffusion-ready ``SimulatorState``s.

    Each yielded state is a batched (prefix shape ``[B]``) Waymax
    ``SimulatorState`` whose SDC ``log_trajectory`` has been replaced by the
    policy's rollout, suitable for ``preprocess_simulator_state``.
    """

    def __init__(
        self,
        source: ScenarioSource,
        *,
        model_path: str | None = None,
        model: Any | None = None,
        deterministic: bool = True,
        max_episode_steps: int = 80,
        goal_threshold_m: float = 3.0,
        action_space_type: str = "bicycle",
        delta_max_dx: float = 6.0,
        delta_max_dy: float = 6.0,
        delta_max_dyaw: float = float(np.pi),
        device: str = "cpu",
        seed: int = 0,
    ):
        self._deterministic = bool(deterministic)
        self._env = WaymaxGymEnv(
            source,
            reward_config=RewardConfig(goal_threshold_m=goal_threshold_m),
            max_episode_steps=max_episode_steps,
            sequential=True,
            seed=seed,
            action_space_type=action_space_type,
            delta_max_dx=delta_max_dx,
            delta_max_dy=delta_max_dy,
            delta_max_dyaw=delta_max_dyaw,
        )

        self._model = model
        if self._model is None and model_path is not None:
            from stable_baselines3 import PPO

            self._model = PPO.load(model_path, device=device)

    # ------------------------------------------------------------------ #
    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        if self._model is None:
            # No policy provided: random actions (smoke tests / sanity checks).
            return self._env.action_space.sample()
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        return action

    def rollout_one(self) -> Any:
        """Rolls out a full episode and returns the spliced (unbatched) state."""
        obs, _ = self._env.reset()
        done = False
        while not done:
            action = self._select_action(obs)
            obs, _reward, terminated, truncated, _info = self._env.step(action)
            done = terminated or truncated
        return self._env.simulated_log_state()

    def generate_batch(self, batch_size: int) -> Any:
        """Rolls out ``batch_size`` episodes and stacks them into a [B, ...] state."""
        states = [self.rollout_one() for _ in range(int(batch_size))]
        # All states share structure/shapes (same num_objects and horizon), so a
        # leaf-wise stack reproduces the batched layout of the Waymax dataloader.
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *states)

    def generator(self, batch_size: int) -> Iterator[Any]:
        """Infinite generator of batched ``SimulatorState``s (cycles scenarios)."""
        while True:
            yield self.generate_batch(batch_size)


def build_source(args: argparse.Namespace) -> ScenarioSource:
    if args.failure_dir:
        return ScenarioSource.from_failure_dir(
            args.failure_dir,
            max_num_objects=args.max_num_objects,
            failures_only=not args.include_successes,
            limit=args.limit,
        )
    if args.tfrecord:
        if args.indices:
            indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        elif args.num_scenarios:
            indices = list(range(int(args.num_scenarios)))
        else:
            raise ValueError("Provide --indices or --num-scenarios with --tfrecord.")
        return ScenarioSource.from_tfrecord(
            args.tfrecord, indices, max_num_objects=args.max_num_objects
        )
    raise ValueError("Provide either --failure-dir or --tfrecord.")


def _run_smoke(args: argparse.Namespace) -> None:
    """Rollout -> splice -> preprocess, asserting the diffusion target is sane."""
    import jax.numpy as jnp  # local alias for clarity

    from data.preprocess import preprocess_simulator_state
    from data.types import PreprocessConfig

    source = build_source(args)
    rl_source = RLDiffusionSource(
        source,
        model_path=args.model,
        max_episode_steps=args.max_episode_steps,
        goal_threshold_m=args.goal_threshold_m,
        action_space_type=args.action_space,
        device=args.device,
        seed=args.seed,
    )

    policy_desc = f"PPO model {args.model}" if args.model else "random policy"
    print(f"[rl_diffusion_source] Rolling out {args.batch_size} scenarios with {policy_desc} ...")
    sim_state = rl_source.generate_batch(args.batch_size)
    print(f"[rl_diffusion_source] Batched SimulatorState: "
          f"log_trajectory.x shape = {tuple(np.asarray(sim_state.log_trajectory.x).shape)}")

    cfg = PreprocessConfig(
        model_dt=args.model_dt,
        predict_horizon=args.predict_horizon,
        ego_range=args.ego_range,
        max_velocity=args.max_velocity,
    )
    rng = jax.random.PRNGKey(args.seed)
    pre_batch, _ = preprocess_simulator_state(sim_state, rng, cfg)
    ego_traj = pre_batch.features["ego_trajectory"]  # [B, H, 5] (normalized)

    finite = bool(jnp.all(jnp.isfinite(ego_traj)))
    print(f"[rl_diffusion_source] Diffusion target 'ego_trajectory' shape = {tuple(ego_traj.shape)}")
    print(f"[rl_diffusion_source] target finite = {finite} | "
          f"min={float(jnp.min(ego_traj)):.4f} max={float(jnp.max(ego_traj)):.4f}")
    if not finite:
        raise RuntimeError("Preprocessed RL trajectory contains non-finite values.")

    print("[rl_diffusion_source] OK: RL rollouts flow through diffusion preprocessing.")


def main() -> None:
    args = _parse_args()
    _run_smoke(args)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce diffusion training states from RL rollouts.")
    # Data source.
    p.add_argument("--failure-dir", type=str, default=None)
    p.add_argument("--tfrecord", type=str, default=None)
    p.add_argument("--indices", type=str, default=None)
    p.add_argument("--num-scenarios", type=int, default=None)
    p.add_argument("--include-successes", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-num-objects", type=int, default=32)

    # Policy / rollout.
    p.add_argument("--model", type=str, default=None,
                   help="Path to a PPO .zip checkpoint. If omitted, a random policy is used.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--action-space", type=str, default="bicycle", choices=["bicycle", "delta"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # Preprocess config (must match diffusion training).
    p.add_argument("--model-dt", type=float, default=0.2)
    p.add_argument("--predict-horizon", type=int, default=25)
    p.add_argument("--ego-range", type=float, default=100.0)
    p.add_argument("--max-velocity", type=float, default=25.0)

    p.add_argument("--smoke", action="store_true",
                   help="Run the rollout->preprocess sanity check and exit.")
    return p.parse_args()


if __name__ == "__main__":
    main()
