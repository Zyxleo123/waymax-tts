"""SB3 callbacks for Waymax SAC: episode-quality metrics + periodic eval."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class WaymaxEpisodeStatsWrapper(gym.Wrapper):
    """Accumulate per-episode collision / offroad / route deviation for logging."""

    def reset(self, *, seed=None, options=None):
        self._ep_collision = False
        self._ep_offroad = False
        self._ep_reached = False
        self._ep_max_lateral_m = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_collision |= bool(info.get("collision", False))
        self._ep_offroad |= bool(info.get("offroad", False))
        self._ep_reached |= bool(info.get("reached", False))
        self._ep_max_lateral_m = max(
            self._ep_max_lateral_m, float(info.get("lateral_deviation_m", 0.0))
        )
        if terminated or truncated:
            info = dict(info)
            info["episode_collision"] = self._ep_collision
            info["episode_offroad"] = self._ep_offroad
            info["episode_reached"] = self._ep_reached
            info["episode_clean"] = (
                self._ep_reached and not self._ep_collision and not self._ep_offroad
            )
            info["episode_max_lateral_m"] = self._ep_max_lateral_m
        return obs, reward, terminated, truncated, info


_EPISODE_INFO_KEYS = (
    "episode_collision",
    "episode_offroad",
    "episode_reached",
    "episode_clean",
    "episode_max_lateral_m",
)


class WaymaxEpisodeMetricsCallback(BaseCallback):
    """Log rolling train collision / offroad / clean-success from collection rollouts."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = int(window)
        self._buf: dict[str, deque] = {k: deque(maxlen=self.window) for k in _EPISODE_INFO_KEYS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not any(k in info for k in _EPISODE_INFO_KEYS):
                continue
            for key in _EPISODE_INFO_KEYS:
                if key in info:
                    val = info[key]
                    self._buf[key].append(float(val) if key == "episode_max_lateral_m" else int(val))

        if not self._buf["episode_clean"]:
            return True

        n = len(self._buf["episode_clean"])
        self.logger.record("train/collision_rate", np.mean(self._buf["episode_collision"]))
        self.logger.record("train/offroad_rate", np.mean(self._buf["episode_offroad"]))
        self.logger.record("train/goal_reached_rate", np.mean(self._buf["episode_reached"]))
        self.logger.record("train/clean_success_rate", np.mean(self._buf["episode_clean"]))
        self.logger.record("train/mean_max_lateral_m", np.mean(self._buf["episode_max_lateral_m"]))
        self.logger.record("train/metrics_episodes", n)
        return True


def evaluate_waymax_policy(
    model,
    env: gym.Env,
    *,
    n_episodes: int,
    deterministic: bool = True,
) -> dict[str, float]:
    """Roll the policy over ``n_episodes`` sequential resets; return aggregate rates."""
    n_collision = n_offroad = n_reached = n_clean = 0
    lateral_vals: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_collision = ep_offroad = ep_reached = False
        ep_max_lateral = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            ep_collision |= bool(info.get("collision", False))
            ep_offroad |= bool(info.get("offroad", False))
            ep_reached |= bool(info.get("reached", False))
            ep_max_lateral = max(
                ep_max_lateral, float(info.get("lateral_deviation_m", 0.0))
            )
            done = terminated or truncated

        n_collision += int(ep_collision)
        n_offroad += int(ep_offroad)
        n_reached += int(ep_reached)
        n_clean += int(ep_reached and not ep_collision and not ep_offroad)
        lateral_vals.append(ep_max_lateral)

    n = max(n_episodes, 1)
    return _episode_rate_metrics(
        n_episodes=n_episodes,
        n_collision=n_collision,
        n_offroad=n_offroad,
        n_reached=n_reached,
        n_clean=n_clean,
        lateral_vals=lateral_vals,
        prefix="eval",
    )


def _episode_rate_metrics(
    *,
    n_episodes: int,
    n_collision: int,
    n_offroad: int,
    n_reached: int,
    n_clean: int,
    lateral_vals: list[float],
    prefix: str,
) -> dict[str, float]:
    n = max(n_episodes, 1)
    return {
        f"{prefix}/episodes": float(n_episodes),
        f"{prefix}/collision_rate": n_collision / n,
        f"{prefix}/offroad_rate": n_offroad / n,
        f"{prefix}/goal_reached_rate": n_reached / n,
        f"{prefix}/clean_success_rate": n_clean / n,
        f"{prefix}/mean_max_lateral_m": float(np.mean(lateral_vals)) if lateral_vals else 0.0,
    }


def evaluate_expert_replay(
    env: gym.Env,
    *,
    n_episodes: int,
    prefix: str = "eval",
) -> dict[str, float]:
    """Closed-loop bicycle expert replay in ``env`` (upper bound for BC sanity)."""
    from rl.bc_core import expert_action_sdc

    n_collision = n_offroad = n_reached = n_clean = 0
    lateral_vals: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_collision = ep_offroad = ep_reached = False
        ep_max_lateral = 0.0
        while not done:
            action = expert_action_sdc(env._state, env._dynamics)
            obs, _, terminated, truncated, info = env.step(
                action[:env._action_dim]
            )
            ep_collision |= bool(info.get("collision", False))
            ep_offroad |= bool(info.get("offroad", False))
            ep_reached |= bool(info.get("reached", False))
            ep_max_lateral = max(
                ep_max_lateral, float(info.get("lateral_deviation_m", 0.0))
            )
            done = terminated or truncated

        n_collision += int(ep_collision)
        n_offroad += int(ep_offroad)
        n_reached += int(ep_reached)
        n_clean += int(ep_reached and not ep_collision and not ep_offroad)
        lateral_vals.append(ep_max_lateral)

    return _episode_rate_metrics(
        n_episodes=n_episodes,
        n_collision=n_collision,
        n_offroad=n_offroad,
        n_reached=n_reached,
        n_clean=n_clean,
        lateral_vals=lateral_vals,
        prefix=prefix,
    )


class WaymaxPeriodicEvalCallback(BaseCallback):
    """Run a full sequential eval every ``eval_freq`` env steps.

    Optionally supports **early stopping** (used by the overfitting driver): when
    ``stop_threshold`` is set and ``eval/<stop_metric>`` stays at or above it for
    ``stop_patience`` consecutive evals, ``_on_step`` returns ``False`` so
    ``model.learn`` halts. ``solved`` / ``solved_at_timestep`` record the outcome;
    ``last_metrics`` holds the most recent eval dict either way. Defaults leave the
    original always-continue behavior unchanged.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        n_episodes: int,
        *,
        deterministic: bool = True,
        verbose: int = 1,
        stop_threshold: float | None = None,
        stop_patience: int = 1,
        stop_metric: str = "eval/clean_success_rate",
        log_fn: Callable[[dict[str, float], int], None] | None = None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_episodes = int(n_episodes)
        self.deterministic = deterministic
        self._last_eval_step = 0
        # Early-stop state.
        self.stop_threshold = None if stop_threshold is None else float(stop_threshold)
        self.stop_patience = max(1, int(stop_patience))
        self.stop_metric = str(stop_metric)
        self._consec_pass = 0
        self.last_metrics: dict[str, float] | None = None
        self.last_eval_timestep: int | None = None
        self.solved = False
        self.solved_at_timestep: int | None = None
        # Optional sink for metrics (e.g. a shared wandb run). Called as
        # ``log_fn(metrics, num_timesteps)`` after each eval. Defaults to None
        # so existing callers are unaffected.
        self.log_fn = log_fn

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        metrics = evaluate_waymax_policy(
            self.model,
            self.eval_env,
            n_episodes=self.n_episodes,
            deterministic=self.deterministic,
        )
        if hasattr(self.model, "_actor_frozen"):
            metrics["eval/actor_frozen"] = float(self.model._actor_frozen)
        actor = getattr(self.model, "actor", None)
        if actor is not None:
            weight_abs = sum(float(p.detach().abs().sum()) for p in actor.parameters())
            metrics["eval/actor_weight_abs_sum"] = weight_abs
        for key, val in metrics.items():
            self.logger.record(key, val)

        self.last_metrics = metrics
        self.last_eval_timestep = int(self.num_timesteps)
        if self.log_fn is not None:
            self.log_fn(metrics, int(self.num_timesteps))

        if self.verbose:
            print(
                f"[eval step {self.num_timesteps}] "
                f"clean={metrics['eval/clean_success_rate']:.3f} "
                f"reach={metrics['eval/goal_reached_rate']:.3f} "
                f"col={metrics['eval/collision_rate']:.3f} "
                f"off={metrics['eval/offroad_rate']:.3f} "
                f"lat={metrics['eval/mean_max_lateral_m']:.1f}m"
            )

        if self.stop_threshold is not None:
            val = float(metrics.get(self.stop_metric, 0.0))
            self._consec_pass = self._consec_pass + 1 if val >= self.stop_threshold else 0
            if self._consec_pass >= self.stop_patience:
                self.solved = True
                self.solved_at_timestep = int(self.num_timesteps)
                if self.verbose:
                    print(
                        f"[overfit] SOLVED: {self.stop_metric}={val:.3f} "
                        f">= {self.stop_threshold:.3f} for {self._consec_pass} eval(s) "
                        f"at step {self.num_timesteps}; stopping."
                    )
                return False
        return True


class WandbSB3MetricsCallback(BaseCallback):
    """Mirror SB3 logger scalars (actor/critic loss, entropy, rollout reward, …)
    into an already-open wandb run.

    Uses an explicit ``global_step = offset + num_timesteps`` so multi-unit
    overfit jobs do not collide on the wandb step axis (unlike
    ``sync_tensorboard=True``, which resets every unit).
    """

    def __init__(
        self,
        *,
        step_offset: int = 0,
        log_freq: int = 500,
        key_prefixes: tuple[str, ...] = ("train/", "rollout/", "time/"),
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.step_offset = int(step_offset)
        self.log_freq = max(1, int(log_freq))
        self.key_prefixes = key_prefixes
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step < self.log_freq:
            return True
        self._last_log_step = int(self.num_timesteps)
        self._flush()
        return True

    def _on_training_end(self) -> None:
        # Final dump so the last few gradient steps are not lost.
        self._flush()

    def _flush(self) -> None:
        import wandb

        if wandb.run is None:
            return
        payload: dict[str, Any] = {
            "global_step": self.step_offset + int(self.num_timesteps),
        }
        values = getattr(self.logger, "name_to_value", None) or {}
        for key, val in values.items():
            if not any(key.startswith(p) for p in self.key_prefixes):
                continue
            try:
                payload[key] = float(val)
            except (TypeError, ValueError):
                continue

        # rollout/* is written then cleared inside SB3 dump_logs; pull the same
        # numbers from the episode buffer so reward/length still show up.
        ep_info = getattr(self.model, "ep_info_buffer", None)
        if ep_info:
            rews = [float(x["r"]) for x in ep_info if "r" in x]
            lens = [float(x["l"]) for x in ep_info if "l" in x]
            if rews:
                payload["rollout/ep_rew_mean"] = float(np.mean(rews))
            if lens:
                payload["rollout/ep_len_mean"] = float(np.mean(lens))

        if len(payload) > 1:
            wandb.log(payload)


def make_monitored_env(env_fn):
    """Build env with episode stats wrapper + Monitor (for SB3 vec env)."""
    from stable_baselines3.common.monitor import Monitor

    def _thunk():
        env = WaymaxEpisodeStatsWrapper(env_fn())
        return Monitor(env, info_keywords=_EPISODE_INFO_KEYS)

    return _thunk
