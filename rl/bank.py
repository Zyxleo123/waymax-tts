"""Bank clean rollouts per scenario, for BC-diffusion distillation.

Why this exists
---------------
``overfit_failures.py`` stops a unit when ``eval/clean_success_rate`` clears a
threshold -- an *aggregate* rate at a *single checkpoint*. With N scenarios and a
0.95 threshold that demands all N be clean simultaneously (``p**N``). But the
artifact we actually want is a clean *trajectory per scenario*, and a scenario
only has to succeed **once, at any point in training** -- a union over time, not
an intersection at one checkpoint.

So: evaluate periodically, and the first time a scenario comes out clean, write
its trajectory to disk and mark it banked. SAC oscillating between scenarios
(solve A, drift, solve B while losing A) then stops being a failure mode and
becomes the harvest mechanism.

Banked scenarios can be dropped from training entirely -- see ``remaining()``.
The trajectory on disk is the product; the policy is only the means, so letting
the policy forget a banked scenario costs nothing and frees the whole budget for
the unsolved ones.

Identity
--------
``make_env_for_evaluation`` has **no AutoResetWrapper** (unlike the training env),
so a slot holds its scenario for the entire rollout and slot index is a stable
identity. We read the dataset with ``shuffle_seed=None`` so slot -> record index
is just file order. (The trainer's own failures eval uses ``seed=69`` and 256
slots for 218 records, so it shuffles *and* resamples -- never use it for identity.)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np


class TrajectoryBank:
    """Tracks which scenarios have yielded a clean rollout, and stores them.

    The manifest is rewritten atomically after every update so an interrupted run
    (or a crash mid-training) never loses banked scenarios.
    """

    MANIFEST = "bank_manifest.json"

    def __init__(self, out_dir: str, num_scenarios: int) -> None:
        self.out_dir = out_dir
        self.num_scenarios = int(num_scenarios)
        self.traj_dir = os.path.join(out_dir, "trajectories")
        os.makedirs(self.traj_dir, exist_ok=True)
        self._banked: dict[int, dict[str, Any]] = {}
        self._last_eval: dict[str, Any] | None = None
        self._load()

    # -- persistence ------------------------------------------------------

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.out_dir, self.MANIFEST)

    def _load(self) -> None:
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path) as f:
            payload = json.load(f)
        self._banked = {int(k): v for k, v in payload.get("banked", {}).items()}
        print(f"[bank] resumed: {len(self._banked)}/{self.num_scenarios} already banked")

    def _save(self) -> None:
        payload = {
            "num_scenarios": self.num_scenarios,
            "num_banked": len(self._banked),
            "banked": {str(k): v for k, v in sorted(self._banked.items())},
            "remaining": self.remaining(),
            "last_eval": self._last_eval,
        }
        tmp = f"{self.manifest_path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.manifest_path)

    # -- state ------------------------------------------------------------

    def is_banked(self, idx: int) -> bool:
        return int(idx) in self._banked

    def remaining(self) -> list[int]:
        """Scenario indices with no clean rollout yet -- what training should sample."""
        return [i for i in range(self.num_scenarios) if i not in self._banked]

    @property
    def num_banked(self) -> int:
        return len(self._banked)

    # -- banking ----------------------------------------------------------

    def update(
        self,
        clean_mask: np.ndarray,
        trajectories: dict[str, np.ndarray],
        step: int,
        flags: dict[str, np.ndarray] | None = None,
        quality: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Bank every newly-clean scenario, and upgrade banked ones that got better.

        "First clean wins" is only right when every clean rollout is equally good. It is
        not: a scenario banked at the first eval keeps that rollout forever, so with the
        smoothness gate disabled the bank freezes the *pretrained* policy's chatter and no
        amount of further training can replace it (observed 2026-07-16: 151 of 218 banked
        at step 5120, 65% of them above the yaw-rate limit, permanently). So a clean
        rollout that is strictly better than the stored one replaces it.

        `frac_banked` stays monotone -- an upgrade never un-banks a scenario -- so it is
        still the progress metric. Quality is persisted per entry precisely so a resumed
        run can compare against what is already on disk; an entry written before quality
        existed compares as +inf, i.e. the next clean rollout upgrades it.

        Args:
            clean_mask: bool array [num_scenarios]; True where the rollout was clean.
            trajectories: dict of arrays shaped [num_scenarios, T, ...] to store.
            step: env step this rollout came from (provenance).
            flags: per-episode {reached_goal, offroad, overlap} arrays [num_scenarios].
                Recorded per scenario so the *unbanked* ones can be triaged -- knowing
                a scenario is unsolved is not actionable, knowing it is unsolved
                because it never reaches the goal (vs. reaches it but clips a road
                edge) is.
            quality: [num_scenarios] lower-is-better score (max yaw rate, rad/s). None
                keeps the old first-clean-wins behaviour.
        """
        clean_mask = np.asarray(clean_mask).astype(bool)
        if clean_mask.shape[0] != self.num_scenarios:
            raise ValueError(
                f"clean_mask has {clean_mask.shape[0]} entries but bank tracks "
                f"{self.num_scenarios} scenarios."
            )
        if quality is not None:
            quality = np.asarray(quality, dtype=np.float64)
            if quality.shape[0] != self.num_scenarios:
                raise ValueError(
                    f"quality has {quality.shape[0]} entries but bank tracks "
                    f"{self.num_scenarios} scenarios."
                )

        newly: list[int] = []
        upgraded: list[int] = []
        for i in np.flatnonzero(clean_mask):
            i = int(i)
            prev = self._banked.get(i)
            if prev is None:
                newly.append(i)
            elif quality is None:
                continue  # already banked, no quality to compare -> first clean wins
            else:
                prev_q = prev.get("quality")
                prev_q = math.inf if prev_q is None else float(prev_q)
                if quality[i] >= prev_q - QUALITY_EPS:
                    continue  # not a meaningful improvement; keep what we have
                upgraded.append(i)

            path = os.path.join(self.traj_dir, f"scenario_{i:05d}.npz")
            np.savez_compressed(path, **{k: np.asarray(v[i]) for k, v in trajectories.items()})
            entry = {"step": int(step), "path": path}
            if quality is not None:
                entry["quality"] = float(quality[i])
            self._banked[i] = entry

        metrics = {
            "bank/num_banked": float(self.num_banked),
            "bank/frac_banked": self.num_banked / max(self.num_scenarios, 1),
            "bank/newly_banked": float(len(newly)),
            # Rollouts that replaced a worse stored one. Expected to be >0 while the
            # policy is still smoothing out, and to fall to 0 once it has converged.
            "bank/upgraded": float(len(upgraded)),
            # Current clean rate is NOT the progress metric -- it can go down.
            # frac_banked is monotone and is what you actually care about.
            "bank/current_clean_rate": float(clean_mask.mean()),
        }

        if flags is not None:
            self._last_eval = {
                "step": int(step),
                "per_scenario": {
                    str(i): {k: float(v[i]) for k, v in flags.items()}
                    for i in range(self.num_scenarios)
                },
            }
            reached = np.asarray(flags["reached_goal"]) > 0.5
            offroad = np.asarray(flags["offroad"]) > 0.5
            overlap = np.asarray(flags["overlap"]) > 0.5
            if "max_yaw_rate" in flags:
                yr = np.asarray(flags["max_yaw_rate"])
                solved = reached & ~offroad & ~overlap
                # Quality of what is actually ON DISK, not of the latest rollout. This is
                # the number that has to come down for the harvest to be worth distilling;
                # bank/max_yaw_rate_p50 can improve while the stored set stays bad.
                stored_q = [
                    e["quality"] for e in self._banked.values() if e.get("quality") is not None
                ]
                if stored_q:
                    metrics["bank/stored_max_yaw_rate_p50"] = float(np.median(stored_q))
                    metrics["bank/stored_frac_over_limit"] = float(
                        np.mean(np.asarray(stored_q) > MAX_YAW_RATE_RAD_S)
                    )
                metrics.update({
                    "bank/max_yaw_rate_p50": float(np.median(yr)),
                    # Scenarios the policy solved but we refused to bank: the price of the
                    # smoothness gate. If this stays high the policy is chattering, not
                    # the gate misfiring -- turn on reward_config.comfort rather than
                    # loosening the threshold.
                    "bank/gated_out_rate": float((solved & (yr > MAX_YAW_RATE_RAD_S)).mean()),
                })
            metrics.update({
                "bank/reached_rate": float(reached.mean()),
                "bank/offroad_rate": float(offroad.mean()),
                "bank/overlap_rate": float(overlap.mean()),
                # Why the not-yet-clean ones are not clean. These partition the
                # failures by first cause and are what to aim the next run at.
                "bank/fail_no_goal": float((~reached).mean()),
                "bank/fail_offroad_only": float((reached & offroad & ~overlap).mean()),
                "bank/fail_overlap_only": float((reached & overlap & ~offroad).mean()),
                "bank/fail_both": float((reached & offroad & overlap).mean()),
            })

        # Save whenever there is anything new to record: banked scenarios, or a
        # fresh set of flags for the unbanked.
        if newly or upgraded or flags is not None:
            self._save()
        if newly:
            print(f"[bank] +{len(newly)} banked at step {step} "
                  f"({self.num_banked}/{self.num_scenarios}); new: {newly[:10]}")
        if upgraded:
            print(f"[bank] ^{len(upgraded)} upgraded at step {step} "
                  f"(smoother rollout replaced the stored one): {upgraded[:10]}")

        return metrics


def episode_flags(
    per_step: dict[str, np.ndarray],
    done: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce per-step metrics to per-episode flags, ignoring anything after `done`.

    The evaluation env has no auto-reset, so a finished scenario keeps getting
    stepped. Those trailing steps are meaningless and would otherwise let a
    scenario that already succeeded pick up a spurious offroad. Mask to the
    prefix up to and including the first `done`.

    Args:
        per_step: {name: [num_scenarios, T]} metric values.
        done: [num_scenarios, T] bool/0-1.

    Returns:
        {name: [num_scenarios]} max over the live prefix.
    """
    done = np.asarray(done).astype(bool)
    n, T = done.shape
    # First True per row, or T-1 if it never finishes.
    first_done = np.where(done.any(axis=1), done.argmax(axis=1), T - 1)
    live = np.arange(T)[None, :] <= first_done[:, None]

    out = {}
    for name, vals in per_step.items():
        v = np.asarray(vals)
        out[name] = np.where(live, v, -np.inf).max(axis=1)
    return out


def clean_mask_from(flags: dict[str, np.ndarray], goal_key: str = "reached_goal") -> np.ndarray:
    """CLEAN == reached the goal AND never went offroad AND never collided.

    Deliberately the same definition the eval reports, so a banked trajectory is
    exactly what the downstream metric calls a success. Note this does NOT include
    off_route/overspeed/red_light: they are not part of clean, and requiring them
    would only shrink the harvest.

    Verified 2026-07-16 against the raw failure labels this whole set was cut from
    (``failure_samples/*.json``): ``success == goal_reached & ~offroad & ~overlap`` on
    all 245 records, with ``goal_reached`` = min-over-episode distance < 2 m. Identical.
    Do not add BC-quality conditions here -- they belong in the gates below, so the
    bank's notion of "success" keeps matching the data it is trying to fix.
    """
    reached = flags[goal_key] > 0.5
    offroad = flags["offroad"] > 0.5
    overlap = flags["overlap"] > 0.5
    return reached & ~offroad & ~overlap


# ---------------------------------------------------------------------------
# BC-quality gates: truncation + smoothness
#
# These are deliberately NOT part of `clean_mask_from`. Clean answers "did the
# policy solve the scene the way the failure labels define solving it". These
# answer "is the resulting trajectory a sane thing to distill into the planner",
# which is a different question with a different owner (the diffusion trainer).
# ---------------------------------------------------------------------------

# V-Max's own comfort limit (`metrics/comfort.py`: value_yaw_rate penalises above this).
#
# Calibrated 2026-07-16 against the expert log on the same 218 scenes, under this exact
# statistic (raw finite difference, max over the prefix): the human never exceeds
# **1.09 rad/s** (p50 0.35, p90 0.59), so 0.95 is a real limit and only 2/218 logs touch
# it. The policy's *median peak* is 3.07 rad/s -- 186/202 banked rollouts contain at
# least one frame outside the expert's entire observed range.
#
# So this threshold is honest but, today, unaffordable as a harvest filter: it refuses
# ~92% of the bank. The intended order is to fix generation first (turn on
# `reward_config.comfort`, stop `alpha` collapsing) and let the gate become cheap --
# not to loosen the gate until the harvest fits through it. Tune via
# `Banker(max_yaw_rate_rad_s=...)` while that is in flight.
MAX_YAW_RATE_RAD_S = 0.95
SIM_DT_S = 0.1

# A banked trajectory is only replaced when the new one is better by more than this
# (rad/s). Without a deadband, run-to-run noise would rewrite the bank every eval and the
# stored trajectory would be whichever rollout happened to be last, not the best.
QUALITY_EPS = 0.01


def goal_cut_index(
    distance_to_goal: np.ndarray,
    live: np.ndarray,
    start_t: int = 0,
) -> np.ndarray:
    """Per scenario, the step of closest approach to the goal within the live prefix.

    This is where a banked trajectory should end. The policy reaches the goal and then
    keeps driving (median 33 m past it, measured over 202 banked rollouts) because
    ``reached_goal`` is "reached at least once" and the expert's own log simply *stops*
    at the goal rather than decelerating into it -- so cutting at closest approach is
    what reproduces the expert's structure.

    Args:
        distance_to_goal: [N, T] per-step distance in meters.
        live: [N, T] bool; steps before and including the first `done`.
        start_t: steps before this are the logged history, never a valid cut.

    Returns:
        [N] int cut index (inclusive).
    """
    d = np.asarray(distance_to_goal, dtype=np.float64)
    live = np.asarray(live, dtype=bool).copy()
    if start_t:
        live[:, :start_t] = False
    # +inf outside the live prefix so argmin can never select it.
    masked = np.where(live, d, np.inf)
    cut = np.argmin(masked, axis=1)
    # A row with no live step at all would argmin to 0; keep it in bounds and let the
    # clean mask reject it (it cannot have reached the goal).
    return cut.astype(np.int32)


def truncate_at(
    trajectory: dict[str, np.ndarray],
    cut: np.ndarray,
    valid_key: str = "valid",
) -> dict[str, np.ndarray]:
    """Mark every step after `cut` invalid, leaving the arrays full length.

    **Truncate by validity, never by shortening the arrays.** Every splice path in this
    repo (``simulation/planning_utils.py:apply_ego_replacements_to_expanded_state``,
    ``viz/render.py:_apply_ego_override_to_state_batch``) writes only the slice it is
    given and forces ``valid=True`` on it -- frames past the slice keep the *original*
    log and stay valid. A shortened array therefore teleports the ego from its endpoint
    back onto the human log mid-episode: measured median 16.7 m in one 0.1 s frame
    (max 35 m) against a normal 1.35 m step. The diffusion preprocessing
    (``data/preprocess.py:sample_future_goal_step``) masks on the ego valid flags, so
    clearing them is exactly what drops the post-goal tail from both the target and the
    sampled goals.

    Args:
        trajectory: {name: [N, T, ...]} arrays; must contain `valid_key`.
        cut: [N] inclusive last valid step.

    Returns:
        A new dict with `valid_key` cleared after `cut`. Other arrays are untouched --
        their values past the cut are simply never read once invalid.
    """
    out = dict(trajectory)
    valid = np.asarray(out[valid_key])
    steps = np.arange(valid.shape[1])[None, :]
    out[valid_key] = valid & (steps <= np.asarray(cut)[:, None])
    return out


def max_yaw_rate(yaw: np.ndarray, cut: np.ndarray, start_t: int = 0, dt: float = SIM_DT_S) -> np.ndarray:
    """[N] peak |yaw rate| (rad/s) over each scenario's kept prefix.

    Raw finite difference on the stored yaw, wrapped to (-pi, pi]. Deliberately not
    smoothed: the failure mode is a *period-2 limit cycle* (yaw flipping +-25 deg every
    frame), and any smoothing window >= 2 frames averages the very thing we are trying
    to catch back out. V-Max's ComfortMetric savgol-filters before differentiating,
    which is why comfort-as-a-reward and this gate are complementary rather than
    redundant.
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    d = np.diff(yaw, axis=1)
    d = np.arctan2(np.sin(d), np.cos(d)) / dt  # wrap, then rad/s
    steps = np.arange(d.shape[1])[None, :]
    # Difference i lives between step i and i+1; keep those fully inside [start_t, cut].
    keep = (steps >= start_t) & (steps < np.asarray(cut)[:, None])
    return np.where(keep, np.abs(d), 0.0).max(axis=1)


def smooth_mask_from(
    trajectory: dict[str, np.ndarray],
    cut: np.ndarray,
    start_t: int = 0,
    max_yaw_rate_rad_s: float = MAX_YAW_RATE_RAD_S,
) -> np.ndarray:
    """[N] True where the kept prefix is kinematically sane enough to distill."""
    return max_yaw_rate(trajectory["yaw"], cut, start_t=start_t) <= float(max_yaw_rate_rad_s)


# ---------------------------------------------------------------------------
# Rollout integration
# ---------------------------------------------------------------------------

# Metrics we reduce per episode. Everything else in env_transition.metrics is
# ignored -- we only need enough to decide "clean" and to log the near-misses.
_BANK_METRIC_KEYS = ("reached_goal", "offroad", "overlap")

# Per-step, kept unreduced: the cut needs argmin over time, not a max.
_BANK_SERIES_KEYS = ("distance_to_goal",)


def make_bank_scenarios(
    path: str,
    max_num_objects: int,
    include_sdc_paths: bool,
    num_records: int,
    chunk_size: int | None = None,
):
    """Deterministic, file-order pass over ``path``.

    ``shuffle_seed=None`` is the whole point: slot i of chunk k is record
    ``(k*chunk + i) % num_records``, which is what makes banking-by-index sound.
    Do not pass a seed here.

    Yields ``(batch_scenarios, record_indices)``.
    """
    from vmax.simulator import make_data_generator

    chunk = int(chunk_size or num_records)
    gen = make_data_generator(
        path=path,
        max_num_objects=max_num_objects,
        include_sdc_paths=include_sdc_paths,
        seed=None,  # never shuffle: index identity depends on file order
        batch_dims=(chunk,),
        repeat=None,
    )
    n_chunks = math.ceil(num_records / chunk)
    for k in range(n_chunks):
        batch = next(gen)
        idx = (np.arange(chunk) + k * chunk) % num_records
        yield batch, idx


class Banker:
    """Periodic deterministic rollout that banks newly-clean scenarios.

    Built against ``make_env_for_evaluation`` (no AutoResetWrapper), so a slot
    holds its scenario for the whole rollout and the slot index stays a valid
    identity. Passing a training env here would silently corrupt the mapping the
    first time a scenario terminates early.
    """

    def __init__(
        self,
        env,
        policy_fn,
        bank: TrajectoryBank,
        scenario_chunks: list[tuple[Any, np.ndarray]],
        scenario_length: int,
        seed: int = 0,
        max_yaw_rate_rad_s: float = MAX_YAW_RATE_RAD_S,
    ) -> None:
        import jax

        self.env = env
        self.bank = bank
        self.chunks = scenario_chunks
        self.scenario_length = int(scenario_length)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self.key = jax.random.PRNGKey(seed)
        self._policy_fn = policy_fn
        self._rollout = jax.jit(self._build_rollout())

    def _build_rollout(self):
        import jax
        import jax.numpy as jnp
        from vmax.agents.pipeline import inference

        env, policy_fn, length = self.env, self._policy_fn, self.scenario_length

        def rollout(policy_params, scenarios, key):
            policy = policy_fn(policy_params, deterministic=True)
            n = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
            reset_keys = jax.random.split(key, n)
            et = env.reset(scenarios, reset_keys)

            def body(carry, _):
                et = carry
                et, _ = inference.policy_step(et, env, policy)
                out = {k: et.metrics[k] for k in _BANK_METRIC_KEYS}
                out.update({k: et.metrics[k] for k in _BANK_SERIES_KEYS})
                out["done"] = et.done
                return et, out

            final_et, per_step = jax.lax.scan(body, et, (), length=length)

            # [T, N] -> [N, T]
            per_step = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), per_step)

            # SDC rows of the rolled-out trajectory: [N, num_objects, T] -> [N, T]
            traj = final_et.state.sim_trajectory
            sdc = jnp.argmax(final_et.state.object_metadata.is_sdc.astype(jnp.int32), axis=-1)
            take = lambda arr: jnp.take_along_axis(arr, sdc[:, None, None], axis=1)[:, 0, :]
            trajectory = {
                "x": take(traj.x),
                "y": take(traj.y),
                "yaw": take(traj.yaw),
                "vel_x": take(traj.vel_x),
                "vel_y": take(traj.vel_y),
                "valid": take(traj.valid),
            }

            # The goal, in the trajectory's own index space. `per_step` metrics are
            # indexed by policy step (length `scenario_length`) while `trajectory` is
            # indexed by absolute sim timestep, and the two are offset by the logged
            # history -- so deriving the goal here and re-computing distance against
            # `trajectory` in numpy avoids having to align them at all. Definition is
            # V-Max's own (`operations.get_sdc_goal_xy`): the SDC's last valid *logged*
            # position.
            log = final_et.state.log_trajectory
            log_xy = jnp.take_along_axis(log.xy, sdc[:, None, None, None], axis=1)[:, 0]
            log_valid = jnp.take_along_axis(log.valid, sdc[:, None, None], axis=1)[:, 0]
            steps = jnp.arange(log_valid.shape[-1])[None, :]
            last_valid = jnp.max(jnp.where(log_valid, steps, -1), axis=-1)
            last_valid = jnp.maximum(last_valid, 0)
            goal_xy = jnp.take_along_axis(log_xy, last_valid[:, None, None], axis=1)[:, 0]

            return per_step, trajectory, goal_xy

        return rollout

    def step(self, params, env_step: int) -> dict[str, float]:
        """Roll out every scenario deterministically and bank the newly-clean ones.

        ``params`` may be a full {SAC,BCSAC}NetworkParams or bare policy params.
        We pull ``.policy`` out here rather than tracing the whole struct: older
        checkpoints predate ``log_alpha``, and flattening a struct with a missing
        declared field raises inside jit.
        """
        import jax

        policy_params = getattr(params, "policy", params)
        clean_all = np.zeros(self.bank.num_scenarios, dtype=bool)
        flags_all: dict[str, np.ndarray] = {}
        traj_all: dict[str, np.ndarray] = {}

        for scenarios, record_idx in self.chunks:
            self.key, sub = jax.random.split(self.key)
            per_step, trajectory, goal_xy = self._rollout(policy_params, scenarios, sub)
            per_step = jax.tree_util.tree_map(np.asarray, per_step)
            trajectory = jax.tree_util.tree_map(np.asarray, trajectory)
            goal_xy = np.asarray(goal_xy)

            done = per_step.pop("done")
            for k in _BANK_SERIES_KEYS:
                per_step.pop(k, None)  # kept per-step; a max over it would be meaningless
            flags = episode_flags(per_step, done)

            trajectory, cut, gate = self._cut_and_gate(trajectory, goal_xy)
            # Clean stays the failure-set criterion, verbatim. The smoothness gate is a
            # separate BC-quality veto: a chattering rollout still *solved* the scene, we
            # just decline to distill it and leave the scenario open for a better one.
            clean = clean_mask_from(flags) & gate
            flags["max_yaw_rate"] = max_yaw_rate(trajectory["yaw"], cut, start_t=self._start_t(trajectory))
            flags["cut_step"] = cut.astype(np.float32)

            # Scatter chunk slots back to record indices. Wrap-around duplicates
            # (last chunk) just rewrite the same value, which is harmless.
            for slot, rec in enumerate(record_idx):
                if clean[slot]:
                    clean_all[rec] = True
                for k, v in flags.items():
                    flags_all.setdefault(k, np.zeros(self.bank.num_scenarios, np.float32))
                    flags_all[k][rec] = v[slot]
                for k, v in trajectory.items():
                    traj_all.setdefault(k, np.zeros((self.bank.num_scenarios,) + v.shape[1:], v.dtype))
                    traj_all[k][rec] = v[slot]

        # Quality = peak raw yaw rate over the kept prefix, lower is better. Same statistic
        # the gate uses, so "upgrade" and "passes the gate" mean the same thing.
        return self.bank.update(
            clean_all,
            traj_all,
            step=env_step,
            flags=flags_all,
            quality=flags_all.get("max_yaw_rate"),
        )

    def _start_t(self, trajectory: dict[str, np.ndarray]) -> int:
        """First policy-controlled step: everything before it is the logged history.

        The rollout runs exactly `scenario_length` policy steps into a trajectory of
        `T` absolute timesteps, so the history is whatever is left over.
        """
        return max(int(trajectory["valid"].shape[1]) - self.scenario_length, 0)

    def _cut_and_gate(self, trajectory, goal_xy):
        start_t = self._start_t(trajectory)
        dist = np.linalg.norm(
            np.stack([trajectory["x"], trajectory["y"]], axis=-1) - goal_xy[:, None, :], axis=-1
        )
        cut = goal_cut_index(dist, trajectory["valid"], start_t=start_t)
        gate = smooth_mask_from(
            trajectory, cut, start_t=start_t, max_yaw_rate_rad_s=self.max_yaw_rate_rad_s
        )
        return truncate_at(trajectory, cut), cut, gate
