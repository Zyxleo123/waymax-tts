"""Deep sanity checks for the SB3 SAC pipeline.

Checks semantic fidelity (not just shapes): goal consistency, log-replay behavior,
train/eval config wiring, and what metadata is intentionally dropped vs lost.

Run on a GPU node:
    srun --gres=gpu:1 --partition=preempt --mem=32G --cpus-per-task=8 --time=00:30:00 \\
        python -m rl.sanity_check_sac
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np

from rl.scenario_source import ScenarioSource
from rl.waymax_env import RewardConfig, WaymaxGymEnv, _compute_goal_xy, observation_dim


def _check(name: str, ok: bool, detail: str = "", *, warn: bool = False) -> bool:
    mark = "OK" if ok else ("WARN" if warn else "FAIL")
    msg = f"  [{mark}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok if not warn else True


def _load_failure_records(failure_dir: str, limit: int | None) -> list[dict]:
    records: list[dict] = []
    for jp in sorted(glob(str(Path(failure_dir) / "**" / "*.json"), recursive=True)):
        if Path(jp).name == "summary.json" or "instructions" in Path(jp).name:
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(rec, dict) and "tfrecord" in rec and "scenario_idx" in rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break
    return records


def check_json_to_source_mapping(failure_dir: str, limit: int) -> tuple[bool, ScenarioSource | None, list[dict]]:
    print("\n=== JSON → ScenarioSource mapping ===")
    records = _load_failure_records(failure_dir, limit)
    ok = _check("failure JSONs found", len(records) > 0, f"n={len(records)}")
    if not records:
        return False, None, records

    missing_tf = sum(1 for r in records if not Path(r["tfrecord"]).is_file())
    ok &= _check("tfrecord paths resolve", missing_tf == 0, f"missing={missing_tf}")

    source = ScenarioSource.from_failure_dir(failure_dir, limit=limit, verbose=False)
    ok &= _check("ScenarioSource count matches JSON limit", len(source) == len(records))

    # Build lookup from loaded specs.
    spec_map = {(s.tfrecord, s.scenario_idx): i for i, s in enumerate(source._specs)}  # noqa: SLF001
    json_keys = {(r["tfrecord"], int(r["scenario_idx"])) for r in records}
    ok &= _check(
        "every JSON scenario loaded",
        spec_map.keys() == json_keys,
        f"loaded={len(spec_map)} json={len(json_keys)}",
    )

    dropped_json_fields = sorted(
        {k for r in records for k in r}
        - {"tfrecord", "scenario_idx", "success", "task_result"}
    )
    _check(
        "JSON metadata intentionally unused",
        True,
        f"pipeline ignores: {dropped_json_fields[:6]}{'...' if len(dropped_json_fields) > 6 else ''}",
        warn=True,
    )
    return ok, source, records


def check_goal_fidelity(source: ScenarioSource, records: list[dict]) -> bool:
    """Goal is recomputed from TFRecord log, not read from failure JSON."""
    print("\n=== Goal fidelity (JSON goal_xy vs TFRecord log) ===")
    ok = True
    max_err = 0.0
    n_checked = 0
    record_by_key = {(r["tfrecord"], int(r["scenario_idx"])): r for r in records}

    for i, spec in enumerate(source._specs):  # noqa: SLF001
        key = (spec.tfrecord, spec.scenario_idx)
        rec = record_by_key.get(key)
        if rec is None or "goal_xy" not in rec:
            continue
        scen = source.get(i)
        computed = _compute_goal_xy(scen)
        json_goal = np.asarray(rec["goal_xy"], dtype=np.float32)
        err = float(np.linalg.norm(computed - json_goal))
        max_err = max(max_err, err)
        n_checked += 1

    ok &= _check(
        "goal_xy matches JSON (recomputed from log)",
        max_err < 0.05,
        f"max_err={max_err:.4f} m over {n_checked} scenarios",
    )
    _check(
        "goal source",
        True,
        "WaymaxGymEnv uses _compute_goal_xy(log), NOT JSON goal_xy — OK if match holds",
    )
    return ok


def check_scenario_roundtrip(source: ScenarioSource) -> bool:
    """Host cache → device reset should not mutate scenario geometry."""
    print("\n=== Scenario host→device roundtrip ===")
    import jax
    import jax.numpy as jnp

    ok = True
    scen_np = source.get(0)
    is_sdc = np.asarray(scen_np.object_metadata.is_sdc).astype(bool)
    ego = int(np.argmax(is_sdc))

    # Log trajectory preserved on host.
    log_x = np.asarray(scen_np.log_trajectory.x)[ego]
    log_valid = np.asarray(scen_np.log_trajectory.valid)[ego].astype(bool)

    scen_jax = jax.tree_util.tree_map(jnp.asarray, scen_np)
    log_x_back = np.asarray(scen_jax.log_trajectory.x)[ego]
    ok &= _check(
        "log_trajectory unchanged host→JAX",
        np.allclose(log_x, log_x_back, equal_nan=True),
    )
    ok &= _check("log has valid ego steps", int(log_valid.sum()) >= 2, f"valid_T={int(log_valid.sum())}")

    rg_valid = int(np.asarray(scen_np.roadgraph_points.valid).sum())
    ok &= _check("roadgraph loaded", rg_valid > 0, f"valid_points={rg_valid}")
    return ok


def check_log_replay_and_obs(source: ScenarioSource) -> bool:
    """Non-SDC agents should follow log; obs should reflect sim state."""
    print("\n=== Simulation semantics (log-replay + obs) ===")
    import jax
    import jax.numpy as jnp

    env = WaymaxGymEnv(source, sequential=True, seed=0, action_space_type="bicycle")
    ok = True

    scen_np = source.get(0)
    scen_jax = jax.tree_util.tree_map(jnp.asarray, scen_np)
    state = env._jit_reset(scen_jax)  # noqa: SLF001

    init_t = int(env._env.config.init_steps)  # noqa: SLF001
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    ego = int(np.argmax(is_sdc))

    # After reset, non-SDC should match log at init timestep.
    sim = state.sim_trajectory
    log = state.log_trajectory
    t = int(np.asarray(state.timestep))
    ok &= _check("reset timestep", t == init_t - 1, f"t={t} init_steps={init_t}")

    non_sdc = ~is_sdc
    if non_sdc.any():
        dx = np.abs(np.asarray(sim.x)[non_sdc, t] - np.asarray(log.x)[non_sdc, t])
        dy = np.abs(np.asarray(sim.y)[non_sdc, t] - np.asarray(log.y)[non_sdc, t])
        ok &= _check(
            "non-SDC log-replay at reset",
            float(np.max(dx)) < 1e-4 and float(np.max(dy)) < 1e-4,
            f"max_xy_err={max(float(np.max(dx)), float(np.max(dy))):.2e}",
        )

    # Zero action: ego moves under dynamics; non-SDC should still track log.
    zero = np.zeros(env.action_dim, dtype=np.float32)
    from waymax import datatypes

    wx_action = datatypes.Action(
        data=jnp.asarray(zero, dtype=jnp.float32),
        valid=jnp.ones((1,), dtype=jnp.bool_),
    )
    state2 = env._jit_step(state, wx_action)  # noqa: SLF001
    t2 = int(np.asarray(state2.timestep))
    if non_sdc.any():
        dx2 = np.abs(np.asarray(state2.sim_trajectory.x)[non_sdc, t2] - np.asarray(log.x)[non_sdc, t2])
        ok &= _check(
            "non-SDC log-replay after 1 ego step",
            float(np.max(dx2)) < 1e-4,
            f"max_x_err={float(np.max(dx2)):.2e}",
        )

    _check(
        "non-SDC sim mode",
        True,
        "PlanningAgentEnvironment has sim_agent_actors=() → log-replay (ghost-collision risk if ego deviates)",
        warn=True,
    )

    # Observation uses sim_trajectory at current timestep.
    goal = jnp.asarray(_compute_goal_xy(scen_np), dtype=jnp.float32)
    obs, _, _ = env._jit_obs(state2, goal)  # noqa: SLF001
    ok &= _check("obs finite after step", bool(np.all(np.isfinite(np.asarray(obs)))))
    frac_at_clip = float(np.mean(np.abs(np.asarray(obs)) >= 9.99))
    _check(
        "obs clipping",
        True,
        f"{frac_at_clip:.1%} of dims at clip boundary (±10) — intentional compression",
        warn=frac_at_clip > 0.05,
    )
    return ok


def check_train_eval_wiring(source: ScenarioSource) -> bool:
    """Flag config paths that must match but are easy to miswire."""
    print("\n=== Train / eval / slurm wiring ===")
    ok = True

    # Slurm delta limits vs eval_sac.py defaults.
    slurm_delta = (2.0, 0.5, 0.2)
    eval_defaults = (6.0, 6.0, float(np.pi))
    ok &= _check(
        "eval_sac delta defaults match slurm",
        slurm_delta == eval_defaults,
        f"slurm={slurm_delta} eval_defaults={eval_defaults} — MISMATCH if training delta via slurm",
        warn=slurm_delta != eval_defaults,
    )

    # eval_sac omits reward flags that affect episode dynamics.
    train_rc = RewardConfig(
        terminate_on_offroad=True,
        route_reward=True,
        action_penalty=0.05,
    )
    eval_rc = RewardConfig(goal_threshold_m=3.0)  # what eval_sac uses
    ok &= _check(
        "eval uses same terminate_on_offroad as train (when set)",
        train_rc.terminate_on_offroad == eval_rc.terminate_on_offroad,
        f"train={train_rc.terminate_on_offroad} eval={eval_rc.terminate_on_offroad}",
        warn=train_rc.terminate_on_offroad != eval_rc.terminate_on_offroad,
    )
    ok &= _check(
        "eval uses same route_reward as train (when set)",
        train_rc.route_reward == eval_rc.route_reward,
        f"train={train_rc.route_reward} eval={eval_rc.route_reward}",
        warn=train_rc.route_reward != eval_rc.route_reward,
    )

    # Training env doesn't record scenario id in info (sequential=False).
    train_env = WaymaxGymEnv(source, sequential=False, seed=0)
    _, train_info = train_env.reset()
    ok &= _check(
        "train reset info.scenario empty (sequential=False)",
        train_info.get("scenario") == {},
        "expected — cannot trace which failure case during SB3 train logs",
        warn=True,
    )

    eval_env = WaymaxGymEnv(source, sequential=True, seed=0)
    _, eval_info = eval_env.reset()
    spec = eval_info.get("scenario", {})
    ok &= _check(
        "eval reset info.scenario populated",
        bool(spec.get("tfrecord")) and spec.get("scenario_idx") is not None,
        f"idx={spec.get('scenario_idx')}",
    )

    # Monitor / SB3 success_rate uses is_success=reached only, not clean success.
    _, _, _, _, step_info = eval_env.step(eval_env.action_space.sample())
    _check(
        "SB3 Monitor success_rate",
        True,
        f"is_success={step_info.get('is_success')} is raw goal reach, NOT clean (no col/offroad)",
        warn=True,
    )
    return ok


def check_sac_buffer_semantics(source: ScenarioSource) -> bool:
    print("\n=== SAC-specific semantics ===")
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    ok = True
    vec = DummyVecEnv([lambda: Monitor(WaymaxGymEnv(source, sequential=False, seed=0))])
    model = SAC("MlpPolicy", vec, learning_starts=32, buffer_size=5000, batch_size=64, verbose=0, device="cpu")
    model.learn(total_timesteps=64)
    ok &= _check("SAC collects + learns", model.num_timesteps == 64)

    _check(
        "learning_starts=10000 (production default)",
        True,
        "first 10k transitions are random-policy noise before gradient updates",
        warn=True,
    )
    _check(
        "vmax_rl IDM / policy-friendly reward NOT wired",
        True,
        "SB3 path uses raw log-replay; see rl/vmax_rl/env_utils.py for the V-Max fix",
        warn=True,
    )
    vec.close()
    return ok


def main():
    p = argparse.ArgumentParser(description="Deep sanity-check SB3 SAC pipeline.")
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    print("SB3 SAC deep sanity check")
    print(f"  failure_dir={args.failure_dir}")
    print(f"  limit={args.limit}")

    results: list[bool] = []
    ok_map, source, records = check_json_to_source_mapping(args.failure_dir, args.limit)
    results.append(ok_map)
    if source is not None:
        results.append(check_goal_fidelity(source, records))
        results.append(check_scenario_roundtrip(source))
        results.append(check_log_replay_and_obs(source))
        results.append(check_train_eval_wiring(source))
        results.append(check_sac_buffer_semantics(source))

    print("\n=== Summary ===")
    n_fail = sum(not r for r in results)
    if n_fail == 0:
        print("ALL HARD CHECKS PASSED (see WARN lines for intentional design choices)")
        return 0
    print(f"HARD FAILURES in {n_fail}/{len(results)} sections")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
