"""Gate 1: prove the environment is trustworthy before any fine-tuning.

Runs the six Phase-1 gates from the plan against the *real* checkpoint and real
scenarios:

1. Expert replay produces clean metrics (no collision / offroad; goal reached).
2. Batched and single-scene rollout rewards match (given identical actions).
3. Saved per-step rewards sum exactly to the episode return.
4. Replan observations change when the simulated ego deviates from the log.
5. Permuting a batch permutes rewards and trajectories identically.
6. A frozen checkpoint reproduces its own rollout (bitwise) under a fixed seed.

Configuration is by environment variable:

* ``DFT_CKPT``        -- checkpoint tag dir (default: base without-subgoal ckpt).
* ``DFT_TFRECORD``    -- a WOMD tfrecord shard.
* ``DFT_INDICES``     -- comma-separated scenario indices (>= 2 for parity gates).
* ``DFT_RG_POINTS``   -- roadgraph point cap (default 30000; lower for CPU smoke).
* ``DFT_MAX_REPLANS`` -- cap outer replans/episode (default: run to horizon).

The env is sized to each loaded scenario's actual object count (WOMD pads objects
and ignores an under-count request), so no ``max_num_objects`` is passed in.
Exit code is non-zero if any gate fails.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

# Persistent compile cache: many fresh envs recompile the same jitted step/metrics
# shapes, so caching to disk makes the 2nd+ compile cheap (keeps CPU smoke runs
# under login-node watchdog limits). Set before any jit fires.
_CACHE_DIR = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    "/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache",
)
try:
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:  # noqa: BLE001
    pass

from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig


# Module-level knobs, set from env vars in main().
_RG_POINTS = 30000
_MAX_REPLANS: int | None = None


# --------------------------------------------------------------------------- #
def _load_batch(tfr: str, indices: list[int]):
    """Batched SimulatorState for ``indices``, ordered to match the request.

    Uses the gymnasium-free fast loader (as the eval runner does); it returns
    scenarios in ascending index order plus a scenario->position map, so we gather
    back into the requested order (parity/permutation gates depend on the batch
    axis matching ``indices`` exactly). ``max_num_objects=None`` keeps each
    scene's real object count.
    """
    n_unique = len(sorted(set(int(i) for i in indices)))
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING,
        path=str(tfr),
        max_num_objects=None,
        max_num_rg_points=int(_RG_POINTS),
        batch_dims=(n_unique,),
        shuffle_seed=0,
        include_sdc_paths=False,
    )
    state, idx_map = load_scenario_state_batch_fast(ds_cfg, indices)
    order = jnp.asarray([idx_map[int(i)] for i in indices], dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[order], state)


def _num_objects(state) -> int:
    return int(jnp.asarray(state.log_trajectory.x).shape[1])


def _fresh_env(ckpt_bundle, tfr, indices, *, use_ema=False, seed=0, reset_seed=None):
    """Load ``indices``, build an env sized to the data, reset, and return it."""
    state = _load_batch(tfr, indices)
    env = DiffusionWaymaxEnv(
        ckpt_bundle,
        reward_config=RewardConfig(),
        max_num_objects=_num_objects(state),
        use_ema=use_ema,
        seed=seed,
        max_replans=_MAX_REPLANS,
    )
    env.reset(state, seed=reset_seed)
    return env


def _cfg():
    ckpt = os.environ.get("DFT_CKPT", DEFAULT_CHECKPOINT_PATH)
    tfr = os.environ.get(
        "DFT_TFRECORD",
        "/zfsauton/scratch/eshau/womd/tf_example/training/"
        "training_tfexample.tfrecord-00000-of-01000",
    )
    indices = [int(x) for x in os.environ.get("DFT_INDICES", "0,1,2,3").split(",") if x.strip()]
    return ckpt, tfr, indices


def _close(a, b, atol=1e-4, rtol=1e-4):
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))


# --------------------------------------------------------------------------- #
def gate1_expert_replay(ckpt_bundle, tfr, indices) -> bool:
    """Expert (log) replay must be clean and reach the goal."""
    env = _fresh_env(ckpt_bundle, tfr, indices)
    out = env.expert_replay_episode()
    overlap = np.asarray(out["overlap_any_b"])
    offroad = np.asarray(out["offroad_any_b"])
    reached = np.asarray(out["reached_any_b"])
    print(f"  expert replay: overlap={overlap.tolist()} offroad={offroad.tolist()} "
          f"reached={reached.tolist()} return={np.asarray(out['episode_return_b']).tolist()}")
    clean = (~overlap).all() and (~offroad).all()
    # A capped smoke run may stop before the goal; only require goal-reaching when
    # running full episodes.
    reach_ok = True if _MAX_REPLANS is not None else bool(reached.any())
    return bool(clean and reach_ok)


def gate2_batched_equals_single(ckpt_bundle, tfr, indices) -> bool:
    """Reward/return under identical (log) actions is independent of batch."""
    env_b = _fresh_env(ckpt_bundle, tfr, indices)
    ret_b = np.asarray(env_b.expert_replay_episode()["episode_return_b"])

    ok = True
    for pos, idx in enumerate(indices):
        env_s = _fresh_env(ckpt_bundle, tfr, [idx])
        ret_s = float(np.asarray(env_s.expert_replay_episode()["episode_return_b"])[0])
        match = _close(ret_b[pos], ret_s)
        print(f"  scene {idx}: batched={ret_b[pos]:.5f} single={ret_s:.5f} match={match}")
        ok = ok and match
    return ok


def gate3_per_step_sums_to_return(ckpt_bundle, tfr, indices) -> bool:
    """Sum of per-step rewards == episode return, per world."""
    env = _fresh_env(ckpt_bundle, tfr, indices)
    out = env.expert_replay_episode()
    per_step_sum = np.asarray(jnp.sum(out["per_step_reward_bk"], axis=-1))
    ret = np.asarray(out["episode_return_b"])
    outer = np.asarray(out["outer_reward_b"])
    ok = _close(per_step_sum, ret, atol=1e-5) and _close(outer, ret, atol=1e-5)
    print(f"  per_step_sum={per_step_sum.tolist()} return={ret.tolist()} outer={outer.tolist()} ok={ok}")
    return bool(ok)


def gate4_obs_changes_on_deviation(ckpt_bundle, tfr, indices) -> bool:
    """After one replan, features from a deviating rollout differ from log replay."""
    idx = indices[0]

    def one_step_features(deviate: bool):
        env = _fresh_env(ckpt_bundle, tfr, [idx])
        log_world = env.ego_log_world_trajectory()  # [1, T, 5]
        t = env.timestep
        window = np.asarray(log_world[:, t : t + env._replan + 1, :]).copy()
        if deviate:
            window[:, 1:, 1] += 3.0  # push ego ~3 m laterally over the prefix
        out = env.step(trajectory_world_bt5=jnp.asarray(window))
        return out.features

    f_log = one_step_features(deviate=False)
    f_dev = one_step_features(deviate=True)
    keys = ["goal_xy", "ego_state", "other_states", "map_features"]
    diff = sum(float(np.linalg.norm(np.asarray(f_log[k]) - np.asarray(f_dev[k]))) for k in keys)
    print(f"  feature L2 difference (log vs deviated) = {diff:.4f}")
    return diff > 1e-2


def gate5_permutation_equivariance(ckpt_bundle, tfr, indices) -> bool:
    """Permuting scenarios permutes returns identically."""
    env = _fresh_env(ckpt_bundle, tfr, indices)
    ret = np.asarray(env.expert_replay_episode()["episode_return_b"])

    perm = list(reversed(range(len(indices))))
    perm_indices = [indices[i] for i in perm]
    env_p = _fresh_env(ckpt_bundle, tfr, perm_indices)
    ret_p = np.asarray(env_p.expert_replay_episode()["episode_return_b"])

    expected = ret[perm]
    ok = _close(ret_p, expected, atol=1e-5)
    print(f"  return={ret.tolist()} permuted={ret_p.tolist()} expected={expected.tolist()} ok={ok}")
    return bool(ok)


def gate6_frozen_reproducible(ckpt_bundle, tfr, indices) -> bool:
    """Same seed + frozen checkpoint => bitwise-identical sampled rollout."""
    def rollout_once():
        env = _fresh_env(ckpt_bundle, tfr, indices, use_ema=True, reset_seed=123)
        rng = jax.random.PRNGKey(7)
        feats = env._build_features()
        traj = env.sample_trajectory(feats, rng=rng)
        out = env.step(trajectory_world_bt5=traj)
        return np.asarray(traj), np.asarray(out.reward_b)

    traj_a, rew_a = rollout_once()
    traj_b, rew_b = rollout_once()
    ok = _close(traj_a, traj_b, atol=0.0, rtol=0.0) and _close(rew_a, rew_b, atol=0.0, rtol=0.0)
    print(f"  reward run A={rew_a.tolist()} run B={rew_b.tolist()} identical={ok}")
    return bool(ok)


# --------------------------------------------------------------------------- #
def main() -> int:
    global _RG_POINTS, _MAX_REPLANS
    ckpt_path, tfr, indices = _cfg()
    _RG_POINTS = int(os.environ.get("DFT_RG_POINTS", "30000"))
    mr = os.environ.get("DFT_MAX_REPLANS", "").strip()
    _MAX_REPLANS = int(mr) if mr else None
    if len(indices) < 2:
        print("Need >= 2 indices for parity/permutation gates.", file=sys.stderr)
        return 2
    print(f"[gate1] devices={jax.devices()}")
    print(f"[gate1] ckpt={ckpt_path}")
    print(f"[gate1] tfrecord={tfr}")
    print(f"[gate1] indices={indices} rg_points={_RG_POINTS} max_replans={_MAX_REPLANS}")

    ckpt_bundle = load_diffusion_checkpoint(ckpt_path)
    print(f"[gate1] loaded metadata: inst_dim={ckpt_bundle.metadata['inst_dim']} "
          f"max_range={ckpt_bundle.metadata['max_range']} horizon={ckpt_bundle.predict_horizon}")

    all_gates = [
        ("1 expert replay clean", gate1_expert_replay),
        ("2 batched == single", gate2_batched_equals_single),
        ("3 per-step sums to return", gate3_per_step_sums_to_return),
        ("4 obs changes on deviation", gate4_obs_changes_on_deviation),
        ("5 permutation equivariance", gate5_permutation_equivariance),
        ("6 frozen reproducible", gate6_frozen_reproducible),
    ]
    # DFT_GATES selects a subset by leading number (e.g. "1,2,3,4,5" for a CPU
    # smoke that skips the diffusion-sampling gate 6). Default: all.
    sel = os.environ.get("DFT_GATES", "").strip()
    if sel:
        want = {s.strip() for s in sel.split(",") if s.strip()}
        gates = [(n, f) for (n, f) in all_gates if n.split()[0] in want]
    else:
        gates = all_gates

    results = {}
    for name, fn in gates:
        print(f"\n=== Gate {name} ===")
        try:
            results[name] = bool(fn(ckpt_bundle, tfr, indices))
        except Exception:  # noqa: BLE001 - report which gate blew up
            import traceback
            traceback.print_exc()
            results[name] = False
        print(f"  -> {'PASS' if results[name] else 'FAIL'}")

    print("\n================ Gate 1 summary ================")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] Gate {name}")
    all_pass = all(results.values())
    print(f"\nGate 1 overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
