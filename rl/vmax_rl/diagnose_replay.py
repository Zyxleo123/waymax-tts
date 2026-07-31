"""Quantify the reward-corrupting sim artifact and the kinematic-refit fix.

Reproduces the "BREAKING FINDING" (``rl/CONTEXT.md``) and measures whether the
:mod:`rl.vmax_rl.kinematic_refit` SDC log refit resolves it. It compares three
variants on a batch of failure scenarios:

1. ``log_gt``        -- the **ground-truth** human log scored directly
   (``sim_trajectory := log_trajectory``, Waymax overlap/offroad per timestep).
   The trustworthy reference: the recorded scene is (nearly) collision-free.
2. ``bicycle_raw``   -- *closed-loop expert replay* (the SDC follows the analytic
   bicycle inverse of its own log) on the **raw** WOMD log. Reproduces the
   artifact: the bicycle cannot track the log, drifts, and hits the frozen
   log-replayed agents -> ghost collisions / offroads.
3. ``bicycle_refit`` -- same replay on the **kinematically-refit** SDC log
   (:func:`rl.vmax_rl.kinematic_refit.refit_state_sdc_log`). The fix: tracking
   error collapses and the reward becomes clean/learnable again.

Reported per variant: ``collision`` / ``offroad`` (fraction of scenes that ever
overlap / go offroad), ``drift_mean`` / ``drift_max`` (SDC sim-vs-log position
error, m), and the bimodal split ``frac_track`` (<0.1 m) / ``frac_diverge``
(>1 m) that the finding describes.

Example (on a GPU node / inside an srun alloc)::

    python -m rl.vmax_rl.diagnose_replay \
        --failure-dir /zfsauton/scratch/mineuih/waymax_rs/failure_samples \
        --limit 32 --scenario-length 80
"""

from __future__ import annotations

import argparse
import json
from functools import partial

from rl.vmax_rl import compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

# V-Max's ``make_env`` re-registers its custom metrics every call, which raises
# the second time we build an env. Make registration idempotent before importing
# anything that triggers it.
import waymax.metrics as _waymax_metrics
from waymax.metrics import metric_factory as _metric_factory

_orig_register_metric = _metric_factory.register_metric


def _idempotent_register_metric(name, metric):
    if name in _metric_factory._METRICS_REGISTRY:
        return
    _orig_register_metric(name, metric)


_metric_factory.register_metric = _idempotent_register_metric
_waymax_metrics.register_metric = _idempotent_register_metric

from waymax import dynamics  # noqa: E402

from vmax.agents.pipeline import inference  # noqa: E402
from vmax.simulator import make_env_for_evaluation  # noqa: E402

from rl.vmax_rl import data  # noqa: E402
from rl.vmax_rl.kinematic_refit import refit_state_sdc_log  # noqa: E402

_INIT_STEPS = 11  # PlanningAgentEnvironment warmup (control starts at timestep 10)


def _build_env(dynamics_model, max_num_objects: int):
    """Eval env with early termination disabled (so we measure full-horizon drift)."""
    return make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics_model,
        sdc_paths_from_data=True,
        observation_type="gt",
        observation_config=None,
        reward_type="",
        termination_keys=[],  # never terminate: replay the whole horizon
    )


def _replay_chunk(env, scenarios, scenario_length: int, key: jax.Array):
    """Closed-loop expert replay of a batch of scenarios via the bicycle inverse."""
    batch = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
    reset_keys = jax.random.split(key, batch)
    transition = env.reset(scenarios, reset_keys)

    step_fn = partial(inference.expert_step, env=env)

    def body(tr, _):
        next_tr, _ = step_fn(tr)
        out = {
            "overlap": next_tr.metrics.get("overlap", jnp.zeros(batch)),
            "offroad": next_tr.metrics.get("offroad", jnp.zeros(batch)),
        }
        return next_tr, out

    final_tr, stacks = jax.lax.scan(body, transition, (), length=scenario_length)
    return final_tr.state, stacks


def _ground_truth_chunk(scenarios, scenario_length: int):
    """Score the raw human log directly: ``sim := log``, Waymax metrics per step."""
    batch = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
    gt = scenarios.replace(sim_trajectory=scenarios.log_trajectory)
    overlap_metric = _waymax_metrics.OverlapMetric()
    offroad_metric = _waymax_metrics.OffroadMetric()
    is_sdc = gt.object_metadata.is_sdc  # (B, O)

    def body(_, t):
        s = gt.replace(timestep=t)
        ov = jnp.sum(overlap_metric.compute(s).value * is_sdc, axis=-1)  # (B,)
        of = jnp.sum(offroad_metric.compute(s).value * is_sdc, axis=-1)  # (B,)
        return _, {"overlap": ov, "offroad": of}

    timesteps = jnp.arange(_INIT_STEPS, _INIT_STEPS + scenario_length)
    _, stacks = jax.lax.scan(body, None, timesteps)
    return gt, stacks


def _score(state, stacks) -> dict:
    """Host-side reduction of one variant into per-scenario arrays."""
    overlap = np.asarray(stacks["overlap"])  # (T, B)
    offroad = np.asarray(stacks["offroad"])  # (T, B)
    collision = (overlap > 0).any(0)  # (B,)
    went_offroad = (offroad > 0).any(0)  # (B,)

    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)  # (B, O)
    sdc = is_sdc.argmax(-1)  # (B,)
    sim_xy = np.asarray(state.sim_trajectory.xy)  # (B, O, Tt, 2)
    log_xy = np.asarray(state.log_trajectory.xy)
    sim_valid = np.asarray(state.sim_trajectory.valid)  # (B, O, Tt)
    log_valid = np.asarray(state.log_trajectory.valid)

    B = sim_xy.shape[0]
    drift_max = np.zeros(B, np.float32)
    drift_mean = np.zeros(B, np.float32)
    for b in range(B):
        s = sdc[b]
        m = sim_valid[b, s] & log_valid[b, s]
        if not m.any():
            continue
        d = np.linalg.norm(sim_xy[b, s][m] - log_xy[b, s][m], axis=-1)
        drift_max[b] = float(d.max())
        drift_mean[b] = float(d.mean())

    return {
        "collision": collision,
        "offroad": went_offroad,
        "drift_max": drift_max,
        "drift_mean": drift_mean,
    }


def _summarize(scores: dict) -> dict:
    drift_max = scores["drift_max"]
    return {
        "collision": float(scores["collision"].mean()),
        "offroad": float(scores["offroad"].mean()),
        "drift_mean_m": float(scores["drift_mean"].mean()),
        "drift_max_m": float(drift_max.mean()),
        "frac_track_0p1m": float((drift_max <= 0.1).mean()),
        "frac_diverge_1m": float((drift_max > 1.0).mean()),
    }


def main() -> None:
    args = _parse_args()
    print(f"[diagnose] devices={jax.local_device_count()} backend={jax.default_backend()}")

    print("[diagnose] loading failure scenarios...")
    stacked, num = data.load_failure_scenarios(
        args.failure_dir, max_num_objects=args.max_num_objects, limit=args.limit, verbose=True
    )

    bicycle = dynamics.InvertibleBicycleModel(normalize_actions=True)
    env = _build_env(bicycle, args.max_num_objects)
    replay = jax.jit(partial(_replay_chunk, env, scenario_length=args.scenario_length))
    gt_fn = jax.jit(partial(_ground_truth_chunk, scenario_length=args.scenario_length))

    # variant name -> (per-chunk runner, refit?)
    variants = {
        "log_gt": (lambda sub, key: gt_fn(sub), False),
        "bicycle_raw": (lambda sub, key: replay(sub, key=key), False),
        "bicycle_refit": (lambda sub, key: replay(sub, key=key), True),
    }

    results: dict[str, dict] = {}
    for name, (runner, refit) in variants.items():
        key = jax.random.PRNGKey(args.seed)
        agg = {k: [] for k in ("collision", "offroad", "drift_max", "drift_mean")}
        for start in range(0, num, args.chunk_size):
            end = min(start + args.chunk_size, num)
            sub = jax.tree_util.tree_map(lambda x, s=start, e=end: jnp.asarray(x[s:e]), stacked)
            if refit:
                sub = refit_state_sdc_log(sub)
            key, rk = jax.random.split(key)
            state, stacks = runner(sub, rk)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), stacks)
            sc = _score(state, stacks)
            for k in agg:
                agg[k].append(sc[k])
        scores = {k: np.concatenate(v) for k, v in agg.items()}
        results[name] = _summarize(scores)
        print(f"[diagnose] {name:>14}: {results[name]}")

    print("\n==================== EXPERT-REPLAY DIAGNOSIS ====================")
    cols = ["collision", "offroad", "drift_mean_m", "drift_max_m",
            "frac_track_0p1m", "frac_diverge_1m"]
    header = f"  {'variant':>14} | " + " | ".join(f"{c:>15}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, s in results.items():
        row = f"  {name:>14} | " + " | ".join(f"{s[c]:>15.3f}" for c in cols)
        print(row)
    print(f"\n  ({num} scenarios, horizon={args.scenario_length})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"num_scenarios": num, "scenario_length": args.scenario_length,
                       "results": results}, f, indent=2)
        print(f"  results -> {args.out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose the expert-replay reward artifact + refit fix.")
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--limit", type=int, default=32, help="Cap scenarios (default 32 for a quick check).")
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--scenario-length", type=int, default=80)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional JSON path for the summary.")
    return p.parse_args()


if __name__ == "__main__":
    main()
