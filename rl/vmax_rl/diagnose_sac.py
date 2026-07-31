"""Diagnose why SAC RL underperforms BC on Waymax failure cases.

Runs a battery of closed-loop rollouts on failure scenarios comparing action
sources (zero, random, expert, trained policy) under several env configurations.
The goal is to pin down *why* SAC plateaus while BC climbs after the
``expert_step`` fix (see ``rl/CONTEXT.md``).

Tests performed per env variant:

1. **Action baselines** — zero-action, random, expert (bicycle inverse),
   optional trained SAC / BC_SAC checkpoints.
2. **Outcome rates** — reached / collision / offroad / clean success.
3. **Reward stats** — mean cumulative ep reward, mean per-step reward, ep length.
4. **Action stats** — mean abs action, L2 norm (detect coast / collapse).
5. **Reward decomposition** — per-term contribution (progression, overlap, …)
   averaged over steps for the expert rollout (the learnable upper bound for BC).

Env variants (``--env-variant``):

* ``default``   — **policy-friendly sim** (refine.py default): IDM + decoupled overlap/offroad.
* ``log_replay`` — legacy: frozen log-replay agents + harsh overlap/offroad terminate/penalties.
* ``decouple`` / ``idm`` — ablation of the two halves of the default preset.

Example (GPU node)::

    python -m rl.vmax_rl.diagnose_sac --limit 32 \\
        --sac-run-dir /zfsauton/scratch/yixiz/waymax_rs/runs/sac_failures \\
        --bcsac-run-dir /zfsauton/scratch/yixiz/waymax_rs/runs/bcsac_failures_bconfail
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from functools import partial
from typing import Callable

from rl.vmax_rl import compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from waymax import dynamics

import waymax.metrics as _waymax_metrics
from waymax.metrics import metric_factory as _metric_factory

_orig_register_metric = _metric_factory.register_metric


def _idempotent_register_metric(name, metric):
    if name in _metric_factory._METRICS_REGISTRY:
        return
    _orig_register_metric(name, metric)


_metric_factory.register_metric = _idempotent_register_metric
_waymax_metrics.register_metric = _idempotent_register_metric

from vmax.agents.pipeline import inference  # noqa: E402
from vmax.simulator import make_env_for_evaluation  # noqa: E402
from vmax.simulator.wrappers import reward as reward_mod  # noqa: E402

from rl.vmax_rl import data, env_utils  # noqa: E402

_BASE_REWARD = {
    "overlap": -1.0,
    "offroad": -1.0,
    "red_light": -1.0,
    "off_route": -0.6,
    "progression": 0.2,
}
_LEGACY_TERMINATION = ["offroad", "overlap", "run_red_light"]
_POLICY_FRIENDLY_REWARD = {**_BASE_REWARD, **env_utils.POLICY_FRIENDLY_REWARD_WEIGHTS}

_ENV_VARIANTS = {
    "default": {
        "termination_keys": list(env_utils.POLICY_FRIENDLY_TERMINATION_KEYS),
        "reward_config": dict(_POLICY_FRIENDLY_REWARD),
        "reactive": True,
    },
    "log_replay": {
        "termination_keys": _LEGACY_TERMINATION,
        "reward_config": dict(_BASE_REWARD),
        "reactive": False,
    },
    "decouple": {
        "termination_keys": list(env_utils.POLICY_FRIENDLY_TERMINATION_KEYS),
        "reward_config": dict(_POLICY_FRIENDLY_REWARD),
        "reactive": False,
    },
    "idm": {
        "termination_keys": _LEGACY_TERMINATION,
        "reward_config": dict(_BASE_REWARD),
        "reactive": True,
    },
}

_ACTION_SOURCES = ("zero", "random", "expert")


# Training observation config (matches base_config.yaml / refine.py).
_OBS_CONFIG = {
    "obs_past_num_steps": 5,
    "objects": {
        "features": ["waypoints", "velocity", "yaw", "size", "valid"],
        "num_closest_objects": 8,
    },
    "roadgraphs": {
        "features": ["waypoints", "direction", "valid"],
        "element_types": [15, 16],
        "interval": 2,
        "max_meters": 70,
        "roadgraph_top_k": 200,
        "meters_box": {"front": 70, "back": 5, "left": 20, "right": 20},
        "max_num_lanes": 10,
        "max_num_points_per_lane": 20,
    },
    "traffic_lights": {
        "features": ["waypoints", "state", "valid"],
        "num_closest_traffic_lights": 5,
    },
    "path_target": {"features": ["waypoints"], "num_points": 3, "points_gap": 12},
}


def _build_env(variant: str, max_num_objects: int):
    cfg = _ENV_VARIANTS[variant]
    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type="vec",
        observation_config=_OBS_CONFIG,
        reward_type="linear",
        reward_config=cfg["reward_config"],
        termination_keys=cfg["termination_keys"],
    )
    if cfg["reactive"]:
        env = env_utils.attach_idm_sim_agents(env, desired_vel=30.0)
    return env


def _load_policy(run_dir: str | None, env):
    if not run_dir:
        return None
    meta_path = os.path.join(run_dir, "run_config.json")
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No run_config.json in {run_dir}")
    meta = json.load(open(meta_path))
    final = os.path.join(model_dir, "model_final.pkl")
    if os.path.isfile(final):
        ckpt = final
    else:
        ckpts = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pkl")]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint in {model_dir}")
        ckpt = os.path.join(model_dir, max(ckpts, key=lambda n: int("".join(c for c in n if c.isdigit()) or "0")))
    params = pickle.loads(open(ckpt, "rb").read())
    from vmax.agents.learning.hybrid.bc_sac import make_inference_fn, make_networks

    alg = meta["algorithm_config"]
    network_config = meta["network_config"]
    network = make_networks(
        observation_size=env.observation_spec(),
        action_size=env.action_spec().data.shape[0],
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        rl_learning_rate=alg["rl_learning_rate"],
        imitation_learning_rate=alg["imitation_learning_rate"],
        network_config=network_config,
    )
    return make_inference_fn(network)(params.policy, deterministic=True)


def _random_step(env, tr, key):
    batch = tr.observation.shape[0] if hasattr(tr.observation, "shape") else 1
    key, sk = jax.random.split(key)
    actions = jax.random.uniform(sk, (batch, 2), minval=-1.0, maxval=1.0)
    from vmax.agents.pipeline.inference import _create_valid_action

    actions = _create_valid_action(actions)
    next_tr = env.step(tr, actions)
    from vmax.agents import datatypes

    partial = datatypes.RLPartialTransition(
        observation=tr.observation,
        action=actions.data,
        reward=next_tr.reward,
        flag=next_tr.flag,
        done=next_tr.done,
    )
    return next_tr, partial


def _zero_step(env, tr, key=None):
    """Batched zero-action step (constant_step only handles batch=1)."""
    batch = tr.observation.shape[0]
    from vmax.agents.pipeline.inference import _create_valid_action
    from vmax.agents import datatypes

    actions = _create_valid_action(jnp.zeros((batch, 2), dtype=jnp.float32))
    next_tr = env.step(tr, actions)
    partial = datatypes.RLPartialTransition(
        observation=tr.observation,
        action=actions.data,
        reward=next_tr.reward,
        flag=next_tr.flag,
        done=next_tr.done,
    )
    return next_tr, partial


def _make_step_fn(env, source: str, policy=None):
    if source == "zero":
        return partial(_zero_step, env)
    if source == "expert":
        return partial(inference.expert_step, env=env, use_partial_transition=True)
    if source == "random":
        return partial(_random_step, env)
    if source == "policy":
        if policy is None:
            raise ValueError("policy source requires a loaded checkpoint")
        return partial(inference.policy_step, env=env, policy_fn=policy, use_partial_transition=True)
    raise ValueError(f"Unknown action source: {source}")


def _rollout_chunk(env, step_fn, scenarios, scenario_length: int, key: jax.Array):
    batch = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
    reset_keys = jax.random.split(key, batch)
    transition = env.reset(scenarios, reset_keys)

    def body(carry, _):
        tr, k = carry
        k, sk = jax.random.split(k)
        next_tr, rl_tr = step_fn(tr, key=sk)
        out = {
            "reward": rl_tr.reward,
            "action": rl_tr.action,
            "overlap": next_tr.metrics.get("overlap", jnp.zeros(batch)),
            "offroad": next_tr.metrics.get("offroad", jnp.zeros(batch)),
            "done": next_tr.done,
            "ep_rew": next_tr.info.get("rewards", jnp.zeros(batch)),
            "ep_len": next_tr.info.get("steps", jnp.zeros(batch)),
        }
        return (next_tr, k), out

    (final_tr, _), stacks = jax.lax.scan(body, (transition, key), (), length=scenario_length)
    return final_tr, stacks


def _goal_outcomes(final_state, stacks, goal_threshold_m: float) -> dict:
    overlap = np.asarray(stacks["overlap"])
    offroad = np.asarray(stacks["offroad"])
    done = np.asarray(stacks["done"]).astype(bool)
    T, B = overlap.shape
    first_done = np.where(done.any(0), done.argmax(0), T - 1)
    step_idx = np.arange(T)[:, None]
    active = step_idx <= first_done[None, :]
    collision = ((overlap > 0) & active).any(0)
    went_offroad = ((offroad > 0) & active).any(0)

    is_sdc = np.asarray(final_state.state.object_metadata.is_sdc).astype(bool)
    sdc = is_sdc.argmax(-1)
    sim_xy = np.asarray(final_state.state.sim_trajectory.xy)
    log_xy = np.asarray(final_state.state.log_trajectory.xy)
    log_valid = np.asarray(final_state.state.log_trajectory.valid)
    sim_valid = np.asarray(final_state.state.sim_trajectory.valid)

    min_goal_dist = np.full(B, np.inf, np.float32)
    for b in range(B):
        s = sdc[b]
        lv = log_valid[b, s]
        if not lv.any():
            continue
        goal = log_xy[b, s, np.flatnonzero(lv)[-1]]
        sv = sim_valid[b, s]
        if not sv.any():
            continue
        d = np.linalg.norm(sim_xy[b, s][sv] - goal[None], axis=-1)
        min_goal_dist[b] = float(d.min())

    reached = min_goal_dist < goal_threshold_m
    clean = reached & (~collision) & (~went_offroad)
    return {
        "reached": reached.astype(np.float32),
        "collision": collision.astype(np.float32),
        "offroad": went_offroad.astype(np.float32),
        "clean_success": clean.astype(np.float32),
        "min_goal_distance_m": min_goal_dist,
    }


def _summarize_rollout(stacks, outcomes) -> dict:
    rewards = np.asarray(stacks["reward"])  # (T, B)
    actions = np.asarray(stacks["action"])  # (T, B, 2)
    ep_rew = np.asarray(stacks["ep_rew"][-1])  # (B,) cumulative at last step
    ep_len = np.asarray(stacks["ep_len"][-1])
    action_l2 = np.linalg.norm(actions, axis=-1)
    return {
        "ep_rew_mean": float(ep_rew.mean()),
        "ep_rew_std": float(ep_rew.std()),
        "step_rew_mean": float(rewards.mean()),
        "ep_len_mean": float(ep_len.mean()),
        "action_abs_mean": float(np.abs(actions).mean()),
        "action_l2_mean": float(action_l2.mean()),
        "reached": float(outcomes["reached"].mean()),
        "collision": float(outcomes["collision"].mean()),
        "offroad": float(outcomes["offroad"].mean()),
        "clean_success": float(outcomes["clean_success"].mean()),
        "median_goal_dist_m": float(np.median(outcomes["min_goal_distance_m"])),
    }


def _reward_decomposition(state, reward_config: dict) -> dict:
    """Per-term mean reward contribution over one SimulatorState timestep."""
    out = {}
    total = 0.0
    for name, weight in reward_config.items():
        try:
            fn = reward_mod._get_reward_fn(name)
            val = float(np.asarray(jax.device_get(fn(state))).mean())
        except Exception:
            val = 0.0
        contrib = val * weight
        out[f"term_{name}"] = contrib
        total += contrib
    out["term_total"] = total
    return out


def _expert_reward_decomposition(scenarios, env, scenario_length: int, key: jax.Array) -> dict:
    """Average per-term reward for expert actions over a rollout horizon."""
    step_fn = partial(inference.expert_step, env=env, use_partial_transition=True)
    batch = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
    reset_keys = jax.random.split(key, batch)
    tr = env.reset(scenarios, reset_keys)
    reward_cfg = env.get_wrapper_attr("_reward_config") if hasattr(env, "get_wrapper_attr") else _BASE_REWARD
    try:
        reward_cfg = env.get_wrapper_attr("_reward_config")
    except AttributeError:
        # Walk wrapper chain
        w = env
        reward_cfg = _BASE_REWARD
        while w is not None:
            if hasattr(w, "_reward_config"):
                reward_cfg = w._reward_config
                break
            w = getattr(w, "env", None)

    agg = {f"term_{k}": [] for k in reward_cfg}
    agg["term_total"] = []

    def body(carry, _):
        tr_, k = carry
        k, sk = jax.random.split(k)
        next_tr, _ = step_fn(tr_, key=sk)
        return (next_tr, k), next_tr.state

    (_, _), states = jax.lax.scan(body, (tr, key), (), length=scenario_length)
    # states is a pytree with leading dim T; reduce on host per step
    for t in range(scenario_length):
        st = jax.tree_util.tree_map(lambda x: x[t], states)
        st_host = jax.device_get(st)
        dec = _reward_decomposition(st_host, reward_cfg)
        for k, v in dec.items():
            agg[k].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def _run_variant(
    variant: str,
    stacked,
    num: int,
    scenario_length: int,
    chunk_size: int,
    goal_threshold_m: float,
    max_num_objects: int,
    sac_policy,
    bcsac_policy,
    seed: int,
) -> dict:
    env = _build_env(variant, max_num_objects)

    sources: dict[str, Callable | None] = {s: None for s in _ACTION_SOURCES}
    if sac_policy is not None:
        sources["sac_trained"] = sac_policy
    if bcsac_policy is not None:
        sources["bcsac_trained"] = bcsac_policy

    results = {}
    key = jax.random.PRNGKey(seed)

    for src_name, policy in sources.items():
        src = "policy" if src_name.endswith("_trained") else src_name
        step_fn = _make_step_fn(env, src, policy=policy)
        rollout = jax.jit(partial(_rollout_chunk, env, step_fn, scenario_length=scenario_length))
        agg_outcomes = {k: [] for k in ("reached", "collision", "offroad", "clean_success", "min_goal_distance_m")}
        metrics_acc = []

        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            sub = jax.tree_util.tree_map(lambda x, s=start, e=end: jnp.asarray(x[s:e]), stacked)
            key, rk = jax.random.split(key)
            final_tr, stacks = rollout(sub, key=rk)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), stacks)
            outcomes = _goal_outcomes(final_tr, stacks, goal_threshold_m)
            for k in agg_outcomes:
                agg_outcomes[k].append(outcomes[k])
            metrics_acc.append(_summarize_rollout(stacks, outcomes))

        outcomes_all = {k: np.concatenate(v) for k, v in agg_outcomes.items()}
        summary = {
            "ep_rew_mean": float(np.mean([m["ep_rew_mean"] for m in metrics_acc])),
            "ep_rew_std": float(np.mean([m["ep_rew_std"] for m in metrics_acc])),
            "step_rew_mean": float(np.mean([m["step_rew_mean"] for m in metrics_acc])),
            "ep_len_mean": float(np.mean([m["ep_len_mean"] for m in metrics_acc])),
            "action_abs_mean": float(np.mean([m["action_abs_mean"] for m in metrics_acc])),
            "action_l2_mean": float(np.mean([m["action_l2_mean"] for m in metrics_acc])),
            "reached": float(outcomes_all["reached"].mean()),
            "collision": float(outcomes_all["collision"].mean()),
            "offroad": float(outcomes_all["offroad"].mean()),
            "clean_success": float(outcomes_all["clean_success"].mean()),
            "median_goal_dist_m": float(np.median(outcomes_all["min_goal_distance_m"])),
        }
        results[src_name] = summary
        print(f"  [{variant:>8}] {src_name:>14}: ep_rew={summary['ep_rew_mean']:+.3f}  "
              f"clean={summary['clean_success']:.3f}  coll={summary['collision']:.3f}  "
              f"act_l2={summary['action_l2_mean']:.3f}")

    # Expert reward decomposition (optional; can fail on some metric paths).
    try:
        key, rk = jax.random.split(key)
        sub = jax.tree_util.tree_map(lambda x: x[: min(chunk_size, num)], stacked)
        decomp = _expert_reward_decomposition(sub, env, scenario_length, rk)
        results["_expert_reward_decomp"] = decomp
        print(f"  [{variant:>8}] expert reward terms: "
              + " ".join(f"{k}={v:+.4f}" for k, v in sorted(decomp.items()) if k.startswith("term_")))
    except Exception as exc:
        print(f"  [{variant:>8}] expert reward decomp skipped: {exc}")
    return results


def main() -> None:
    args = _parse_args()
    print(f"[diagnose_sac] devices={jax.local_device_count()} backend={jax.default_backend()}")

    print("[diagnose_sac] loading failure scenarios...")
    stacked, num = data.load_failure_scenarios(
        args.failure_dir, max_num_objects=args.max_num_objects, limit=args.limit, verbose=True
    )

    # Build one env to load policies (structure must match training).
    ref_env = _build_env("default", args.max_num_objects)
    sac_policy = _load_policy(args.sac_run_dir, ref_env) if args.sac_run_dir else None
    bcsac_policy = _load_policy(args.bcsac_run_dir, ref_env) if args.bcsac_run_dir else None
    if sac_policy:
        print(f"[diagnose_sac] loaded SAC policy from {args.sac_run_dir}")
    if bcsac_policy:
        print(f"[diagnose_sac] loaded BC_SAC policy from {args.bcsac_run_dir}")

    variants = args.env_variants or list(_ENV_VARIANTS)
    all_results = {}
    for variant in variants:
        print(f"\n--- env variant: {variant} ---")
        all_results[variant] = _run_variant(
            variant, stacked, num, args.scenario_length, args.chunk_size,
            args.goal_threshold_m, args.max_num_objects, sac_policy, bcsac_policy, args.seed,
        )

    print("\n==================== SAC DIAGNOSIS SUMMARY ====================")
    cols = ["ep_rew_mean", "clean_success", "collision", "offroad", "action_l2_mean", "ep_len_mean"]
    for variant, vres in all_results.items():
        print(f"\n  [{variant}]")
        header = f"    {'source':>14} | " + " | ".join(f"{c:>12}" for c in cols)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for src, s in vres.items():
            if src.startswith("_"):
                continue
            row = f"    {src:>14} | " + " | ".join(f"{s[c]:>12.3f}" for c in cols)
            print(row)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"num_scenarios": num, "scenario_length": args.scenario_length, "results": all_results}, f, indent=2)
        print(f"\n  results -> {args.out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose SAC vs BC performance on failure cases.")
    p.add_argument("--failure-dir", default="/zfsauton/scratch/mineuih/waymax_rs/failure_samples")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--scenario-length", type=int, default=80)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sac-run-dir", default=None, help="Pure-SAC training run (loads model_final.pkl).")
    p.add_argument("--bcsac-run-dir", default=None, help="BC_SAC training run for comparison.")
    p.add_argument("--env-variant", dest="env_variants", action="append", choices=list(_ENV_VARIANTS),
                   help="Env config to test (repeatable; default: all four).")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    return p.parse_args()


if __name__ == "__main__":
    main()
