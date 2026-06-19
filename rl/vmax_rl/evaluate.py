"""Evaluate a trained BC_SAC policy on the failure cases and export rollouts.

Reports **clean success** = reached goal AND no collision (overlap) AND no
offroad -- the metric that actually matters for refinement (raw "reached" can be
gamed). Optionally splices the policy's simulated ego trajectory back into each
scenario's ``log_trajectory`` and dumps the resulting ``SimulatorState`` batch so
the diffusion planner can be fine-tuned on the policy's (hopefully clean)
behaviour instead of the human log.

Example::

    python -m rl.vmax_rl.evaluate \
        --run-dir /zfsauton/scratch/yixiz/waymax_rs/runs/bcsac_failures \
        --failure-dir /zfsauton/scratch/mineuih/waymax_rs/failure_samples \
        --goal-threshold-m 3.0 \
        --export-diffusion /zfsauton/scratch/yixiz/waymax_rs/runs/bcsac_failures/diffusion_states.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from functools import partial

from rl.vmax_rl import compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from waymax import dynamics

from vmax.agents.pipeline import inference
from vmax.simulator import make_env_for_evaluation

from rl.vmax_rl import data
from rl.waymax_env import _splice_sdc_sim_into_log


def _load_params(path: str):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def _resolve_checkpoint(model_dir: str) -> str:
    """Return model_final.pkl, else the latest model_<step>.pkl in ``model_dir``."""
    final = os.path.join(model_dir, "model_final.pkl")
    if os.path.exists(final):
        return final
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pkl")] \
        if os.path.isdir(model_dir) else []

    def _step(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint in {model_dir}. Training likely did not save one yet "
            f"(it may have been killed early -- check for an OOM/`Killed` in the train log)."
        )
    latest = max(ckpts, key=_step)
    print(f"[evaluate] model_final.pkl missing; using latest checkpoint {latest}")
    return os.path.join(model_dir, latest)


def _build_policy(env, meta: dict, params):
    """Reconstruct the deterministic BC_SAC policy from saved params + config."""
    from vmax.agents.learning.hybrid.bc_sac import make_inference_fn, make_networks

    network_config = meta["network_config"]
    alg = meta["algorithm_config"]
    network = make_networks(
        observation_size=env.observation_spec(),
        action_size=env.action_spec().data.shape[0],
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        rl_learning_rate=alg["rl_learning_rate"],
        imitation_learning_rate=alg["imitation_learning_rate"],
        network_config=network_config,
    )
    make_policy = make_inference_fn(network)
    return make_policy(params.policy, deterministic=True)


def _rollout_chunk(env, policy, scenarios, scenario_length: int, key: jax.Array):
    """Roll the deterministic policy over a batch of scenarios.

    Returns the final env transition and per-step metric stacks of shape (T, B).
    """
    batch = jax.tree_util.tree_leaves(scenarios)[0].shape[0]
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, batch)
    transition = env.reset(scenarios, reset_keys)

    step_fn = partial(inference.policy_step, env=env, policy_fn=policy)

    def body(carry, _):
        tr, k = carry
        k, sk = jax.random.split(k)
        next_tr, _ = step_fn(tr, key=sk)
        out = {
            "overlap": next_tr.metrics.get("overlap", jnp.zeros(batch)),
            "offroad": next_tr.metrics.get("offroad", jnp.zeros(batch)),
            "done": next_tr.done,
        }
        return (next_tr, k), out

    (final_tr, _), stacks = jax.lax.scan(body, (transition, key), (), length=scenario_length)
    return final_tr, stacks


def _episode_outcomes(final_state, stacks, goal_threshold_m: float):
    """Compute per-scenario clean-success outcomes on the host."""
    overlap = np.asarray(stacks["overlap"])  # (T, B)
    offroad = np.asarray(stacks["offroad"])  # (T, B)
    done = np.asarray(stacks["done"]).astype(bool)  # (T, B)
    T, B = overlap.shape

    # Mask steps strictly after the episode terminated (first done == end).
    first_done = np.where(done.any(0), done.argmax(0), T - 1)  # (B,)
    step_idx = np.arange(T)[:, None]
    active = step_idx <= first_done[None, :]  # include the terminating step

    collision = ((overlap > 0) & active).any(0)  # (B,)
    went_offroad = ((offroad > 0) & active).any(0)  # (B,)

    # Goal = SDC's last valid logged (x, y). Reached = min sim distance < thresh.
    is_sdc = np.asarray(final_state.state.object_metadata.is_sdc).astype(bool)  # (B, O)
    sdc = is_sdc.argmax(-1)  # (B,)
    sim_xy = np.asarray(final_state.state.sim_trajectory.xy)  # (B, O, T2, 2)
    log_xy = np.asarray(final_state.state.log_trajectory.xy)  # (B, O, T2, 2)
    log_valid = np.asarray(final_state.state.log_trajectory.valid)  # (B, O, T2)
    sim_valid = np.asarray(final_state.state.sim_trajectory.valid)  # (B, O, T2)

    min_goal_dist = np.full(B, np.inf, dtype=np.float32)
    for b in range(B):
        s = sdc[b]
        lv = log_valid[b, s]
        if not lv.any():
            continue
        goal = log_xy[b, s, np.flatnonzero(lv)[-1]]  # last valid logged xy
        sv = sim_valid[b, s]
        if not sv.any():
            continue
        d = np.linalg.norm(sim_xy[b, s][sv] - goal[None], axis=-1)
        min_goal_dist[b] = float(d.min())

    reached = min_goal_dist < goal_threshold_m
    clean = reached & (~collision) & (~went_offroad)
    return {
        "reached": reached,
        "collision": collision,
        "offroad": went_offroad,
        "clean_success": clean,
        "min_goal_distance_m": min_goal_dist,
    }


def _log_eval_to_wandb(args, meta: dict, summary: dict, records: list[dict]) -> None:
    """Log eval summary to wandb, resuming the training run if its id is known."""
    import wandb

    wb = meta.get("wandb") or {}
    run_id = wb.get("run_id")
    wandb.init(
        project=args.wandb_project or wb.get("project") or "vmax-bcsac",
        entity=args.wandb_entity or wb.get("entity"),
        id=run_id,
        resume="allow" if run_id else None,
        name=None if run_id else os.path.basename(os.path.normpath(args.run_dir)) + "-eval",
        mode=args.wandb_mode,
        dir=args.run_dir,
    )
    wandb.log({f"eval/{k}": v for k, v in summary.items()})
    # Distribution of per-scenario min goal distance, for a quick histogram.
    dists = [r["min_goal_distance_m"] for r in records]
    try:
        wandb.log({"eval/min_goal_distance_hist": wandb.Histogram(dists)})
    except Exception:
        pass
    wandb.finish()


def main() -> None:
    args = _parse_args()
    meta = json.load(open(os.path.join(args.run_dir, "run_config.json")))
    failure_dir = args.failure_dir or meta["args"]["failure_dir"]
    max_num_objects = meta["max_num_objects"]
    scenario_length = args.scenario_length or meta.get("scenario_length", 80)

    model_path = args.model or _resolve_checkpoint(os.path.join(args.run_dir, "model"))
    print(f"[evaluate] model: {model_path}")

    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=False,
        observation_type=meta["observation_type"],
        observation_config=meta["observation_config"],
        termination_keys=meta["termination_keys"],
    )

    params = _load_params(model_path)
    policy = _build_policy(env, meta, params)

    print("[evaluate] loading failure scenarios...")
    stacked, num = data.load_failure_scenarios(
        failure_dir, max_num_objects=max_num_objects, limit=args.limit, verbose=True
    )

    key = jax.random.PRNGKey(args.seed)
    chunk = args.chunk_size
    records: list[dict] = []
    spliced_states: list = []

    rollout = jax.jit(partial(_rollout_chunk, env, policy, scenario_length=scenario_length))
    splice_batched = jax.jit(jax.vmap(_splice_sdc_sim_into_log))

    for start in range(0, num, chunk):
        end = min(start + chunk, num)
        sub = jax.tree_util.tree_map(lambda x, s=start, e=end: jnp.asarray(x[s:e]), stacked)
        key, rk = jax.random.split(key)
        final_tr, stacks = rollout(sub, key=rk)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), stacks)
        outcomes = _episode_outcomes(final_tr, stacks, args.goal_threshold_m)
        b = end - start
        for i in range(b):
            records.append({k: float(v[i]) for k, v in outcomes.items()})
        if args.export_diffusion:
            spliced = jax.device_get(splice_batched(final_tr.state))
            if args.clean_only:
                keep = np.flatnonzero(outcomes["clean_success"])
                if keep.size:
                    spliced = jax.tree_util.tree_map(lambda x, k=keep: x[k], spliced)
                    spliced_states.append(spliced)
            else:
                spliced_states.append(spliced)
        print(f"[evaluate] scenarios {start}:{end}  "
              f"clean={outcomes['clean_success'].mean():.3f}  "
              f"collision={outcomes['collision'].mean():.3f}  "
              f"offroad={outcomes['offroad'].mean():.3f}")

    summary = {
        "num_scenarios": num,
        "reached": float(np.mean([r["reached"] for r in records])),
        "clean_success": float(np.mean([r["clean_success"] for r in records])),
        "collision": float(np.mean([r["collision"] for r in records])),
        "offroad": float(np.mean([r["offroad"] for r in records])),
        "median_min_goal_distance_m": float(np.median([r["min_goal_distance_m"] for r in records])),
    }
    out_dir = args.out_dir or os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"summary": summary, "per_scenario": records},
              open(os.path.join(out_dir, "eval.json"), "w"), indent=2)
    print("\n==================== EVAL SUMMARY ====================")
    for k, v in summary.items():
        print(f"  {k:>26}: {v}")
    print(f"  results -> {os.path.join(out_dir, 'eval.json')}")

    if args.wandb:
        _log_eval_to_wandb(args, meta, summary, records)

    if args.export_diffusion:
        if not spliced_states:
            print("  [export] no scenarios to export (clean-only filtered everything out).")
        else:
            # Concatenate spliced batches into one [N, ...] SimulatorState pickle.
            merged = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *spliced_states)
            n_exported = int(jax.tree_util.tree_leaves(merged)[0].shape[0])
            with open(args.export_diffusion, "wb") as f:
                pickle.dump(merged, f)
            tag = "clean-only" if args.clean_only else "all"
            print(f"  diffusion states ({n_exported}, {tag}) -> {args.export_diffusion}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BC_SAC on failure cases + export diffusion rollouts.")
    p.add_argument("--run-dir", required=True, help="Training run dir (contains run_config.json + model/).")
    p.add_argument("--model", default=None, help="Override checkpoint path (default model/model_final.pkl).")
    p.add_argument("--failure-dir", default=None, help="Override failure dir (default from run_config.json).")
    p.add_argument("--scenario-length", type=int, default=None)
    p.add_argument("--goal-threshold-m", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--export-diffusion", default=None,
                   help="If set, dump spliced SimulatorState batch (policy traj in log_trajectory) here.")
    p.add_argument("--clean-only", action="store_true",
                   help="Export only rollouts the policy solved cleanly (reached & no collision & no offroad).")
    # Weights & Biases (opt-in; resumes the training run when its id is in run_config.json)
    p.add_argument("--wandb", action="store_true", help="Log the eval summary to Weights & Biases.")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


if __name__ == "__main__":
    main()
