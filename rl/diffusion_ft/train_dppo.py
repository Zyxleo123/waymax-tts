"""DPPO fine-tuning of the diffusion policy against the simulator reward.

Primary direct-RL method (Phase 3E). Structure:

* **Collection** -- roll a batched episode in :class:`DiffusionWaymaxEnv`. At each
  replan the frozen scene tokenizer produces the condition; the actor
  (:class:`DiffusionRLActor`) samples a trajectory with the frozen base denoiser
  on the early denoising steps and the trainable copy on the last ``k_trainable``
  steps (RL sampling mode), recording those transitions and their old-policy
  log-probs; the sampled trajectory is executed by the env and its scalar reward
  stored. Only the executed prefix (``prefix_len`` model points) receives
  policy-gradient credit.
* **Advantages** -- GAE over env-step rewards with a state-only critic
  (:class:`ValueCritic`) on stopped-gradient scene features.
* **Update** -- clipped-PPO epochs on the trainable denoiser only (frozen old-
  policy snapshot per iteration; early stop on approx-KL), optionally plus one
  expert diffusion minibatch per update (DPPO + expert anchor), and a value MSE
  step.

Run via ``rl.diffusion_ft.train`` (unified CLI) or directly:
    python -m rl.diffusion_ft.train_dppo --indices 0,1,2,3,4,5,6,7 --iters 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as waymax_config

from data.scenario_loader import load_scenario_state_batch_fast
from rl.diffusion_ft.checkpoint import DEFAULT_CHECKPOINT_PATH, load_diffusion_checkpoint
from rl.diffusion_ft.env import DiffusionWaymaxEnv
from rl.diffusion_ft.reward import RewardConfig
from rl.diffusion_ft.rl_actor import DiffusionRLActor
from rl.diffusion_ft.value_net import ValueCritic


# --------------------------------------------------------------------------- #
def load_batch(tfr: str, indices: list[int]):
    n_unique = len(sorted(set(int(i) for i in indices)))
    ds_cfg = dataclasses.replace(
        waymax_config.WOD_1_3_1_TRAINING, path=str(tfr), max_num_objects=None,
        max_num_rg_points=30000, batch_dims=(n_unique,), shuffle_seed=0, include_sdc_paths=False,
    )
    state, idx_map = load_scenario_state_batch_fast(ds_cfg, indices)
    order = jnp.asarray([idx_map[int(i)] for i in indices], dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[order], state)


def compute_gae(rewards, values, done, last_value, gamma, lam):
    """GAE over ``[n_steps, B]`` env-step rewards. Returns (adv, returns).

    ``done`` is ``terminated | truncated``: both a real terminal (goal/crash) and
    the WOMD horizon cut-off end the finite-horizon return, so neither bootstraps
    the critic. This is what makes ``lam=1`` a genuine Monte-Carlo advantage --
    treating the horizon truncation as non-terminal (bootstrapping ``V(s_H)``)
    would leave the critic in the return path for exactly the scenes that end by
    reaching the horizon, i.e. almost all of them, defeating the intent.

    At ``lam=1`` this degenerates to the Monte-Carlo advantage: ``adv`` is the
    observed discounted return minus ``V(s_t)``, and ``returns`` is the observed
    discounted return alone. That matters here because ``returns = adv + values``
    is otherwise self-referential -- with a critic that under-predicts, the GAE
    deltas are large and positive, the targets inflate, and the critic chases a
    target it is itself pushing away (observed: return-to-go 8 -> 38 -> 86 while
    the true episode return stayed ~20). At ``lam=1`` the target is bounded by
    the rewards actually collected, so a bad critic costs variance, not drift.
    """
    n, B = rewards.shape
    adv = np.zeros((n, B), dtype=np.float64)
    lastgae = np.zeros(B, dtype=np.float64)
    for t in reversed(range(n)):
        nextnonterm = 1.0 - done[t]
        nextval = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextval * nextnonterm - values[t]
        lastgae = delta + gamma * lam * nextnonterm * lastgae
        adv[t] = lastgae
    returns = adv + values
    return adv, returns


# --------------------------------------------------------------------------- #
class DPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.ckpt = load_diffusion_checkpoint(args.ckpt)
        self.cond_dim = int(self.ckpt.metadata["cond_dim"])
        self.critic = ValueCritic(self.cond_dim, lr=args.value_lr, seed=args.seed)
        self._rng = jax.random.PRNGKey(args.seed + 1)
        self.env = None
        self.actor = None

    # --------------------------------------------------------------- #
    def _build_for(self, state):
        """Build the env + actor sized to the loaded scenarios' object count.

        The env asserts ``max_num_objects == num_objects`` and WOMD pads objects,
        so the env must adopt the batch's actual count (see Gate 1 notes).
        """
        n_obj = int(jnp.asarray(state.log_trajectory.x).shape[1])
        if self.env is not None and self.env._max_num_objects == n_obj:
            return
        self.env = DiffusionWaymaxEnv(
            self.ckpt, reward_config=RewardConfig(), max_num_objects=n_obj,
            use_ema=False, replan_interval_steps=self.args.replan_interval_steps, seed=self.args.seed,
        )
        self.actor = DiffusionRLActor(
            self.env.policy.diffusion,
            k_trainable=self.args.k_trainable,
            prefix_len=self.args.prefix_len,
            sigma_sample_floor=self.args.sigma_sample_floor,
            sigma_logprob_floor=self.args.sigma_logprob_floor,
            actor_lr=self.args.actor_lr,
            grad_clip_norm=self.args.grad_clip_norm,
            ppo_clip=self.args.ppo_clip,
            pg_scale=(self.args.pg_scale if self.args.pg_scale > 0 else None),
            seed=self.args.seed,
        )

    # --------------------------------------------------------------- #
    def collect(self, state):
        """Roll one batched episode; return the per-step buffer + GAE targets."""
        env, actor, critic = self.env, self.actor, self.critic
        features = env.reset(state)
        B = env.batch_size
        horizon = int(jnp.asarray(env.sim_state.log_trajectory.x).shape[-1])
        n_steps = self.args.max_replans or ((horizon - env.timestep) // self.args.replan_interval_steps)

        buf = {k: [] for k in ("cond1", "cond2", "mask", "trace", "expert_target",
                               "reward", "value", "done", "alive")}
        prev_done = np.zeros(B, dtype=bool)

        for _ in range(int(n_steps)):
            cond1, cond2 = env.compute_condition(features)
            mask = features["inst_valid"]
            self._rng, krng = jax.random.split(self._rng)
            traj_ncT, trace = actor.collect(cond1, cond2, mask, rng=krng)
            world = env.norm_to_world(jnp.transpose(traj_ncT, (0, 2, 1)))
            value = np.asarray(critic.value(cond1))

            buf["cond1"].append(cond1); buf["cond2"].append(cond2); buf["mask"].append(mask)
            buf["trace"].append(trace); buf["expert_target"].append(features["ego_trajectory"])
            buf["value"].append(value); buf["alive"].append(~prev_done)

            out = env.step(trajectory_world_bt5=world)
            buf["reward"].append(np.asarray(out.reward_b))
            # `done` = terminated | truncated: the horizon cut-off ends the
            # finite-horizon return just as a real terminal does, so GAE must not
            # bootstrap past it (see `compute_gae`).
            done = np.asarray(out.terminated_b) | np.asarray(out.truncated_b)
            buf["done"].append(done)
            prev_done = prev_done | done
            features = out.features
            if prev_done.all():
                break

        cond1_last, _ = env.compute_condition(features)
        last_value = np.asarray(critic.value(cond1_last))

        rewards = np.stack(buf["reward"], 0)
        values = np.stack(buf["value"], 0)
        done = np.stack(buf["done"], 0).astype(np.float64)
        adv, returns = compute_gae(rewards, values, done, last_value,
                                   self.args.gamma, self.args.gae_lambda)
        alive = np.stack(buf["alive"], 0)  # [n, B]
        return buf, adv, returns, alive, rewards

    # --------------------------------------------------------------- #
    def _merge(self, buf, adv, returns, alive):
        """Flatten the [n_steps, B] buffer into a single [n_steps*B] batch."""
        n = len(buf["cond1"])
        cat = lambda key: jnp.concatenate([buf[key][t] for t in range(n)], axis=0)
        cond1 = cat("cond1"); cond2 = cat("cond2"); mask = cat("mask")
        expert_target = cat("expert_target")
        # trace: list over steps of {field: [K, B, ...]} -> {field: [K, n*B, ...]}
        keys = buf["trace"][0].keys()
        trace = {k: jnp.concatenate([buf["trace"][t][k] for t in range(n)], axis=1) for k in keys}
        alive_f = jnp.asarray(alive.reshape(-1).astype(np.float32))
        adv_f = jnp.asarray(adv.reshape(-1).astype(np.float32)) * alive_f
        ret_f = jnp.asarray(returns.reshape(-1).astype(np.float32))
        # Normalize advantages over the alive set.
        m = alive.reshape(-1)
        if m.sum() > 1:
            a = np.asarray(adv_f)[m]
            adv_f = jnp.asarray(((np.asarray(adv_f) - a.mean()) / (a.std() + 1e-8)) * m.astype(np.float32))
        old_lp = trace["log_prob"]  # [K, n*B]
        return cond1, cond2, mask, trace, old_lp, adv_f, ret_f, expert_target, alive_f

    # --------------------------------------------------------------- #
    def update(self, merged):
        cond1, cond2, mask, trace, old_lp, adv, ret, expert_target, alive = merged
        args = self.args
        last = {}
        for epoch in range(args.ppo_epochs):
            expert = None
            if args.expert_weight > 0.0:
                self._rng, erng = jax.random.split(self._rng)
                expert = (expert_target, cond1, cond2, mask, erng)
            m = self.actor.ppo_update(
                cond1, cond2, mask, trace, old_lp, adv,
                expert=expert, expert_weight=args.expert_weight,
            )
            last = {k: float(np.asarray(v)) for k, v in m.items()}
            if last["approx_kl"] > args.target_kl:
                last["stopped_epoch"] = epoch
                break
        # Returns from already-terminated envs are meaningless -- mask them out
        # (adv is masked in `_merge`; ret was not).
        vloss = self.critic.update(cond1, ret, weights=alive, epochs=args.value_epochs)
        last["value_loss"] = float(np.asarray(vloss["value_loss"]))
        last["value_loss_final"] = float(np.asarray(vloss["value_loss_final"]))
        return last

    # --------------------------------------------------------------- #
    def train(self):
        args = self.args
        out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train_log.jsonl"
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
        state = load_batch(args.tfrecord, indices)
        self._build_for(state)

        print(f"[dppo] out={out_dir} scenes={indices} k_trainable={args.k_trainable} "
              f"expert_weight={args.expert_weight}")
        for it in range(args.iters):
            t0 = time.time()
            buf, adv, returns, alive, rewards = self.collect(state)
            merged = self._merge(buf, adv, returns, alive)
            metrics = self.update(merged)
            ep_return = float(np.sum(rewards, axis=0).mean())
            row = {
                "iter": it, "mean_episode_return": ep_return,
                "mean_return_to_go": float(returns.mean()),
                "n_steps": len(buf["cond1"]), "time_s": round(time.time() - t0, 2),
                **metrics,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            # Fixed-point formats rounded pg/kl/(ratio-1) to zero and hid the fact
            # that the gradient was nonzero all along -- keep these in scientific
            # notation, and show the pg-vs-expert split that decides what the
            # gradient is actually optimizing.
            print(f"[dppo] it={it:04d} return={ep_return:.3f} "
                  f"pg={metrics.get('pg_loss',0):+.3e} exp={metrics.get('expert_term',0):.3e} "
                  f"gnorm={metrics.get('grad_global_norm',0):.4f} "
                  f"kl={metrics.get('approx_kl',0):+.3e} "
                  f"ratio-1={metrics.get('mean_ratio',1)-1:+.3e} "
                  f"clipf={metrics.get('clip_frac',0):.3f} "
                  f"vloss={metrics.get('value_loss',0):.1f}->{metrics.get('value_loss_final',0):.1f} "
                  f"t={row['time_s']}s")
            if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
                self.save(out_dir / f"ckpt_{it+1:04d}")
        print(f"[dppo] done -> {out_dir}")

    def save(self, path: Path):
        import pickle
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "actor_value.pkl", "wb") as f:
            pickle.dump({
                "actor": jax.tree_util.tree_map(np.asarray, self.actor.state_dict()),
                "critic": jax.tree_util.tree_map(np.asarray, self.critic.state_dict()),
            }, f)


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPPO fine-tuning of the diffusion policy.")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--tfrecord", type=str,
                   default="/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000")
    p.add_argument("--indices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--out_dir", type=str, default="/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    # DPPO / diffusion
    p.add_argument("--k_trainable", type=int, default=10)
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--replan_interval_steps", type=int, default=10)
    p.add_argument("--max_replans", type=int, default=0, help="0 = run to horizon")
    p.add_argument("--sigma_sample_floor", type=float, default=0.05)
    p.add_argument("--sigma_logprob_floor", type=float, default=0.1)
    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=1.0,
                   help="1.0 = Monte-Carlo advantage (critic is a baseline only, not a "
                        "bootstrap); episodes here are ~8 replan steps, so MC is cheap")
    p.add_argument("--ppo_clip", type=float, default=0.1)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--target_kl", type=float, default=0.05)
    p.add_argument("--actor_lr", type=float, default=3e-5)
    p.add_argument("--value_lr", type=float, default=1e-3)
    p.add_argument("--value_epochs", type=int, default=10,
                   help="critic gradient steps per iteration (1 cannot track the GAE target)")
    p.add_argument("--pg_scale", type=float, default=0.0,
                   help="0 = auto (channels*prefix_len), undoing the log-density averaging")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--expert_weight", type=float, default=0.0,
                   help=">0 enables the expert anchor; ~0.001-0.01 (0.1 swamps the PG term)")
    return p


def main():
    args = build_argparser().parse_args()
    if args.max_replans <= 0:
        args.max_replans = None
    DPPOTrainer(args).train()


if __name__ == "__main__":
    main()
