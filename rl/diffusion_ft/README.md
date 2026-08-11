# `rl/diffusion_ft` — diffusion policy simulator-reward fine-tuning

Unified framework for fine-tuning the pretrained diffusion motion policy against
the **simulator** reward (filtered FT, DRWR, DAWR, preference, DPPO, hybrids,
ES + distillation). Deliberately separate from
`train_diffusion/train_diffusion_policy.py`, which only updates
instruction/subgoal/FiLM params. The outer action here is *one sampled
trajectory*.

## Phase 1 (this commit): one trustworthy environment

| Module          | What it is |
| --------------- | ---------- |
| `checkpoint.py` | Loads the architecture from checkpoint **metadata** (`metadata.json`), not config defaults (the dataclass/argparse defaults disagree). Exposes both raw params (collection/likelihoods) and EMA params (evaluation). |
| `reward.py`     | The single batched simulator-reward function. Reproduces `rl/waymax_env.py` step reward exactly, vectorized, pure. Every method consumes this scalar. |
| `env.py`        | `DiffusionWaymaxEnv`: batched, `StateDynamics` execution, real per-scene goal timestep, `log_trajectory` immutable (executed motion lives in `sim_trajectory`), trajectory-as-action (samples internally or accepts an external candidate for best-of-N / ES). |

Shared-code repairs that Phase 1 depends on:
- `data/preprocess.py`: `goal_step_override` now accepts a per-scene **array** (real goal step), not just a scalar.
- `planner/{diffusion,vla}_planner.py` + `simulation/planning_utils.py` + `simulation/runner.py`: removed hard-coded `goal_step_override=90`; threaded the real goal step.
- `simulation/runner.py`: fixed the post-goal metric window (was indexing a `[start_timestep:]`-sliced timeline with an absolute step) and the `video_paths` crash / mis-indexing when visualization is a subset or disabled.
- `scores/scorer.py`: `set_speed` weighted metric was dead code (scored under the name `speed`); fixed the name.

## Gate 1 (run before ANY fine-tuning)

```bash
bash rl/diffusion_ft/slurm/submit_gate1.sh          # from repo root, on rhea
# override scenes:  DFT_INDICES=0,1,2,3,4,5 bash rl/diffusion_ft/slurm/submit_gate1.sh
```

Six gates: expert-replay clean · batched==single · per-step sums to return ·
obs changes on ego deviation · permutation equivariance · frozen-ckpt
reproducible. Log lands in `logs/dft/slurm-dft_gate1-<jobid>.out`.

Interpreter: `/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python` (flax 0.12.7).
Base checkpoint: `.../pretrain_diffusion_without_subgoal_20260616_013642/latest`.

## Phase 2 — diffusion API extensions (Gate 2 PASS)

`model/diffusion/diffusion.py` (inference unchanged): `loss_per_example`,
`sample_with_trace` (frozen early / trainable late; records transitions), and
`transition_log_prob` (v→x0→posterior mean, floored std, averaged over channels +
executed prefix). `rl/diffusion_ft/rl_actor.py` = `DiffusionRLActor`: frozen base
denoiser + cloned trainable copy (structural gradient isolation), clipped-PPO
update with optional expert anchor. Tests: `test_gate2.py` (core, 4/4),
`test_gate2b.py` (RL-actor, 3/3) — tiny synthetic model, CPU only.

**Submit them; do not run them on a login node.** The model is small but the XLA
CPU compile is not, and the process gets killed (observed: exit 137).

```bash
bash rl/diffusion_ft/slurm/submit_gate2b.sh   # cpu partition (lov*), ~5 min
```

## Phase 3 — training (DPPO first)

`rl/diffusion_ft/train_dppo.py` (`DiffusionRLActor` + `ValueCritic` + GAE + env):
collect a batched episode, credit only the executed prefix, GAE with a state-only
critic on stopped-grad scene features, clipped-PPO on the trainable denoiser,
optional `--expert_weight` (DPPO + expert anchor). Unified CLI: `rl.diffusion_ft.train`.

```bash
# short GPU smoke first (shakes out integration shapes)
DFT_ITERS=2 DFT_INDICES=0,1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# overfit test (8 scenes)
DFT_JOBNAME=dppo_pure bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# DPPO + expert anchor -- keep the weight small, see below
DFT_JOBNAME=dppo_anchor DFT_EXPERT_WEIGHT=0.01 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
```

Always set `DFT_JOBNAME` when submitting more than one arm: it names the log
(`%x`) *and* the env file, so two arms cannot silently share one config.

Logs/checkpoints under `--out_dir`; per-iter metrics in `train_log.jsonl`.
Planned next behind the same CLI (`--method`): `filtered`, `drwr`, `dawr`, `preference`.

### Status (2026-08-10): gradient fixed, learning not yet demonstrated

The first DPPO runs printed `pg=-0.0000 kl=0.0000 ratio=1.000` and looked frozen.
They were not — the console format string was rounding. `train_log.jsonl` had
`ratio=0.99999`, `kl=5e-6`, and a nonzero `grad_global_norm` every iteration.
**Always read the JSONL, not the console line, before concluding anything is
zero.** Four separate problems were behind the flat curve:

| Problem | Evidence | Fix |
| --- | --- | --- |
| Expert anchor swamped the PG term ~1000:1 | `loss = -1.4e-05 + 0.1*0.169`; PG was 0.08% of the objective | `--expert_weight` 0.1 → ~0.01 |
| `_reduce_logdensity` averages over `channels*prefix_len`, shrinking the PG gradient by that factor while the expert loss is unscaled | ratio pinned within 1e-5 of 1 after 4 epochs | `--pg_scale` (auto = `channels*prefix_len`) applied to `pg_loss` only; the clip still acts on the un-scaled ratio |
| Critic got one gradient step per iteration and could not track its own target | `value_loss` 409 → 1154 while return-to-go inflated 8 → 38 → 86, true return ~20 | `--value_epochs` 10, and `gae_lambda` 1.0 so `returns` is the observed discounted return instead of `adv + values` |
| Critic trained on returns from already-terminated envs | `adv` was masked by `alive` in `_merge`, `ret` was not | pass the mask; weight-normalized MSE |

After the fix the gradient is unambiguously live — `pg` -1.4e-05 → -4.4e-02,
`grad_global_norm` 0.02–0.09 → 0.11–1.8, `clip_frac` nonzero, `approx_kl`
approaching `target_kl`. Gate 2b re-passed 3/3 (job 27646).

**Episode return then got worse, not better** (16.2 → -5.7 over 13 iterations),
because the advantage the now-live gradient follows is still noise: pre-update
`value_loss` sat at 1000–5158 and jumped between iterations, i.e. critic error
exceeded the whole spread of episode returns. The `gae_lambda=1.0` /
`actor_lr=3e-5` retune is a response to that and is **untested** as of this
writing. If return still does not climb, the remaining suspect is advantage
*variance* (8 scenes × 1 rollout each), and the fix is a group baseline —
N rollouts per scene, advantage normalized within the scene — which removes the
value net from the critical path entirely. More tuning will not help there.

Scenes `0..7` are the argparser default: the first 8 of raw WOMD shard 0,
**not** the failure set. Fine as an overfit target, but do not read a result on
them as a result on the failure set.

### Operational notes

- **GPU class matters.** Five runs (27447/27448/27456/27457/27463) died in
  cuDNN init within seconds; all had landed on v100s (gpu20, gpu23) via
  `legacy`. Every run that reached iteration 0 was on an a6000 under `general`,
  which is the sbatch default. Check `host=` in the first line of the `.out`
  before theorizing about a failure.
- **`--out_dir` is on a quota'd filesystem.** Job 27279 died 27 iterations in on
  `OSError: [Errno 122] Disk quota exceeded` writing `train_log.jsonl`. The
  sbatch now write-probes `DFT_OUT_DIR` and exits non-zero in seconds instead.
