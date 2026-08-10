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
`test_gate2b.py` (RL-actor, 3/3) — both run fast on CPU with a tiny synthetic model.

## Phase 3 — training (DPPO first)

`rl/diffusion_ft/train_dppo.py` (`DiffusionRLActor` + `ValueCritic` + GAE + env):
collect a batched episode, credit only the executed prefix, GAE with a state-only
critic on stopped-grad scene features, clipped-PPO on the trainable denoiser,
optional `--expert_weight` (DPPO + expert anchor). Unified CLI: `rl.diffusion_ft.train`.

```bash
# short GPU smoke first (shakes out integration shapes)
DFT_ITERS=2 DFT_INDICES=0,1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# overfit test (8 scenes)
bash rl/diffusion_ft/slurm/submit_train_dppo.sh
# DPPO + expert anchor
DFT_EXPERT_WEIGHT=0.1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
```

Logs/checkpoints under `--out_dir`; per-iter metrics in `train_log.jsonl`.
Planned next behind the same CLI (`--method`): `filtered`, `drwr`, `dawr`, `preference`.
