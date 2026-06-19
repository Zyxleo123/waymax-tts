# PPO-on-Waymax-failure-cases — working context

Last updated: 2026-06-19. Scratch notes for the RL effort under `rl/`.

> **2026-06-19 update:** added a second, working RL path — the **V-Max BC_SAC**
> integration under `rl/vmax_rl/` (JAX-native, GPU end-to-end). The SB3 PPO notes
> below are still valid history; the new pipeline is documented in
> **[V-Max BC_SAC integration](#v-max-bc_sac-integration-2026-06-19)** at the
> bottom and is now the recommended approach.

## What we're doing

Training a PPO policy (Stable-Baselines3) to control the ego (SDC) in Waymax
scenarios, specifically on **failure cases** harvested from a goal-reaching /
reward-search pipeline. The longer-term goal is to get a policy that reaches the
logged goal *cleanly* (no collision, no offroad) so its rollouts can be distilled
back into the diffusion planner (`train_diffusion/`). Right now we are debugging
*why the policy performs poorly* (lots of collisions / offroad).

## Environment / how to run

- Conda env: **`waymax`** at `/zfsauton/scratch/yixiz/miniconda3/envs/waymax`
  (has `stable_baselines3 2.9.0`, `gymnasium 1.3.0`, `torch 2.11.0`, `jax 0.10`).
- The login node (`rhea`) has **no GPU** — JAX fails with `cuInit 303` there.
  Always run through **`srun --gres=gpu:1 --partition=general ...`**.
- Policy runs on **CPU** (`--device cpu`); JAX/Waymax sim uses the GPU. Forcing
  the SB3 MLP policy onto CUDA (`--device cuda`) caused a native `SIGFPE`
  (torch+JAX both grabbing the same GPU) — do not do that.
- Slurm launcher: **`~/slurm/ppo_waymax.sh`** (LR sweep for delta + per-run eval).

## Data directories

- **Failure cases** (per-scenario result files):
  `/zfsauton/scratch/mineuih/waymax_rs/failure_samples`
  - 245 scenarios. Each scenario `<shard>.scenario_<NNN>` has 4 files:
    - `*.json` — result record (the one `ScenarioSource.from_failure_dir` reads)
    - `*.instructions.json` — language/instruction annotation
    - `*.mp4` — rollout video
    - `*.trajectory.npz` — saved trajectory arrays
  - Result JSON keys: `ego_idx, final_goal_distance_m, goal_reached,
    goal_timestep, goal_xy, min_goal_distance_m, offroad, overlap,
    reached_timestep, scenario_idx, success, tfrecord, tl_violation, video_path`.
  - `success == false` marks a failure (the default filter keeps these). Note a
    case can have `goal_reached == true` but `success == false` because it went
    `offroad`/`overlap` (e.g. scenario 052: reached goal but offroad).
  - `ScenarioSource` reads `tfrecord` + `scenario_idx` from each JSON, then loads
    the actual scene from the TFRecord.

- **Source WOMD TFRecords** (the real scenario data, referenced by the JSONs):
  `/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-NNNNN-of-01000`
  - WOMD 1.3.1 training shards; 128 objects/scene, 91 timesteps, dt = 0.1 s.

- **Run outputs**: `/zfsauton/scratch/yixiz/waymax_rs/runs/`
  - `ppo_failures/ppo_waymax.zip` — fully-trained **bicycle** model (~1M steps).
  - `ppo_failures_delta/lr_*/` — delta sweep dirs (train + `eval_rollouts/`).

## Code layout (`rl/`)

- `scenario_source.py` — `ScenarioSource.from_failure_dir(...)` /
  `from_tfrecord(...)`; caches unbatched scenarios on host. Depends on
  `viz.render._load_scenario_state_batch_fast` (unaffected by the recent merge).
- `waymax_env.py` — `WaymaxGymEnv` (gymnasium env) + `RewardConfig`.
  - Action spaces: `"bicycle"` (`InvertibleBicycleModel`, 2-dim accel/steer,
    kinematically feasible) and `"delta"` (`DeltaLocal`, 3-dim dx/dy/dyaw in ego
    frame, scaled from [-1,1] to physical limits).
  - **DeltaLocal gotcha**: `vel = d / dt` with `dt = 0.1 s`. The model defaults
    `max_dx=max_dy=6`, `max_dyaw=π` allow ~60 m/s and ~180°/step — physically
    absurd. The policy exploits this to *teleport* to the goal.
  - Reward (`step()`): straight-line `progress*(prev_dist - goal_dist)` by
    default; **`route_reward=True`** rewards progress along the expert log path
    minus `lateral_penalty * deviation` (added 2026-06-18). Plus collision
    (`-10`, terminates), offroad (`-5`, optional terminate), goal bonus (`+10`).
  - Goal = ego's last valid **logged** (x, y).
  - Export helpers: `ego_trajectories()` (sim vs log path), `simulated_log_state()`.
- `train_ppo.py` — training entrypoint. Key flags: `--failure-dir`, `--tfrecord`,
  `--action-space {bicycle,delta}`, `--delta-max-dx/dy/dyaw`,
  `--r-progress/-action-penalty/-collision/-offroad/-goal-bonus`,
  `--terminate-on-offroad`, `--route-reward`, `--r-lateral-penalty`.
- `eval_ppo.py` — rolls the policy over scenarios and **saves to disk**:
  per-scenario JSON (trajectory `sim_*`/`log_*`, `actions`, outcome) +
  `summary.json`. Default out: `<model_dir>/eval_rollouts`. Reports **CLEAN
  success** = reached AND no collision AND no offroad (the metric that matters;
  raw `success_rate` is gamed by teleporting).

## Experiments so far (40-scenario eval unless noted; CLEAN = reached & no col & no off)

| config | reached | CLEAN | collision | offroad | notes |
|---|---|---|---|---|---|
| delta OLD (6/6/π, straight-line) | 0.525 | 0.15 | 0.48 | 0.65 | `ep_len≈6.5` → teleports to goal |
| delta NEW (2/0.5/0.2, straight-line) | 0.125 | 0.125 | 0.68 | 0.45 | small limits → can't reach goal |
| delta PEN (6/6/π, −50/−50, term-offroad) | 0.55 | 0.175 | 0.50 | 0.65 | dies in ~3 steps, learns little |
| **bicycle (full 1M-step)** | 0.53 | **0.19** | 0.49 | 0.50 | eval on 100; best so far |
| delta ROUTE (6/6/π, route reward) | 0.425 | 0.05 | 0.60 | 0.65 | route reward didn't help |
| bicycle ROUTE (route reward, lat-pen 0.5) | 0.05 | 0.05 | 0.675 | 0.225 | offroad ↓ but stops reaching goal |

All short runs are **80k timesteps, lr 3e-4** (smoke-scale, not converged).

## Key findings

1. **Raw `success_rate` is misleading** — a delta policy reaches the goal by
   teleporting (≈6 steps of up to 6 m), so high "success" hides ~50% collisions
   and ~65% offroad. Use CLEAN success.
2. The poor performance is **not** specific to delta vs bicycle, action limits,
   or penalty strength — every config lands at **CLEAN ≈ 0.05–0.19** with
   **collision ≈ 0.5–0.68**.
3. **Collision rate (~0.5–0.68) is the dominant, persistent blocker** and did
   not improve even with route-following (which should track the collision-free
   expert path). This is the main thing to investigate next.
4. Route-following reward *did* cut bicycle offroad (0.50 → 0.225), but with the
   current weights it kills forward progress (reached → 0.05). Lateral penalty
   likely too strong / progress weight too weak.

## Open questions / next steps

- **Why so many collisions even when tracking the expert path?** Hypotheses:
  other agents are log-replayed, so if the ego's timing/position differs from the
  log it can overlap agents the human avoided by timing; or failure cases are
  intrinsically conflict-heavy. Inspect saved `eval_rollouts/*.json` (sim vs log
  trajectory + collision timestep) and the `*.mp4` videos to see *where/when*
  collisions happen.
- Re-tune route reward (lower `--r-lateral-penalty`, higher progress) so it
  reaches the goal while staying on path.
- Consider behavior-cloning warm-start from the expert log before PPO.
- Verify the `overlap`/`offroad` metric semantics (is overlap counting the goal
  endpoint near parked cars? is offroad triggered by goal off the drivable area?).
- Longer training (current results are 80k steps; the only converged model is the
  1M-step bicycle).

## Scratch output locations (compute-node /tmp, may be purged)

Comparison runs wrote models/evals to `/tmp/cmp_*` and `/tmp/eval_bike` **on the
gpu node** (not the login node). W&B project for the comparisons:
`ppo-waymax-deltacmp` (runs `delta_cmp_OLD/NEW`, `delta_penalties`,
`route_delta`, `route_bicycle`).

---

# V-Max BC_SAC integration (2026-06-19)

A JAX-native alternative to the SB3 PPO path above, built on **V-Max** (Valeo's
Waymax RL/IL framework, vendored at repo root `V-Max/`). Goal is the same
refinement loop, but the algorithm and sim both run on the GPU and BC warm-up is
first-class.

## The idea (matches the diffusion refinement loop)

1. **BC warm-up** a shared policy on the WOMD **expert** data we *do* have —
   every shard *except* the ones containing failure cases (so the failures stay
   "unseen" for the loop).
2. **SAC RL** that same policy on the **245 failure cases** (reward only, no
   demonstrations) so it learns to solve them.
3. **Export** the policy's clean rollouts (spliced into `log_trajectory`) to
   fine-tune the diffusion planner. *(export is wired; the diffusion fine-tune
   step itself is not yet connected — see open items.)*

BC and SAC are interleaved in one run via V-Max's **BC_SAC** (shared policy net,
`imitation_frequency=8` → 1 BC step per 8 SAC steps). We made it **dual-source**:
BC pulls from the expert stream, SAC from the failure set.

## Code layout (`rl/vmax_rl/`)

- `compat.py` — JAX-0.10 shim. V-Max targets `jax<0.6`; the `waymax` env has
  `jax 0.10`, which **removed `jax.device_put_replicated`**. Shim restores it
  (also patched directly in `V-Max/.../pmap.py` + the 4 algo factories). This was
  the *only* thing blocking V-Max in this env; `jax.pmap` still works.
- `data.py` — the two data sources, both emitting batched `SimulatorState` with
  dims `(num_devices, num_envs, num_episode_per_epoch)`, `include_sdc_paths=False`
  (SDC route generated on `reset` by V-Max's `SDCPathWrapper`, i.e.
  `waymo_dataset=true` mode):
  - `load_failure_scenarios()` / `make_failure_generator()` — loads the failures
    via Waymax's **own** `simulator_state_generator` per shard (so objects are
    truncated to `max_num_objects=64` with the SDC preserved — the fast
    `viz.render` loader keeps 128 and mismatches the env). Caches on-GPU, samples
    device batches each step. `scenario_idx` = sequential record position.
  - `failure_shard_indices()` + `build_expert_shard_path()` — failures live in
    just **7 shards (0–6)**. Waymax only understands `name@N` sharded paths, so we
    build a **symlink farm** of the kept shards (7–999 → `…tfrecord@993`) to
    cleanly exclude failures from BC.
  - `make_expert_generator()` — streams the non-failure expert via V-Max's
    `make_data_generator` on that `@993` path.
- `bc_sac_dual.py` — adaptation of V-Max's `bc_sac_trainer.train` that takes
  **two** generators (RL=failures, imitation=expert) and two replay buffers; all
  networks/losses/`pmap` loop reused verbatim. Returns + checkpoints final params.
- `refine.py` — CLI entrypoint. Composes V-Max's Hydra config (`algorithm=bc_sac`)
  so hyperparams/obs/reward/network are V-Max defaults, builds the bicycle-dynamics
  env, the two generators, runs the dual trainer. Persists `run_config.json`
  (full resolved config) for eval. TensorBoard + optional W&B logging.
- `evaluate.py` — rolls the deterministic policy over the failures, reports
  **CLEAN success** (reached & no collision & no offroad — goal = SDC's last valid
  logged xy, `--goal-threshold-m`), writes `eval/eval.json`, and with
  `--export-diffusion` dumps a spliced `SimulatorState` pickle for diffusion
  fine-tuning (`--clean-only` keeps just solved rollouts). Reuses
  `rl.waymax_env._splice_sdc_sim_into_log` (vmapped).

## How to run

- Single GPU; everything (sim + policy) is JAX on the GPU. Slurm launcher:
  **`~/slurm/vmax_bcsac.sh`** (train `refine.py` → eval/export `evaluate.py`).
- Quick smoke (direct python on a GPU node, *not* `srun` if already in an alloc):
  ```
  python -m rl.vmax_rl.refine --save-dir <dir> --total-timesteps 12000 \
      --num-envs 8 --num-episode-per-epoch 2 --learning-start 400 \
      --buffer-size 40000 --imitation-frequency 4 --limit-failures 32
  python -m rl.vmax_rl.evaluate --run-dir <dir> --clean-only \
      --export-diffusion <dir>/diffusion_states.pkl
  ```
- Defaults are V-Max's `bc_sac.yaml` verbatim (net `[256,64,32]`, `loss=mse`,
  `rl_lr=1e-4`, `imitation_lr=5e-5`, `tau=5e-3`, `alpha=0.2`, `buffer=500k`,
  `grad_updates=4`). **Deviations:** `learning_start` (V-Max 50k; our default 10k,
  slurm 25k — failure set is small) and `total_timesteps` (run length).

## Logging (W&B)

- Opt-in `--wandb` on both scripts (V-Max itself is TensorBoard-only; W&B is
  additive, off by default). Project default `vmax-bcsac`.
- `refine.py` logs training metrics + full config and stores its `run_id` in
  `run_config.json`; `evaluate.py --wandb` **resumes that same run** to log
  `eval/*` summary + a goal-distance histogram.
- Online runs need `wandb login` on the node; otherwise it falls back to offline
  (`wandb sync <dir>/wandb/offline-run-…` later).

## Status (validated end-to-end on gpu17, env `waymax`)

- ✅ Raw WOMD loads + BC_SAC trains natively in V-Max after the JAX shim.
- ✅ Failure loader yields 64-object scenes with SDC preserved (dropped=0).
- ✅ Dual-source BC_SAC trains (BC on shards 7–999, SAC on 245 failures),
  checkpoints `model_final.pkl`.
- ✅ Eval computes CLEAN success; diffusion export (`all` and `--clean-only`) works.
- ✅ W&B online/offline path validated (train + eval into one run).
- ⚠️ Numbers so far are **smoke-scale only** (≤12k steps): CLEAN ≈ 0.06,
  collision ≈ 0.38, offroad ≈ 0.44 — consistent with the SB3 PPO smoke results
  above; a real multi-M-step run is needed for real performance.

## Open items

- Connect the actual **diffusion fine-tune** stage to consume
  `diffusion_states.pkl` (feed through `data.preprocess` + `train_diffusion`).
  Currently we only *produce* that data.
- Run the full `~/slurm/vmax_bcsac.sh` (5M steps) and record CLEAN success vs the
  1M-step PPO baseline (0.19).
- Collision rate is still the likely blocker (same as PPO) — BC warm-up + the
  shared-policy imitation regulariser are the main levers to try (raise/lower
  `imitation_frequency`, longer BC warm-up).
