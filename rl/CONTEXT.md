# PPO-on-Waymax-failure-cases — working context

Last updated: 2026-07-30. Scratch notes for the RL effort under `rl/`.

> **2026-07-30 update:** aligned the SB3 path to V-Max's **`repro_sac_v2`** —
> the **`lq`** encoder, its observation (5-step history, road-edge roadgraph,
> **path_target**), reward and SAC hyperparameters. See
> **[V-Max parity in SB3](#v-max-parity-in-sb3-2026-07-30)** at the bottom.
> Corrects the 2026-07-29 note, which read the wrong config key and wrongly
> concluded V-Max ran encoder-less.
> ⚠️ **The observation changed shape (103 → 457 → 1967)** — old SB3 checkpoints
> and BC caches are incompatible.
>
> **2026-07-01 update:** pseudo-GT rollout + eval pipeline for V-Max SAC on failure
> cases mapped to ScenarioMax (`repro_sac_v2`). See
> **[Pseudo-GT rollouts (repro_sac_v2)](#pseudo-gt-rollouts-repro_sac_v2--2026-07-01)** —
> the low V-Max accuracy (36%) is mostly a **measurement bug** in
> `evaluate_pseudo_gt.py`, not policy collapse on converted data.
>
> **2026-06-20 update:** added/fixed the **SB3 BC+SAC** pipeline (`rl/train_bc_sac.py`,
> `rl/bc_core.py`, `~/slurm/bc_sac_waymax.sh`). See
> **[SB3 BC+SAC pipeline](#sb3-bcsac-pipeline-rltrain_bc_sacpy--2026-06-20)** below.
>
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
  dims `(num_devices, num_envs, num_episode_per_epoch)`, `include_sdc_paths=True`
  (**2026-06-19 fix**: WOMD 1.3.1 tf_example ships the real curated SDC route in
  `path_samples/*` — 45 paths × 800 pts, with per-path `on_route` flags, matching
  Waymax's `WOD_1_3_1_TRAINING` config. We now load these directly instead of
  regenerating an approximate single-path route at reset with V-Max's heuristic
  `SDCPathWrapper`. The route feeds `progression`/`off_route` rewards, the
  `path_target` obs, and the red-light/route metrics, so the dataset's real paths
  remove a train/eval discrepancy vs the ScenarioMax-style data V-Max was tuned
  on. `WOMD_NUM_SDC_PATHS=45`, `WOMD_NUM_POINTS_PER_SDC_PATH=800` in `data.py`;
  `refine.py`/`evaluate.py` build the env with `sdc_paths_from_data=True`):
  - `load_failure_scenarios()` / `make_failure_generator()` — loads the failures
    via Waymax's **own** `simulator_state_generator` per shard (so objects are
    truncated to `max_num_objects=64` with the SDC preserved — the fast
    `viz.render` loader keeps 128 and mismatches the env). Caches on-GPU, samples
    device batches each step. `scenario_idx` = sequential record position.
  - `failure_shard_indices()` + `build_expert_shard_path()` — failures live in
    just **7 shards (0–6)**. Waymax only understands `name@N` sharded paths, so we
    build a **symlink farm** of the kept shards (7–999 → `…tfrecord@993`) to
    cleanly exclude failures from BC.
  - `make_expert_generator()` — streams the non-failure expert on that `@993`
    path. Builds the Waymax `DatasetConfig` directly (not V-Max's
    `make_data_generator`, which hardcodes ScenarioMax path dims 10×300 and would
    mismatch the WOMD 45×800 path tensors) with `include_sdc_paths=True`.
- `bc_sac_dual.py` — adaptation of V-Max's `bc_sac_trainer.train` that takes
  **two** generators (RL=failures, imitation=expert) and two replay buffers; all
  networks/losses/`pmap` loop reused verbatim. Returns + checkpoints final params.
- `refine.py` — CLI entrypoint. Composes V-Max's Hydra config (`algorithm=bc_sac`)
  so hyperparams/obs/reward/network are V-Max defaults, builds the bicycle-dynamics
  env, the two generators, runs the dual trainer. Persists `run_config.json`
  (full resolved config) for eval. TensorBoard + optional W&B logging.
  - `--bc-source {expert,failures,none}` (default `expert`): `expert` = BC on
    disjoint non-failure shards (failures unseen by BC); `failures` = BC *and* SAC
    both on the 245 failures (classic BC_SAC co-training); `none` = pure SAC, no
    behavior cloning (`bc_sac_dual.train` skips the imitation buffer/steps when
    `imitation_data_generator is None`). `failures`/`none` ignore `--womd-dir`.
    Launchers: `~/slurm/vmax_bcsac.sh` (expert), `~/slurm/vmax_bcsac_bcfail.sh`
    (failures), `~/slurm/vmax_sac.sh` (none / pure SAC).
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
- ✅ The corrupted **expert-replay / BC** reward (BREAKING FINDING) was root-caused
  2026-06-19 to a **per-env SDC-index bug in `expert_step`** (not the bicycle
  inverse) and **fixed**: expert replay now reproduces the human log (collision
  0.42 → **0.00**, SDC drift 22 m → **0.26 m**). NB this only touches BC + the
  diagnostics; **pure SAC** (`vmax_sac.sh`) was never affected (its reward is
  per-env-correct), so its curves are unchanged — its plateau is separate RL
  hardness (see fixes 1 & 2 / `vmax_sac_fix.sh`).

## 🚨 BREAKING FINDING (2026-06-19): the reward signal is corrupted by a sim artifact

A 5M-step **pure-SAC** run (`~/slurm/vmax_sac.sh`) plateaus immediately and the
trained policy is **no better than doing nothing**. Adding BC (either source)
changes the curves *not at all*. Root-caused with a series of diagnostics
(`runs/sac_failures`, 16 failure scenarios):

1. **Control loop is fine.** Obs is healthy (shape `(16,1641)` — that is
   *(num_envs, obs_dim)* under `diagnose_sac.py`'s `num_closest_objects=8` and no
   `goal` block, **not** the 16-agent layout used by `repro_sac_v2`; std 0.70, 21%
   zeros, varies per scene); the trained policy ≠ random-init (‖Δaction‖=3.3).
   So it is not a dead-obs / no-gradient wiring bug.
2. **Trained policy ≈ coast.** Rollout (reached / collision / offroad):
   trained `0.19 / 0.38 / 0.12`, **zero-action `0.25 / 0.38 / 0.25`**, random
   `0.00 / 0.44 / 0.44`. The policy's accel output collapsed to ~0 (mean −0.03,
   std 0.09) — it learned to coast, and coasting is as good as it gets.
3. **Ground-truth human log is clean.** Setting `sim_trajectory := log_trajectory`
   and computing Waymax metrics directly: failures **collision 0.000**, offroad
   0.188; non-failures 0.062 / 0.000.
4. **Re-simulating that same log corrupts it.** The **expert** replaying its own
   inverse-dynamics actions through `InvertibleBicycleModel(normalize_actions)`
   scores failures **collision 0.375, offroad 0.312** — i.e. *perfect imitation
   still "fails" 37%*. SDC drift from the logged path is **bimodal**: ~half the
   scenes track to ≤0.03 m, the rest **diverge catastrophically** (up to 11.5 m),
   and those are the ones that collide/terminate. dt is correct (0.1 s) and the
   SDC log is fully valid (91/91), so it is **not** a timestep/validity bug — the
   bicycle model cannot reproduce raw-WOMD trajectories (even a ~1.4 m/s near-parked
   car drifts 3 m → ill-conditioned steering inverse), and once the SDC deviates it
   hits the **non-reactive log-replayed** agents the human avoided by exact timing.

**Consequence:** there is **no clean, learnable reward**. BC imitates a pipeline
that crashes (so it teaches crashing → no effect); SAC is penalized no matter what
→ it converges to "coast / minimize action" and never beats the zero baseline.
This is the deeper form of the data discrepancy: V-Max is tuned on **ScenarioMax**
data whose trajectories are refit to be kinematically bicycle-consistent; raw WOMD
logs are not, so the dynamics layer can't track them and the log-replay sim turns
that drift into unavoidable collision penalties.

*(Caveat: a `StateDynamics` expert replay was also dirty, hinting the artifact may
be partly a 1-step ego/agent desync in `expert_step`, not solely the bicycle
inverse. The rock-solid result is: ground-truth log clean, any re-sim of it not.)*

### Candidate fixes
1. ✅ **Decouple learning from the artifact (fastest):** drop `overlap`/`offroad`
   from `termination_keys` + downweight those penalties so `progression` + goal
   reward dominate. Doesn't fix the artifact, but makes *something* learnable.
   **Implemented** — see "Fixes 1 & 2 implemented" below.
2. ✅ **Reactive sim agents (IDM)** for non-ego objects instead of pure log replay,
   so they avoid the drifting SDC → removes most ghost collisions.
   **Implemented** — see below.
3. ✅ **Fix tracking at the source (done — this was the real bug):** investigate
   the bicycle inverse / suspected `expert_step` desync, or refit the SDC log.
   The `expert_step` desync hunch was right; it was a **per-env SDC-index bug**,
   not the bicycle inverse. **Fixed** — see "RESOLUTION" below.

### Fixes 1 & 2 — now **defaults** (2026-06-20)

Policy-rollout ghost collisions (SAC learning from its own rollouts, not BC
expert replay) are addressed by making the **policy-friendly sim** the default in
``refine.py`` / ``evaluate.py``:

- **Reactive IDM** non-ego agents (``env_utils.attach_idm_sim_agents``).
- **Decoupled overlap/offroad**: terminate on ``run_red_light`` only; overlap/offroad
  penalty −0.25 (progression can dominate).

Opt into legacy log-replay + harsh penalties with ``--log-replay-agents``.
Constants live in ``env_utils.POLICY_FRIENDLY_*``.

### Fixes 1 & 2 implemented (2026-06-19, was opt-in)

Both decoupling and reactive sim-agents are now wired through `refine.py`
(training) and mirrored in `evaluate.py` (eval uses the same env so there's no
train/eval distribution shift).

**Fix 1 — reward/termination decoupling.** New `refine.py` flags patch the resolved
V-Max config in place (`_apply_reward_overrides`):
- `--termination-keys [KEY ...]` overrides which metrics end an episode early.
  Default keeps the config value (`offroad`, `overlap`, `run_red_light`); pass
  e.g. `--termination-keys run_red_light` to stop terminating on the
  artifact-driven keys, or `--termination-keys` (no args) to disable early
  termination entirely.
- `--r-overlap / --r-offroad / --r-off-route / --r-red-light / --r-progression`
  override individual `reward_config` weights (default `None` = keep config:
  overlap −1, offroad −1, red_light −1, off_route −0.6, progression +0.2).

**Fix 2 — reactive IDM non-ego agents.** `rl/vmax_rl/env_utils.py::attach_idm_sim_agents`
mutates the underlying Waymax `PlanningAgentEnvironment` in place to set
`sim_agent_actors = [IDMRoutePolicy(is_controlled_func=~is_sdc)]` (Waymax already
supports `sim_agent_actors`; V-Max just never wires it up, leaving non-ego agents
on rigid log replay). It walks the V-Max wrapper chain (`AutoReset → Vmap → Brax →
reward → obs → base`) down to the base env and sets the actors before the first
`reset` (reset reads `_sim_agent_actors` live and inits per-episode actor state).
Gated by `--reactive-agents` (+ `--idm-desired-vel`, default 30 m/s).
`run_config.json` records `reactive_agents` / `idm_desired_vel` so `evaluate.py`
re-attaches IDM automatically.

**Launcher:** `~/slurm/vmax_sac_fix.sh` — pure SAC on the 245 failures
(`--bc-source none`) with both fixes on: `--reactive-agents --termination-keys
run_red_light --r-overlap -0.25 --r-offroad -0.25`. Apples-to-apples vs
`vmax_sac.sh`; if SAC now climbs above the coast baseline, the artifact was the
blocker. (Smoke-validated: config overrides apply, IDM attaches, env resets/steps.)

## ✅ RESOLUTION (2026-06-19): the artifact was a per-env SDC-index bug in `expert_step`

The "bicycle can't track raw WOMD" diagnosis in finding #4 above was **wrong**.
The bicycle inverse is fine: an *unwrapped* `PlanningAgentEnvironment` expert
replay tracks the SDC log to **<0.05 m** with small, in-range actions (verified
directly). The corruption came from V-Max's `expert_step`
(`V-Max/vmax/agents/pipeline/inference.py`), which selected the SDC action with a
**single global** `operations.get_index(is_sdc)` = `jnp.argmax(is_sdc)` over the
*whole flattened batch*. That returns one scalar object index and applies it to
**every** env. It only works when all batched scenarios store the SDC at the same
object slot — true for the ScenarioMax data V-Max was tuned on, **false for raw
WOMD** (our failures have the SDC at indices `[1,1,1,4,3,5,6,…]`). So for every
env whose SDC wasn't at the global-argmax slot, `expert_step` applied **another
object's** inverse action to the ego → the ego drove a wrong trajectory, drifted
off the log, and hit the frozen log-replayed agents = the ghost collisions. The
tell was bimodal *by SDC index*: scenes whose SDC sat at the global-argmax slot
tracked perfectly; all others diverged (the "bimodal drift" finding #4
mis-attributed to the dynamics; the `StateDynamics`-replay "caveat" was the same
buggy path).

This polluted both halves of the loop: **BC** (`bc_sac_dual.py:110` uses
`inference.expert_step`) cloned wrong-object actions, and every re-sim diagnostic
went through the same buggy path (hence "any re-sim of the log is dirty").

**The fix** (`inference.expert_step`): select the SDC action **per-env** —
`sdc_idx = jnp.argmax(is_sdc, axis=-1)` then
`take_along_axis(actions, sdc_idx[..., None, None], axis=-2)`. Verified with
`rl/vmax_rl/diagnose_replay.py` (closed-loop expert replay, 24 failures, horizon 80):

| variant | collision | offroad | drift_mean | drift_max |
|---|---|---|---|---|
| `log_gt` (sim:=log, reference) | 0.000 | 0.167 | 0.00 m | 0.00 m |
| bicycle expert-replay **before** | 0.417 | 0.542 | 7.35 m | 22.3 m |
| bicycle expert-replay **after** | **0.000** | **0.125** | **0.11 m** | **0.26 m** |

After the fix the bicycle expert replay matches the ground-truth log → **BC now
clones the real (collision-free) expert** instead of wrong-object actions.

**Scope (important):** the bug lived in `expert_step`, which runs *outside* the
`jax.vmap` on the **batched** state. The SAC reward/metrics are computed *inside*
`VmapWrapper` (`brax.py` vmaps `env.step`), i.e. **per-env**, where
`operations.get_index(is_sdc)` sees an unbatched `(O,)` mask and is correct. So:
- **affected:** BC (`bc_sac_dual.py:110` → `expert_step`) and every expert-replay
  / `StateDynamics` diagnostic → this *was* finding #4 / the "caveat".
- **not affected:** **pure SAC** (`--bc-source none`, `vmax_sac.sh`) — it never
  calls `expert_step` and drives the SDC via `policy_step`/`env.step` with the
  correct per-env mask. Its curves are **identical** before/after the fix.

Consequence: the pure-SAC plateau (findings #1–#2) is a **separate RL-hardness
problem**, not this bug — fixes 1 & 2 (decoupling + IDM, `vmax_sac_fix.sh`) still
target *that*. The fix's payoff shows up in the **BC** runs (`vmax_bcsac.sh`,
`vmax_bcsac_bcfail.sh`).

### New code (`rl/vmax_rl/`)
- `diagnose_replay.py` — the diagnostic above (`log_gt` vs `bicycle_raw` vs
  `bicycle_refit`); reproduces the artifact and verifies the fix. Run on a GPU:
  `python -m rl.vmax_rl.diagnose_replay --limit 24`.
- `kinematic_refit.py` — **optional/defensive** ScenarioMax-style refit that
  rewrites the SDC log's `yaw`/velocity to be exactly bicycle-consistent (keeps
  `x,y,z,valid` and the goal unchanged). After the index fix it only shaves the
  residual sub-metre drift (the 22→0.26 m collapse is the index fix; 0.26→0.18 m
  is the refit) and well-conditions the low-speed inverse. Wired as
  `--refit-sdc-log` on `refine.py`/`evaluate.py` and
  `load_failure_scenarios(refit_sdc_log=...)`, **off by default**.

## SB3 BC+SAC pipeline (`rl/train_bc_sac.py`) — 2026-06-20

Third RL path: **Stable-Baselines3 SAC** with optional **offline BC warm-up** on
expert WOMD shards (excluding failure shards), then optional SAC fine-tuning on a
**50/50 mix** of expert stream + harvested failure JSONs. JAX/Waymax sim on GPU;
SB3 MLP on CPU (`--device cpu` in slurm — same torch+JAX coexistence rule as PPO).

### Code layout

- `train_bc_sac.py` — single entrypoint; modes via `--total-timesteps` and
  `--actor-freeze-timesteps`.
- `bc_core.py` — expert shard symlink farm, BC dataset collection, `train_bc_actor`,
  `FrozenActorSACClass` (staged critic warm-up), `MixedScenarioSource`,
  `run_bc_sanity_check`, eval helpers.
- `bc_cli.py` — shared CLI flags (`--bc-scenarios`, `--bc-epochs`, SAC knobs).
- `scenario_source.py` — `CachedScenarioSource`, `_ensure_unbatched_scenario`
  (Waymax `batch_dims=(1,)` still yields `is_sdc` shape `[1, N]`).
- `sac_callbacks.py` — rolling train metrics, periodic failure eval,
  `evaluate_expert_replay` (closed-loop bicycle upper bound).
- `sanity_check_sac.py` — preflight checks for failure loading / goal fidelity.

### Slurm launchers (`~/slurm/`)

| Script | `MODE` | Purpose |
|---|---|---|
| `bc_sac_waymax.sh` | `unstaged` (default) | BC + full SAC immediately |
| `bc_sac_staged_waymax.sh` | `staged` | BC + critic-only window, then full SAC |
| `bc_only_waymax.sh` | `bc_only` | BC only + held-out expert eval |

Defaults: `SAVE_ROOT=/zfsauton/scratch/yixiz/waymax_rs/runs`, `BC_SANITY=1`
(replays expert + BC policy on **exact BC train scenarios**, log-replay agents).
Runs: `bc_expert`, `bc_sac_staged`, `bc_sac_unstaged`.

### Training phases (staged — intended schedule)

All runs share **Phase 1** when `bc_epochs > 0` and `bc_scenarios > 0`:

| Phase | What trains | “Steps” | Training data | Other agents |
|---|---|---|---|---|
| **1. BC** (offline) | Actor only (supervised MSE on `tanh(μ)`) | `bc_epochs × ceil(T / bc_batch_size)` grad steps; **not** env steps | `bc_scenarios` draws from expert `@993` pool, `bc_seed=0`, closed-loop bicycle expert labels; **log-replay** | — |
| **2. Critic warm-up** (staged only) | Critic only; actor **frozen** at BC weights | Env steps `0 … actor_freeze−1` (default **25k**) | On-policy rollouts: `MixedScenarioSource` — `expert_mix_prob=0.5`, expert stream `seed+1`, failures from JSON dir | **IDM** (`--reactive-agents`) |
| **3. Full SAC** | Actor + critic | Env steps `actor_freeze … total−1` (default **475k** more) | Same mixed env as phase 2 | IDM |

**BC hyperparams (slurm defaults):** `BC_SCENARIOS=256`, `BC_EPOCHS=20`,
`BC_BATCH_SIZE=256`, `BC_LR=3e-4`, `max_episode_steps=80`.

**SAC hyperparams (slurm defaults):** `TOTAL_TIMESTEPS=500000`,
`ACTOR_FREEZE_TIMESTEPS=25000` (staged), `EXPERT_MIX_PROB=0.5`,
`LEARNING_STARTS=0` (staged, forced), `BUFFER_SIZE=100k`, `train_freq=1`,
`gradient_steps=1`, `EVAL_FREQ=10000`.

**Unstaged** (`actor_freeze=0`, plain `SAC` class): phases 2+3 merge — after BC,
`learning_starts=10000` random explore, then joint actor+critic (no critic-only window).

### Eval data (do not confuse with training)

| Eval | When | Scenarios | Agents |
|---|---|---|---|
| **BC sanity** (`bc_sanity_summary.json`) | After BC, before SAC | **Exact BC train scenarios** | Log-replay (matches BC collection) |
| **Periodic in-training** (`eval/*` in W&B) | Every `eval_freq` env steps | **Failure JSON set only** (sequential) | IDM |
| **Post-hoc `eval_sac`** | After full training | Failure dir | IDM |
| **BC-only held-out expert** | `total_timesteps=0` | New expert draw `bc_seed+10_000`, `eval_expert_scenarios=64` | IDM |

BC sanity answers “does BC clone the expert on the scenes it trained on?”
Periodic `eval/*` answers “how is the policy doing on failures?” — not comparable
to BC sanity without matching agents and scenario set.

### Fixes / debugging (2026-06-20)

1. **Batched scenario crash** — `_ensure_unbatched_scenario()` strips Waymax batch
   dim before `_compute_goal_xy` / env reset (`is_sdc` `[1,128]` → `[128]`).
2. **BC W&B logging** — `wandb.init` before BC; logs `bc/loss` per epoch
   (`bc/epoch` x-axis), sanity metrics, held-out eval in bc-only mode.
3. **BC sanity check** — `--eval-on-bc-scenarios` / `BC_SANITY=1`; expert replay
   upper bound + BC policy on cached train scenarios.
4. **`learning_starts` bug (staged)** — slurm used global default `10000` before
   staged override, so critic training did not start until env step **10,001**
   even though actor freeze is **25,000**. Logs from `bc_sac_staged_freeze25000`:
   `[bc_core] actor is frozen at env step 10001` → `trainable at env step 25000`.
   **Fixed:** staged mode forces `learning_starts=0` in both slurm and
   `_adjust_staged_sac_args()`.
5. **W&B `train/actor_frozen` chart** — metric only logged when `train()` runs;
   x-axis “Step” can mislead before critic starts. Use `train/env_timesteps` +
   `train/actor_freeze_timesteps` (added) to verify freeze window. Actor weights
   are constant from step 0 through 24,999; eval metrics can still move during
   freeze because eval uses **IDM failures**, not the BC log-replay setup.

### Status

- ✅ End-to-end smoke on `bc_sac_staged` (BC + short SAC).
- ✅ BC sanity + batched-scenario fix validated.
- ✅ Full staged run `bc_sac_staged_freeze25000` reached 20k+ env steps; freeze
  confirmed until step 25k in logs (not ~1.3k as a misread W&B x-axis suggested).
- ⏳ Full 500k staged run + CLEAN success vs PPO baseline (0.19) / V-Max BC_SAC.

### Open items (SB3 path)

- Re-run staged SAC with `learning_starts=0` fix so critic trains from step 0.
- Compare BC sanity (log-replay) vs periodic eval (IDM failures) before tuning SAC.
- Optional: `--bc-sanity-reactive-agents` / `--no-reactive-agents` on held-out eval
  for apples-to-apples with BC collection.

## Open items

- **Re-run the BC runs now that `expert_step` is fixed** (`vmax_bcsac.sh`,
  `vmax_bcsac_bcfail.sh`) — BC clones the real expert now; record CLEAN success vs
  the 1M-step PPO baseline (0.19). Pure `vmax_sac.sh` curves are unchanged by the
  fix (expected); for the SAC plateau use `vmax_sac_fix.sh` (fixes 1 & 2).
- Connect the actual **diffusion fine-tune** stage to consume
  `diffusion_states.pkl` (feed through `data.preprocess` + `train_diffusion`).
  Currently we only *produce* that data.
- Run the full `~/slurm/vmax_bcsac.sh` (5M steps) and record CLEAN success vs the
  1M-step PPO baseline (0.19) — *now meaningful: the reward artifact is fixed.*

---

# Pseudo-GT rollouts (repro_sac_v2) — 2026-07-01

Roll a trained **V-Max SAC** checkpoint on the 245 failure cases, but through
**ScenarioMax-mapped scenes** (same layout the policy was trained on), and export
failure-compatible `*.trajectory.npz` for diffusion / VLA eval.

## Pipeline

| Step | Script / launcher | Output |
|---|---|---|
| Trace failures → ScenarioMax | `rl/trace_failure_to_womd.py` (once) | `/zfsauton/scratch/yixiz/failure_womd_trace.json` |
| Roll policy | `~/slurm/waymax/pseudo_gt_rollout.sh` | `*.trajectory.npz` + `*.pseudo_gt.json` under `PSEUDO_GT_ROOT/NAME_RUN` |
| Score rollouts | `~/slurm/waymax/pseudo_gt_eval.sh` | `eval_summary.csv` + `eval_summary.json` |
| Video viz | `python -m rl.visualize_scenario … --mode video` | MP4 under `pseudo_gt_videos/repro_sac_v2/` |

Shared settings: `~/slurm/waymax/_pseudo_gt_common.sh`
(`NAME_RUN=repro_sac_v2`, checkpoint from
`/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2/model/`).

**Note:** `pseudo_gt_eval.sh` only **scores** saved trajectories (CPU, ~1–2 min/scene).
Rollouts come from `pseudo_gt_rollout.sh` (GPU).

## Data / code

- **Rollout dir:** `/zfsauton/scratch/yixiz/pseudo_gt/repro_sac_v2/` (217 trajectories
  as of 2026-07-01; trace has 245 failures but only mapped hits are rolled).
- **Training run:** `repro_sac_v2` on native ScenarioMax
  (`path_dataset=…/ScenarioMaxWaymo/training.tfrecord`); eval accuracy on
  `ScenarioMaxWaymoValid` ≈ **97%** throughout training.
- **Code:** `rl/rollout_pseudo_gt.py`, `rl/evaluate_pseudo_gt.py`,
  `rl/visualize_scenario.py` (WOMD-map video overlay; see below).

Each `*.pseudo_gt.json` stores `failure_json`, `scenariomax_tfrecord`,
`scenariomax_record_index`, `start_timestep`.

## Eval results (217 scenarios, repro_sac_v2)

| Metric | Rate | Scene / criterion |
|---|---|---|
| V-Max accuracy (`vmax_accuracy_rate`) | **0.364** | ScenarioMax; `offroad`/`overlap`/`run_red_light` |
| Failure success (`failure_success_rate`) | **0.677** | WOMD failure tfexample; goal + no overlap/offroad/TL before goal |
| Failure goal reached | 0.696 | WOMD |
| Original VLA success (input set) | 0.000 | these are all prior failures |
| Still failure by VLA standard | 0.323 | |

At first glance this looks like the policy (**97%** on native ScenarioMaxValid)
**collapsed to 36%** on converted failure ScenarioMax. That interpretation is
**mostly wrong** — see root cause below.

## Root cause: broken V-Max accuracy in `evaluate_pseudo_gt.py`

Decomposing the 217-scenario CSV:

| Failure mode | Share of vmax failures |
|---|---|
| `vmax_offroad > 0` | **64%** (138/217) |
| `vmax_overlap > 0` | 0% |
| `vmax_run_red_light > 0` | 0% |

All 88 cases with `failure_success=True` but `vmax_accuracy=0` fail **only** on
offroad. Of 138 vmax failures, **128** are offroad on ScenarioMax but **not**
offroad on WOMD for the **same saved trajectory**.

**Bug:** `_vmax_accuracy()` splices the policy trajectory into
`log_trajectory` via `apply_ego_replacements_to_expanded_state`, then checks
metrics pointwise by setting `state.timestep = t`. But Waymax's
`OffroadMetric` reads **`sim_trajectory`**, not `log_trajectory` (see
`waymax/metrics/roadgraph.py`). `sim_trajectory` is never updated — it still
holds the original expert positions. The check therefore scores **expert log
positions against the ScenarioMax roadgraph**, not the policy rollout.

Verified on scenario 054: pointwise V-Max offroad fires at `t=10`, but a proper
rollout replay of the spliced log reports offroad **False** on both ScenarioMax
and WOMD.

**Consequence:** `vmax_accuracy_rate = 0.364` in `eval_summary.json` **overstates
failure** and must not be compared to training eval accuracy (~97%). Use
rollout-based metric replay (as in `simulation/evaluation_utils.py`) or splice
into `sim_trajectory` before calling `wrapper.metrics`.

**Trustworthy numbers for pseudo-GT quality:**

- **`failure_success_rate = 0.677`** — same criterion as the original VLA
  failure pipeline, evaluated on WOMD geometry with the saved trajectory.
- **Real remaining gap:** ~32% still fail VLA standard (mostly missed goal: 70
  of 70 failures have `failure_goal_reached=False`; overlap/offroad each ~4–14%
  within that subset). This is expected — inputs are cherry-picked VLA failures.

## ScenarioMax vs WOMD geometry (secondary effect)

Even with a fixed eval, ScenarioMax and WOMD tfexample for the **same**
`scenario/id` differ:

- Path layout: **10×300** (ScenarioMax / V-Max training) vs **45×800** (WOMD 1.3.1).
- Roadgraph point counts differ (~2858 vs ~3637 valid points on spot checks).
- Sparse ScenarioMax road edges can flag offroad where the denser WOMD map does not.

For quick video preview, render on **WOMD** (fast, small `scenario_idx`):

```bash
python -m rl.visualize_scenario \
  --failure-json /zfsauton/scratch/mineuih/waymax_rs/failure_samples/…scenario_052.json \
  --trajectory-npz /zfsauton/scratch/yixiz/pseudo_gt/repro_sac_v2/…scenario_052.trajectory.npz \
  --mode video --out ./pseudo_gt_videos/repro_sac_v2/pseudo_gt_052
```

Use `--scenariomax --trace-json /zfsauton/scratch/yixiz/failure_womd_trace.json`
only when you need the exact training-map scene (slow for large record indices
unless `rl/tfrecord_fast build` index exists).

## Distribution shift (real, but smaller than it looked)

The policy **was** rolled live on ScenarioMax-mapped failure scenes during
`rollout_pseudo_gt.py`. Residual hardness vs native training comes from:

1. **Selection bias** — all 217 inputs are prior VLA failures (0% original success).
2. **Goal miss** — 19 scenarios pass V-Max-style rollout replay but miss the 2 m
   goal on WOMD (`vmax_accuracy=1`, `failure_goal_reached=False`).
3. **Genuine overlap/offroad** — ~4% overlap, ~5% offroad on WOMD (not zero, but
   far below the inflated 64% ScenarioMax pointwise offroad).

## Open items

- **Fix `_vmax_accuracy`** in `rl/evaluate_pseudo_gt.py`: replay spliced trajectory
  through `rollout_predicted_trajectories_with_metrics` (or update `sim_trajectory`)
  before aggregating termination keys; re-run eval on `repro_sac_v2`.
- Recompute corrected ScenarioMax clean rate and compare to training ~97%.
- Optional: batch video script over all 217 `*.trajectory.npz` (~4–7 h CPU).

---

# V-Max parity in SB3 (2026-07-30)

Goal: **keep V-Max's performance, drop V-Max and ScenarioMax.** The V-Max path
requires converting WOMD → ScenarioMax before the policy can be used, which is
the thing we want to stop doing. So V-Max's encoder, observation, reward and SAC
hyperparameters were ported into the native-WOMD SB3 path.

## ⚠️ Read the right config key

A V-Max hydra config has **two** encoder nodes:

| key | value in every run | used? |
|---|---|---|
| `algorithm.network.encoder.type` | `none` | **no** — decoy |
| `network.encoder.type` | `lq` | **yes** (`train_utils.py:262` reads `config["network"]["encoder"]["type"]`) |

An earlier version of this section claimed V-Max ran encoder-less. That was
wrong: it read `algorithm.network.encoder`. **All 25 runs** in the live tree
`/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/` — including `repro_sac_v2`, the
~97% one — used `network.encoder.type: lq`.

Also note the stale copies under `V-Max/runs/` are a *different, older* set (they
do show `none`). The live pipeline writes to `vmax_repro/`
(`name_exp=/zfsauton/scratch/yixiz/waymax_rs/vmax_repro`).

## The reference: `repro_sac_v2`

```
network/encoder=lq   depth 4, dk 64, num_latents 16, ff_mult 2,
                     latent/cross heads 2 x 16, tie_layer_weights: true,
                     embedding_layer_sizes [256,256] relu
observation_config   obs_past_num_steps 5
                     objects: waypoints,velocity,yaw,size,valid; num_closest 16
                     roadgraphs: waypoints,direction,valid; element_types [road_edge]
                                 top_k 200, interval 2, max_meters 70,
                                 meters_box {front 70, back 5, left 20, right 20}
                     traffic_lights: waypoints,state,valid; num_closest 5
                     path_target: waypoints, num_points 3, points_gap 12
termination_keys     [offroad, overlap, run_red_light]
reward_config        overlap -1, offroad -1, red_light -1, off_route -0.2, progression 0.2
algorithm(SAC)       lr 1e-4, buffer 1e6, learning_start 50k, batch 64,
                     grad_updates_per_step 4, tau 5e-3, alpha 0.2, discount 0.99
                     policy & value layer_sizes [256,64,32]
num_envs 16, scenario_length 80, max_num_objects 64, total_timesteps 25M
```

## Why the first SB3 run scored 0.02

The 2026-07-29 port reproduced **wayformer** (no run used it) on a much thinner
observation. Diagnosis of that run (11.5k steps, clean_success 0.02,
offroad 0.45, `mean_max_lateral_m` 13.3):

1. **The reward scored a route the policy could not see.** `--route-reward
   --r-lateral-penalty 0.15` charges `-0.15 x lateral` *every step, unbounded*,
   but there was no route block in the observation and `include_sdc_paths` was
   never enabled. `ep_rew_mean -78.2 / ep_len 78.1 = -1.00/step`, which a mean
   lateral of ~6.7 m accounts for entirely — **the return was essentially just
   the integrated lateral penalty.** Unlearnable at any training length.
2. **The drivable boundary was invisible.** 32 nearest points of *all* types in a
   50 m radius; lane centrelines vastly outnumber road edges, and offroad is
   scored against edges. V-Max keeps **road edges only**, top-k **200**.
3. **No motion history** (single frame vs `obs_past_num_steps: 5`).
4. **Violations never ended the episode**, so they were integrated for ~78 steps.

## What changed

| File | Change |
|---|---|
| `rl/obs_layout.py` | Rewritten: blocks gained a **time dimension**; constants now mirror repro_sac_v2's `observation_config`. |
| `rl/waymax_env.py` | `_compute_observation` rewritten (5-step history, road-edge-only roadgraph in a front-biased box, TL history, **`path_target`**). `RewardConfig` gained `off_route_threshold_m` / `off_route_penalty`. |
| `rl/encoders.py` | Added **`LQFeaturesExtractor`** (the one V-Max used); reworked `WayformerFeaturesExtractor` onto shared primitives. `--encoder` now defaults to `lq`. |
| `rl/train_sac.py` | `include_sdc_paths=True`; hyperparameter defaults → V-Max's; `--off-route-threshold-m` / `--r-off-route`. |
| `~/slurm/waymax/sac_waymax.sh` | `REWARD_PRESET=vmax` (default) + V-Max hyperparameters; `ENCODER=lq` default. |

### Observation: 457 → **1967** dims, 314 tokens

| block | count | steps | feat | tokens | width |
|---|---|---|---|---|---|
| sdc | 1 | 5 | 7 | 5 | 40 |
| agents | 16 | 5 | 7 | 80 | 640 |
| roadgraph | 200 | 1 | 4 | 200 | 1000 |
| traffic_lights | 5 | 5 | 10 | 25 | 275 |
| **path_target** | 3 | 1 | 2 | 3 | 6 |
| goal | 1 | 1 | 5 | 1 | 6 |

(V-Max's own vector at this config is 1961; ours adds the 6-wide `goal` block,
which repro_sac_v2 has no
equivalent of — but our task is goal-reaching and the reward references the goal,
so the policy has to see it.)

### LQ vs Wayformer

**LQ is Perceiver-style**: *all* blocks' tokens are concatenated into **one**
sequence and a single bank of 16 latents cross-attends into it, alternating
cross- and self-attention for 4 depths with **tied weights** (ReZero gates stay
per-depth). Wayformer instead attends per block and concatenates the latents.
The latent bank is `dk * ff_mult` = 128 wide, and V-Max feeds the mean latent
straight to the `[256,64,32]` head — so `features_dim` defaults to 128.

## Reward: bounded indicator, not per-metre

`--off-route-threshold-m 3.0 --r-off-route -0.2` reproduces V-Max's `off_route`:
a **bounded** per-step indicator. `--progression-indicator` does the same for
`progression` (+`--r-progress` on any step where route arclength increased, not
per metre advanced) — without it the term scales with speed and, at 1–2 m/step,
pays several times V-Max's rate for driving fast rather than for progressing. The old `--r-lateral-penalty` (per-metre,
unbounded) is still available but is what let the route term swallow the entire
return. `REWARD_PRESET=legacy` restores the old preset for comparison.

`run_red_light` is in V-Max's termination keys but has **no metric wired in the
SB3 env**, so it is neither rewarded nor terminal here — a remaining gap.

## How to run

```bash
ENCODER=lq SMOKE=1 LIMIT=4 bash ~/slurm/waymax/sac_waymax.sh   # smoke first
ENCODER=lq  bash ~/slurm/waymax/sac_waymax.sh                  # V-Max parity
ENCODER=mlp bash ~/slurm/waymax/sac_waymax.sh                  # encoder ablation
```

`--encoder mlp` runs a flat MLP on the *same* observation — the ablation that
separates "richer observation" from "better encoder".

## Status

- ✅ 19 tests in `rl/tests/test_obs_encoder.py` pass (CPU, ~55 s): observation
  width/finiteness/valid-bit invariants, padding of under-full blocks,
  `path_target` populated (and zero-degrading without SDC paths), history varies
  across timesteps, both encoders' shape/NaN/gradient behaviour, LQ weight-tying
  and token count.
- ⏳ **Not yet run on real scenarios / GPU.** No CLEAN-success number for the
  aligned stack yet.
- ⚠️ **Breaking:** obs is 1967-dim. Old checkpoints will not load; BC caches are
  rejected by the v2 `obs_dim` cache key (rebuild, don't force-reuse).
- Known gaps vs V-Max: no `run_red_light` metric; SB3 SAC uses separate feature
  extractors for actor and critic (`share_features_extractor=False`), so the
  encoder runs twice per update — first knob if throughput disappoints.

---

# Two ways in, from both ends (2026-08-01)

Porting V-Max *into* the SB3 path (`rl/train_sac.py`) has not produced a number yet,
so the same gap is now attacked from the other side as well: teach **V-Max** to read
**raw WOMD**, so the ScenarioMax conversion can be dropped without also dropping
V-Max. Both changes are purely additive — every existing repro script and launcher
composes to exactly the same config as before.

## A. Overfit the SB3 stack (`OVERFIT=1` on `~/slurm/waymax/sac_waymax.sh`)

The V-Max-parity port (1967-dim obs, `lq` encoder) has only ever been unit-tested.
Before spending a full run on it, ask the only question that matters: **can it fit
anything at all?** `OVERFIT=1` trains on a handful of failure scenarios with
train == eval. If `eval/clean_success_rate` does not approach 1.0 on 4 scenarios,
the defect is in the env / observation / reward, not the budget.

```bash
OVERFIT=1 SMOKE=1 bash ~/slurm/waymax/sac_waymax.sh   # wiring check first
OVERFIT=1 bash ~/slurm/waymax/sac_waymax.sh           # 4 scenarios, 300k steps
OVERFIT=1 LIMIT=1 bash ~/slurm/waymax/sac_waymax.sh   # one scenario
OVERFIT=1 ENCODER=mlp bash ~/slurm/waymax/sac_waymax.sh
```

Deviations from the parity defaults, all for the fit test: `learning_starts` 50k→1k,
`lr` 1e-4→3e-4, `ent_coef` 0.2→`auto`, `n_envs` 16→4. The scenario count is in the
save dir (`..._n4`) so n=1 and n=4 never resume off each other.

## B. V-Max on raw WOMD (`~/slurm/waymax/vmax_womd_raw_sac.sh`)

`make_data_generator` hardcoded the SDC route tensor at the **ScenarioMax layout
(10 paths x 300 points)**. WOMD 1.3.1 ships its own curated routes at **45 x 800**,
so the Waymax dataloader's reshape failed and the only escape was
`waymo_dataset=true` — which discards the real routes and regenerates an
approximate single path with the `SDCPathWrapper` heuristic. That is a train/eval
discrepancy in `progression`, `off_route`, `path_target` and the red-light metric,
all of which are defined against the route.

New config keys, **both `null` by default = the ScenarioMax layout**:

| key | file |
|---|---|
| `num_sdc_paths` / `num_points_per_sdc_path` | `vmax/config/base_config.yaml` |
| `num_paths` / `num_points_per_path` kwargs | `vmax/simulator/sim_factory.py::make_data_generator` |
| `constants.WOMD_NUM_SDC_PATHS` = 45, `WOMD_NUM_POINTS_PER_SDC_PATH` = 800 | `vmax/simulator/constants.py` |
| threaded to train / eval / eval_failures generators + the banker | `train.py`, `train_utils.py`, `rl/bank.py` |
| `--num_sdc_paths` / `--num_points_per_sdc_path` | `vmax/scripts/evaluate/evaluate.py` |

`waymo_dataset=false` stays set, so the real routes are used rather than regenerated.

```bash
sbatch ~/slurm/waymax/vmax_womd_raw_check.sbatch     # CPU: does the data load?
SMOKE=1 bash ~/slurm/waymax/vmax_womd_raw_sac.sh     # 10 iters end to end
bash ~/slurm/waymax/vmax_womd_raw_sac.sh             # 25M steps
```

⚠️ Raw WOMD does **not** put the SDC at a fixed object slot (ScenarioMax does).
Anything resolving it with a single global argmax over the batch is wrong on this
data — that was the `expert_step` bug (see the 2026-06-19 RESOLUTION above), so
re-read that section before adding any batched SDC lookup on this path.

## Status
- ✅ Hydra composes; `env_config` carries 45/800; `make_data_generator` signature verified.
- ⏳ Not yet run: the CPU load check, the WOMD smoke, the SB3 overfit.
