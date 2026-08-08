# es_baseline — hermetic vendored copy of the Diffusion-ES pipeline

## What this is
A byte-for-byte snapshot of the collaborator's Diffusion-ES code
(`/zfsauton2/home/mineuih/waymax_rs`), copied here so we can instrument it and run
the reward-resolution / diversity experiments **without touching any of yixiz's
existing top-level modules** (`scores/`, `data/`, `model/`, `planner/`,
`simulation/`, ...), which are stale relative to mineuih's tree and are imported by
active tools (`rl/bank.py`, `rl/rl_diffusion_source.py`).

## Why a vendored copy instead of a plain copy or in-place edit
- **Plain copy of only `diffusion_es_planner.py` would crash**: yixiz's
  `scores/scorer_lane_graph.py` has no `compute_offroad_score_v2` (called by
  `compute_score`), and yixiz's `data/preprocess.py` returns 2-tuple map features
  where mineuih's returns 3-tuples with `lane_ids`.
- **Overwriting yixiz's shared modules in place would change existing behavior**:
  `rl/bank.py` and `rl/rl_diffusion_source.py` import those exact divergent modules,
  and mineuih's `compute_score` body genuinely differs.
- **Hermetic vendoring satisfies both guarantees.** Every internal import in the
  closure is top-level-absolute (`from scores...`, `from data...`) and the entrypoint
  does `sys.path.insert(0, REPO_ROOT)` with `REPO_ROOT = parents[1]`. Running from
  `es_baseline/` therefore resolves imports to the vendored copies only.

## Behavior-preservation proof (static)
`VENDOR_MANIFEST.sha256` lists sha256 of all 71 vendored `.py` files. Every one is
byte-identical to its source in `/zfsauton2/home/mineuih/waymax_rs`. Re-verify:

```bash
M=/zfsauton2/home/mineuih/waymax_rs
cd /zfsauton2/home/yixiz/waymax_rs/es_baseline
while read h f; do rel=${f#./}; s=$(sha256sum "$M/$rel" | cut -d' ' -f1); \
  [ "$h" != "$s" ] && echo "DRIFT $rel"; done < VENDOR_MANIFEST.sha256; echo done
```

Identical source + hermetic imports ⇒ identical execution. The dynamic run below is a
smoke test and defines the Stage-0 reference arm, not the correctness proof.

## Hermeticity notes
- ES path (`--planner es`) imports only vendored dirs + external packages
  (jax, flax, numpy, waymax, tqdm) that are the same installed packages in this env.
- The un-vendored internal module `vla` is imported **only** by the VLA planners
  (`model/vla/*`) and `data/temporal_loader.py`, which are lazy-imported inside the
  `--planner vla` / `temporal-vla` branches. **Only `es`, `diffusion`, and
  `log-replay` planners are supported here by design.**

## ES configuration that produced the failure set
Defaults in `simulation/run_simulation.py`: `population_size=64`,
`resample_timesteps=3`, `elite_size=4`, `num_es_iterations=3`, `start_timestep=10`,
`replan_interval_steps=10`, `num_worlds=10`, `seed=0`.
ES selection score (`scores/scorer_lane_graph.py`): `collision * offroad ∈ {0,1}` —
binary, **goal-agnostic**.

## Target scenes (from the failure dataset)
24 failures + control successes, across 3 tfrecords:
- tfrecord-00000: fail [71,92,113,134,187,252,315,377]  control [0,1,2,3]
- tfrecord-00001: fail [45,62,147,158,200,243,289,360,463]  control [0,1,2,3]
- tfrecord-00002: fail [0,13,20,24,70,94,97]  control [1,2,3,4]

## Environment (important)
This diffusion pipeline requires **flax >= 0.12 with `nnx.List`** (used by
`model/diffusion/modules/mlp.py`, identical in both trees). mineuih's env
**`waymax_rs`** has flax **0.12.7** and works; yixiz's `waymax` env has flax
**0.10.6** (no `nnx.List`) and fails at model construction.

`waymax_rs` lives on shared scratch (mounted on every node), so the launchers
drive the run with its absolute interpreter directly — no fragile cross-user
`conda activate`:

    PYTHON=/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python

Override with `PYTHON=/your/env/bin/python`. To make your own env instead:
`conda create -n es_rs --clone /zfsauton/scratch/mineuih/conda_envs/waymax_rs`
(or `pip install "flax==0.12.7" "jax==0.10.0"` into an existing env).

## How to run the Stage-0 reference arm
Checkpoint default is baked in (`.../pretrain_diffusion_without_subgoal_.../latest`,
`subgoal_conditioned=False`, horizon 25, dt 0.2). Just:
```bash
cd /zfsauton2/home/yixiz/waymax_rs/es_baseline
bash repro_verify.sh      # or wrap in srun/sbatch on a GPU node
```

## Orbax restore fix (no vendored file touched)
orbax 0.11.39 (in `waymax_rs`) rejects the vendored arg-less
`PyTreeCheckpointer().restore(path)` with *"sharding ... Got None"*.
`experiments/orbax_patch.py` monkeypatches `PyTreeCheckpointer.restore` at runtime
to retry with a single-device sharding **only if** the original call raises —
loading the identical weights (value-preserving). Both entrypoints apply it:
`run_experiment.py` calls `orbax_patch.apply()`; the baseline runs through
`experiments/run_baseline.py` (a thin patched wrapper around
`simulation/run_simulation.py`). All 71 vendored files remain byte-identical.

## experiments/ — oracle + diversity toolkit (additive, baseline untouched)
All experiment code imports the vendored modules but never edits them.

- `dense_scorer.py` — **Stage-1 dense oracle**. `DenseReturnScorer` computes a
  weight-normalized, discounted **per-step** return in [0,1] from the same
  simulator quantities the evaluator uses: collision, off-road, goal
  progress/closeness, agent separation, speed, comfort (accel/yaw-rate). It
  mirrors `Scorer.compute_score`'s ego/object construction and
  `compute_offroad_score_v2`'s geometry but keeps the time axis (dense, not a
  bit). `binary_baseline_score` reproduces the reference score via the vendored
  Scorer so both arms share one code path. Because other agents are non-reactive
  log-replay, this open-loop return equals the closed-loop rollout — a faithful
  oracle, not a learned model.
- `descriptors.py` — **Stage-2 features**. 17 behavior descriptors per candidate
  (final long/lat, speed profile, min speed, max brake, max lateral, lane-change,
  maneuver timing, route progress, min separation, off-road fraction, comfort) in
  a shared ego frame, plus `select_diverse` (farthest-point sampling) and
  `count_modes` for the no-ES coverage diagnostic.
- `es_instrumented.py` — `InstrumentedESPlanner(DiffusionESPlanner)`. Identical
  sampling / mutation / top-k elitism / budget; only the **selection score** is
  pluggable (`binary` | `dense`) and every Stage-0 quantity is logged
  (init has-safe, #safe, best binary/dense, unique parents per iteration,
  selected candidate's both-scores).
- `run_experiment.py` — entrypoint; one arm over one tfrecord. Final success is
  always the original binary criterion, whatever score ES used internally.
- `run_arms.sh` — runs both arms (binary, dense) over the 24 failures + controls
  across all 3 tfrecords with paired seeds, then calls the analyzer.
- `analyze_arms.py` — paired report: per-arm success, failure→success
  conversions, Stage-0 instrumentation summary, and Spearman(dense return,
  success).

Run the Stage-0 + Stage-1 comparison:
```bash
cd /zfsauton2/home/yixiz/waymax_rs/es_baseline
bash experiments/run_arms.sh
```

### Online SAC initialization (`--sac_online`)
`sac_bank.py` + `dump_sac_init_bank.py` are the **offline** SAC init: K closed-loop
episodes rolled once per scene from the env's own reset point (scenario step 10),
re-sliced at every replan. Once ES has executed a few non-SAC candidates the live ego
is off that schedule, and the bank has to be re-anchored onto it by nearest-point
search — a splice the policy never proposed.

`sac_online.py` (`--sac_online --sac_run_dir ...`) replaces that with **K rollouts
launched from the live ego state at every replan**:

- the executed ego history goes into a clean V-Max state's `sim_trajectory`, never
  its `log_trajectory` — ES rewrites the ego's log with each accepted plan, and V-Max
  reads the log for the goal and the route, so feeding it back would move the goal;
- trajectory index 0 is the ego's current pose, so `init_sac_anchor_gap_m` is 0 by
  construction and the re-anchoring search is a no-op;
- diversity comes from sampling the policy (`deterministic=False`), with
  `--sac_action_noise` as a knob if the policy's own entropy is too low;
- no bank, no manifest, no ScenarioMax hit list — any scenario in the shard works,
  which is why this arm uses the same scene lists as `run_arms.sh`.

The env, observation config, object budget and network all come from the training
run's own `.hydra/config.yaml`, and the checkpoint defaults to `model/model_best.pkl`
(V-Max's own picker takes the highest-numbered file, which is a later, worse one).

```bash
# smoke: does the policy load, anchor on the live pose, and give a diverse population?
sbatch es_baseline/slurm_sac_online_smoke.sbatch

# the arm (dense selection + online sac_safe init) over all 3 shards
SCRIPT=es_baseline/experiments/run_sac_online_arms.sh sbatch es_baseline/slurm_es.sbatch
```

Running V-Max in the ES interpreter needed three of its imports made optional
(`gymnasium`, `distrax`, `tensorboardX`) — all training/plotting-only dependencies
that the inference path never touches.

### Go/no-go read-out
- **conversions > 0 with controls kept** → dense reward resolution has oracle
  leverage → learning a value/ranking model is justified (Stage 5 GO).
- **init_has_safe high yet binary keeps failing** → the bottleneck is
  selection/scoring mismatch (ES scorer ≠ evaluator) or coverage, not
  reward-ranking-toward-goal — matches the failure-set finding that 19/24
  failures already reach the goal but violate in-objective safety terms.
