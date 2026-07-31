#!/usr/bin/env bash
# Stage 3: dense oracle + SAC-safe offline init on the 29 ScenarioMax-hit scenes.
#
# Requires dump_sac_init_bank.py output on scratch first:
#   sbatch es_baseline/slurm_dump_sac_bank.sbatch
#
# Then:
#   bash es_baseline/experiments/run_sac_init_arms.sh
#   SCRIPT=es_baseline/experiments/run_sac_init_arms.sh sbatch es_baseline/slurm_es.sbatch
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # es_baseline/

PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
CONDA_BASE="${CONDA_BASE:-/opt/miniconda3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  set +eu
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate /zfsauton/scratch/mineuih/conda_envs/waymax_rs 2>/dev/null || true
  set -eu
fi
"$PYTHON" -c "import flax.nnx as nnx; assert hasattr(nnx,'List')" 2>/dev/null || {
  echo "ERROR: $PYTHON lacks flax nnx.List. Point PYTHON at an env with flax>=0.12."; exit 1; }

CKPT="${CKPT:-/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260711_141745/latest}"
SEED="${SEED:-0}"
OUT="${OUT:-/zfsauton/scratch/yixiz/waymax_rs/es_baseline/experiments/out}"
case "$OUT" in
  /zfsauton/scratch/*|/scratch/*) ;;
  *) echo "ERROR: OUT must be on scratch, got: $OUT"; exit 1 ;;
esac
mkdir -p "$OUT"

SMX_DIR="${SMX_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoES}"
SAC_BANK="${SAC_BANK:-$SMX_DIR/sac_init_bank.npz}"
SAC_MANIFEST="${SAC_MANIFEST:-$SMX_DIR/es_scenes_manifest.json}"
INIT_SELECT="${INIT_SELECT:-sac_safe}"
# diverse_sac* only: SAC quota inside K, and diffusion-bank oversampling factor.
SAC_FRAC="$(printf '%g' "${SAC_FRAC:-0.5}")"   # %g so ARM matches run_experiment.py
BANK_MULT="${BANK_MULT:-4}"

# Arm label must match run_experiment.py's naming, or the taxonomy step below
# looks for a directory that does not exist.
case "$INIT_SELECT" in
  sac|sac_safe)              ARM="dense_${INIT_SELECT}" ;;
  diverse_sac|diverse_sac_safe)
                             ARM="dense_${INIT_SELECT}_m${BANK_MULT}_f${SAC_FRAC}" ;;
  *) echo "ERROR: INIT_SELECT must be sac|sac_safe|diverse_sac|diverse_sac_safe, got $INIT_SELECT"; exit 1 ;;
esac

if [[ ! -f "$SAC_BANK" ]]; then
  echo "ERROR: missing SAC bank $SAC_BANK"
  echo "  Run: sbatch es_baseline/slurm_dump_sac_bank.sbatch"
  exit 1
fi
if [[ ! -f "$SAC_MANIFEST" ]]; then
  echo "ERROR: missing manifest $SAC_MANIFEST"
  exit 1
fi

# Build per-shard hit index lists from the manifest (scenariomax_hit=true only).
HIT_MAP="$OUT/_sac_hit_indices.env"
"$PYTHON" - <<PY
import json
from collections import defaultdict
from pathlib import Path
m = json.load(open("$SAC_MANIFEST"))
hits = defaultdict(list)
for r in m["records"]:
    if r.get("scenariomax_hit"):
        hits[str(r["tf_example_shard"])].append(int(r["tf_example_idx"]))
lines = []
total = 0
for shard, idxs in sorted(hits.items()):
    idxs = sorted(idxs)
    total += len(idxs)
    lines.append(f'IDX_{shard}="{ ",".join(str(i) for i in idxs) }"')
    print(f"shard {shard}: {len(idxs)} hits -> {idxs}")
print(f"total hits: {total}")
Path("$HIT_MAP").write_text("\n".join(lines) + "\n")
PY
# shellcheck disable=SC1090
source "$HIT_MAP"

# Optional per-shard scenario filter, e.g. SCEN_IDX_00001="3,45,62" to test only
# a handful of scenes instead of the full shard. Intersected with the actual
# SMX hits so a typo can't silently smuggle in a non-hit scenario.
for shard in 00000 00001 00002; do
  var="IDX_${shard}"
  override_var="SCEN_IDX_${shard}"
  override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    filtered="$("$PYTHON" -c "
hits = set(int(i) for i in '${!var:-}'.split(',') if i)
want = [int(i) for i in '${override}'.split(',') if i]
bad = [i for i in want if i not in hits]
if bad:
    raise SystemExit(f'SCEN_IDX_${shard} has non-hit indices {bad}; hits are {sorted(hits)}')
print(','.join(str(i) for i in want))
")"
    declare "$var=$filtered"
    echo "shard ${shard}: filtered to $filtered"
  fi
done

# Peak GPU memory scales with the simulated batch. runner.run already splits the
# scenario list into --num_worlds chunks, so cap the batch here to trade wall
# time for headroom (MAX_WORLDS=3 is the fallback if a shard still OOMs).
MAX_WORLDS="${MAX_WORLDS:-12}"

# Metric logging: WANDB=1 to enable. Trajectories stay on disk either way.
WANDB_ARGS=()
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT:-es-reward-resolution}"
              --wandb_mode "${WANDB_MODE:-online}" --wandb_dir "${WANDB_DIR:-$OUT/_wandb}")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  mkdir -p "${WANDB_DIR:-$OUT/_wandb}"
fi

failed_shards=()
for shard in ${SHARDS:-00000 00001 00002}; do
  var="IDX_${shard}"
  idx="${!var:-}"
  if [[ -z "$idx" ]]; then
    echo "skip tf${shard}: no SMX hits"
    continue
  fi
  nw=$(awk -F',' '{print NF}' <<<"$idx")
  batch=$(( nw < MAX_WORLDS ? nw : MAX_WORLDS ))
  echo "=== arm=dense+${INIT_SELECT}  tf${shard}  ($nw scenes, batch=$batch) ==="
  # One shard's failure (e.g. OOM) must not strand the remaining shards.
  if ! "$PYTHON" "$HERE/experiments/run_experiment.py" \
    --diffusion_ckpt_path "$CKPT" \
    --tfrecord_dir "$HERE/repro/tf${shard}" \
    --selection dense \
    --init_select "$INIT_SELECT" \
    --sac_frac "$SAC_FRAC" --init_bank_multiplier "$BANK_MULT" \
    --sac_bank_path "$SAC_BANK" \
    --sac_manifest "$SAC_MANIFEST" \
    --tf_shard "$shard" \
    --tag "tf${shard}" --seed "$SEED" \
    --num_worlds "$batch" --num_scenarios "$nw" \
    --scenario_indices "$idx" \
    --exp_dir "$OUT" --visualize_mode none --save_trajectory \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}; then
    echo "ERROR: tf${shard} failed (see traceback above); continuing with next shard"
    failed_shards+=("$shard")
  fi
done

echo
echo "=== taxonomy (dense vs ${ARM} on overlapping scenes) ==="
"$PYTHON" "$HERE/experiments/analyze_taxonomy.py" \
  --out "$OUT" \
  --baseline_arm dense \
  --diverse_arm "$ARM" \
  --overlap_only

if (( ${#failed_shards[@]} > 0 )); then
  echo "FAILED shards: ${failed_shards[*]} (taxonomy above covers the shards that ran)"
  exit 1
fi
