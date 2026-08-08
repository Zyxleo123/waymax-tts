#!/usr/bin/env bash
# Stage 3-online: dense oracle + SAC init rolled *online* from the live ego state.
#
# Same arm as run_sac_init_arms.sh, with the offline NPZ bank replaced by
# --sac_online: at every replan the V-Max SAC policy is rolled K times from the ES
# ego's current pose, so the initialization tracks the search instead of being a
# frozen answer to "what would you do from the start".
#
# Two consequences worth knowing before comparing runs:
#   * No bank, no manifest, no ScenarioMax hit list — any scenario in the shard can
#     be run, so this uses the same scene lists as run_arms.sh (failure set + controls).
#   * It costs GPU time inside the search: K rollouts x 50 sim steps per world per
#     replan, on top of the diffusion population.
#
#   bash es_baseline/experiments/run_sac_online_arms.sh
#   SCRIPT=es_baseline/experiments/run_sac_online_arms.sh sbatch es_baseline/slurm_es.sbatch
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

# womd_raw_parity_sac_lq, model_best.pkl == step 1797120 (0.942 reached_goal).
SAC_RUN_DIR="${SAC_RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_womd_raw_parity/womd_raw_parity_sac_lq}"
SAC_MODEL="${SAC_MODEL:-$SAC_RUN_DIR/model/model_best.pkl}"
SAC_ROLLOUT_STEPS="${SAC_ROLLOUT_STEPS:-50}"   # 5 s at dt=0.1 == the ES predict horizon
SAC_ACTION_NOISE="${SAC_ACTION_NOISE:-0.0}"
INIT_SELECT="${INIT_SELECT:-sac_safe}"
SAC_FRAC="$(printf '%g' "${SAC_FRAC:-0.5}")"   # %g so ARM matches run_experiment.py
BANK_MULT="${BANK_MULT:-4}"
POP="${POP:-64}"
SAC_K="${SAC_K:-$POP}"

case "$INIT_SELECT" in
  sac|sac_safe)              ARM="dense_${INIT_SELECT}_online" ;;
  diverse_sac|diverse_sac_safe)
                             ARM="dense_${INIT_SELECT}_m${BANK_MULT}_f${SAC_FRAC}_online" ;;
  *) echo "ERROR: INIT_SELECT must be sac|sac_safe|diverse_sac|diverse_sac_safe, got $INIT_SELECT"; exit 1 ;;
esac

for f in "$SAC_RUN_DIR/.hydra/config.yaml" "$SAC_MODEL"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

# Same scenes as the binary/dense reference arms (failure set + controls per shard).
declare -A IDX=(
  [tf00000]="71,92,113,134,187,252,315,377,0,1,2,3"
  [tf00001]="45,62,147,158,200,243,289,360,463,0,1,2,3"
  [tf00002]="0,13,20,24,70,94,97,1,2,3,4"
)

# Per-shard scene override, e.g. SCEN_IDX_tf00000="71,92,0" for a cheap end-to-end
# check before committing to the full list. Unlike the offline arm there is no bank to
# be a member of, so any index in the shard is fair game.
for shard in tf00000 tf00001 tf00002; do
  override_var="SCEN_IDX_${shard}"
  override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    IDX[$shard]="$override"
    echo "$shard: overridden to $override"
  fi
done

# Peak GPU memory now also carries K replicated V-Max states; drop MAX_WORLDS first
# if a shard OOMs.
MAX_WORLDS="${MAX_WORLDS:-12}"

WANDB_ARGS=()
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT:-es-reward-resolution}"
              --wandb_mode "${WANDB_MODE:-online}" --wandb_dir "${WANDB_DIR:-$OUT/_wandb}")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  mkdir -p "${WANDB_DIR:-$OUT/_wandb}"
fi

failed_shards=()
for shard in ${SHARDS:-tf00000 tf00001 tf00002}; do
  idx="${IDX[$shard]:-}"
  if [[ -z "$idx" ]]; then
    echo "skip $shard: no scene list"
    continue
  fi
  nw=$(awk -F',' '{print NF}' <<<"$idx")
  batch=$(( nw < MAX_WORLDS ? nw : MAX_WORLDS ))
  echo "=== arm=$ARM  $shard  ($nw scenes, batch=$batch) ==="
  if ! "$PYTHON" "$HERE/experiments/run_experiment.py" \
    --diffusion_ckpt_path "$CKPT" \
    --tfrecord_dir "$HERE/repro/$shard" \
    --selection dense \
    --init_select "$INIT_SELECT" \
    --sac_frac "$SAC_FRAC" --init_bank_multiplier "$BANK_MULT" \
    --sac_online \
    --sac_run_dir "$SAC_RUN_DIR" \
    --sac_model "$SAC_MODEL" \
    --sac_k "$SAC_K" \
    --sac_rollout_steps "$SAC_ROLLOUT_STEPS" \
    --sac_action_noise "$SAC_ACTION_NOISE" \
    --population_size "$POP" \
    --tag "$shard" --seed "$SEED" \
    --num_worlds "$batch" --num_scenarios "$nw" \
    --scenario_indices "$idx" \
    --exp_dir "$OUT" --visualize_mode none --save_trajectory \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}; then
    echo "ERROR: $shard failed (see traceback above); continuing with next shard"
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
