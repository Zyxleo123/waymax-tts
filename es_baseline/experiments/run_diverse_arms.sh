#!/usr/bin/env bash
# Stage 2: dense oracle + diversified (safe-aware) initialization.
# Same 24 failure scenes + controls / seeds as run_arms.sh, then taxonomy report
# comparing against the existing vanilla-dense runs in experiments/out/.
#
#   bash experiments/run_diverse_arms.sh
#   SCRIPT=es_baseline/experiments/run_diverse_arms.sh sbatch es_baseline/slurm_es.sbatch
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # es_baseline/

PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
CONDA_BASE="${CONDA_BASE:-/opt/miniconda3}"
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
BANK_MULT="${BANK_MULT:-4}"
INIT_SELECT="${INIT_SELECT:-diverse_safe}"

declare -A IDX=(
  [tf00000]="71,92,113,134,187,252,315,377,0,1,2,3"
  [tf00001]="45,62,147,158,200,243,289,360,463,0,1,2,3"
  [tf00002]="0,13,20,24,70,94,97,1,2,3,4"
)

for tf in tf00000 tf00001 tf00002; do
  idx="${IDX[$tf]}"; nw=$(( $(grep -o "," <<<"$idx" | wc -l) + 1 ))
  echo "=== arm=dense+${INIT_SELECT}x${BANK_MULT}  $tf  ($nw scenes) ==="
  "$PYTHON" "$HERE/experiments/run_experiment.py" \
    --diffusion_ckpt_path "$CKPT" \
    --tfrecord_dir "$HERE/repro/$tf" \
    --selection dense \
    --init_select "$INIT_SELECT" \
    --init_bank_multiplier "$BANK_MULT" \
    --tag "$tf" --seed "$SEED" \
    --num_worlds "$nw" --num_scenarios "$nw" \
    --scenario_indices "$idx" \
    --exp_dir "$OUT" --visualize_mode none --save_trajectory
done

echo
echo "=== paired taxonomy (dense vs dense+diverse) ==="
"$PYTHON" "$HERE/experiments/analyze_taxonomy.py" --out "$OUT"
