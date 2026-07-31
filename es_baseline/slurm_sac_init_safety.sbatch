#!/bin/bash
#SBATCH --job-name=sac_init64
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#
# Path 2 step 2: stochastic SAC K=64 init safety on ES ScenarioMax shard.
# All large artifacts + logs under scratch OUT_DIR.
#
#   sbatch es_baseline/slurm_sac_init_safety.sbatch
#   BID=$(sbatch --parsable es_baseline/slurm_build_es_scenariomax.sbatch)
#   sbatch --dependency=afterok:$BID es_baseline/slurm_sac_init_safety.sbatch
set -euo pipefail

# Slurm may copy this script to /var/spool/slurm/ — do not derive paths from BASH_SOURCE.
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoES}"
case "$OUT_DIR" in
  /zfsauton/scratch/*|/scratch/*) ;;
  *) echo "ERROR: OUT_DIR must be on scratch, got: $OUT_DIR"; exit 1 ;;
esac
mkdir -p "$OUT_DIR/logs"

RUN_DIR="${RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2}"
MODEL="${MODEL:-$RUN_DIR/model/model_final.pkl}"
K="${K:-64}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"   # e.g. LIMIT=1 for smoke
PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/waymax/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATASET="$OUT_DIR/es_scenes.tfrecord"
OUT_JSON="${OUT_JSON:-$OUT_DIR/sac_init_safety.json}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: missing $DATASET — run slurm_build_es_scenariomax.sbatch first"
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: missing model $MODEL"
  exit 1
fi

EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "dataset=$DATASET  model=$MODEL  k=$K  limit=${LIMIT:-all}  out=$OUT_JSON"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" es_baseline/experiments/diagnose_sac_init_safety.py \
  --run-dir "$RUN_DIR" \
  --path-dataset "$DATASET" \
  --model "$MODEL" \
  --out "$OUT_JSON" \
  --k "$K" \
  --seed "$SEED" \
  "${EXTRA[@]}"

echo "done: $OUT_JSON"
