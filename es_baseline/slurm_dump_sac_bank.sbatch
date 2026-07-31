#!/bin/bash
#SBATCH --job-name=sac_bank64
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#
# Dump K=64 SAC ego trajs for ES sac_safe init (29 SMX-hit scenes).
#
#   cd /zfsauton2/home/yixiz/waymax_rs
#   sbatch es_baseline/slurm_dump_sac_bank.sbatch
set -euo pipefail

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
LIMIT="${LIMIT:-}"
PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/waymax/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATASET="$OUT_DIR/es_scenes.tfrecord"
MANIFEST="$OUT_DIR/es_scenes_manifest.json"
OUT_NPZ="${OUT_NPZ:-$OUT_DIR/sac_init_bank.npz}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: missing $DATASET"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: missing $MANIFEST"
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
if [[ "${NOISY_INIT:-0}" == "1" ]]; then
  # without this the deterministic eval policy yields K identical members
  EXTRA+=(--noisy-init)
fi

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "dataset=$DATASET  model=$MODEL  k=$K  limit=${LIMIT:-all}  out=$OUT_NPZ"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" es_baseline/experiments/dump_sac_init_bank.py \
  --run-dir "$RUN_DIR" \
  --path-dataset "$DATASET" \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --out "$OUT_NPZ" \
  --k "$K" \
  --seed "$SEED" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "done: $OUT_NPZ"
