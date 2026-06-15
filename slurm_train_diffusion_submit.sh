#!/bin/bash
set -euo pipefail

PRETRAINED_CKPT="${1:-}"

RUN_NAME="train_diffusion"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=PRETRAINED_CKPT="${PRETRAINED_CKPT}" \
  slurm_train_diffusion.sh