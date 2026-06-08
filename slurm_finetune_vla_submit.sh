#!/bin/bash
set -euo pipefail

SCENE_TOKENIZER_CKPT_PATH="${1:-}"

RUN_NAME="finetune_vla"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=SCENE_TOKENIZER_CKPT_PATH="${SCENE_TOKENIZER_CKPT_PATH}" \
  slurm_finetune_vla.sh