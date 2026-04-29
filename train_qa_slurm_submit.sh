#!/bin/bash
set -euo pipefail

# if [[ $# -lt 1 || $# -gt 2 ]]; then
#   echo "Usage: $0 <learning_rate> [prefix]"
#   echo "Example: $0 1e-4 waymax_qa"
#   exit 1
# fi

GPU_TYPE="${1:-a6000}"
LR="${2:-0.0001}"
PREFIX="${3:-waymax_qa}"

# Make LR filename/job-safe and easy to scan.
LR_TAG="$(echo "${LR}" | sed -E 's/\./p/g; s/-/m/g; s/\+//g')"
RUN_NAME="${PREFIX}_lr-${LR_TAG}"

mkdir -p logs

echo "Submitting job with:"
echo "  GPU_TYPE=${GPU_TYPE}"
echo "  lr=${LR}"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --gres="gpu:${GPU_TYPE}:1" \
  --export=ALL,LR="${LR}",WANDB_NAME="${RUN_NAME}",GPU_TYPE="${GPU_TYPE}" \
  train_qa_slurm.sh