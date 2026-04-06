#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <learning_rate> [prefix]"
  echo "Example: $0 1e-4 womd"
  exit 1
fi

LR="$1"
PREFIX="${2:-waymax_diffusion}"

# Make LR filename/job-safe and easy to scan.
LR_TAG="$(echo "${LR}" | sed -E 's/\./p/g; s/-/m/g; s/\+//g')"
RUN_NAME="${PREFIX}_lr-${LR_TAG}"

mkdir -p logs

echo "Submitting job with:"
echo "  lr=${LR}"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=ALL,LR="${LR}",WANDB_NAME="${RUN_NAME}" \
  babel_slurm_train.sh
