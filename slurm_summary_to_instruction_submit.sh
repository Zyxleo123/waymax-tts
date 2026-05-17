#!/bin/bash
set -euo pipefail

RUN_NAME="summary_to_instruction"

mkdir -p logs

echo "Submitting job with:"
echo "  run=${RUN_NAME}"
echo "  out=logs/${RUN_NAME}_%j.out"
echo "  err=logs/${RUN_NAME}_%j.err"

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  slurm_summary_to_instruction.sh