#!/bin/bash
set -euo pipefail

TASK="${1:-overtake}"
USE_BASELINE="${2:-true}"
if [[ "$USE_BASELINE" == "true" ]]; then
  RUN_NAME="reward_search_baseline_${TASK}"
else
  RUN_NAME="reward_search_${TASK}"
fi

echo "Submitting job with:"
echo "  task=${TASK}"

mkdir -p logs

sbatch \
  --job-name="${RUN_NAME}" \
  --output="logs/${RUN_NAME}_%j.out" \
  --error="logs/${RUN_NAME}_%j.err" \
  --export=ALL,TASK="${TASK}",NUM_WORLDS=16,USE_BASELINE="${USE_BASELINE}" \
  reward_search_slurm.sh