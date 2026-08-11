#!/bin/bash
# Submitter for the Gate 2b (RL-actor) unit gates. Only calls sbatch.
#
# Usage (from repo root):
#   bash rl/diffusion_ft/slurm/submit_gate2b.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/gate2b_${STAMP}.env"

{
  echo "JAX_PLATFORMS=cpu"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Scrub SLURM_* so submitting from inside an allocation does not leak job state.
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/gate2b.sbatch "$ENV_FILE"
