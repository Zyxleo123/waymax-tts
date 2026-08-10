#!/bin/bash
# Submitter for the Gate 1 smoke test. Only calls sbatch (no work of its own).
#
# Writes a timestamped env file holding this submission's configuration and
# passes its *path* as a positional argument to the batch script -- arguments
# travel through the job record untouched, so we never use `sbatch --export`
# (which triggers user-env retrieval and a requeue-hold on this cluster).
#
# Usage (from the repo root):
#     bash rl/diffusion_ft/slurm/submit_gate1.sh
#     DFT_INDICES=0,1,2,3,4,5 bash rl/diffusion_ft/slurm/submit_gate1.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/gate1_${STAMP}.env"

{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3}"
  # GPU has no ~60s login-node watchdog: run all six gates over full episodes.
  echo "DFT_GATES=${DFT_GATES:-1,2,3,4,5,6}"
  echo "DFT_RG_POINTS=${DFT_RG_POINTS:-30000}"
  # DFT_MAX_REPLANS unset == run each episode to the scenario horizon.
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"
cat "$ENV_FILE"

# Scrub SLURM_* so a submission from inside an allocation does not leak state.
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch rl/diffusion_ft/slurm/gate1.sbatch "$ENV_FILE"
