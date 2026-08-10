#!/bin/bash
# Submitter for DPPO fine-tuning. Only calls sbatch.
#
# Writes a timestamped env file with this run's config and passes its path as a
# positional arg to the batch script (no `sbatch --export`).
#
# Usage (from repo root):
#   # short GPU smoke (2 iters) to shake out integration shapes:
#   DFT_ITERS=2 DFT_INDICES=0,1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#   # overfit test (8 scenes):
#   bash rl/diffusion_ft/slurm/submit_train_dppo.sh
#   # DPPO + expert anchor:
#   DFT_EXPERT_WEIGHT=0.1 bash rl/diffusion_ft/slurm/submit_train_dppo.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
ENV_DIR="$ROOT/logs/dft/env"
mkdir -p "$ENV_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="$ENV_DIR/dppo_${STAMP}.env"

{
  echo "DFT_CKPT=${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
  echo "DFT_TFRECORD=${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"
  echo "DFT_INDICES=${DFT_INDICES:-0,1,2,3,4,5,6,7}"
  echo "DFT_ITERS=${DFT_ITERS:-50}"
  echo "DFT_EXPERT_WEIGHT=${DFT_EXPERT_WEIGHT:-0.0}"
  echo "DFT_K_TRAINABLE=${DFT_K_TRAINABLE:-10}"
  echo "DFT_PREFIX_LEN=${DFT_PREFIX_LEN:-5}"
  echo "DFT_ACTOR_LR=${DFT_ACTOR_LR:-1e-5}"
  echo "DFT_SEED=${DFT_SEED:-0}"
  echo "DFT_OUT_DIR=${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo}"
} > "$ENV_FILE"

echo "Wrote env file: $ENV_FILE"; cat "$ENV_FILE"

# Optional scheduling overrides (all `general` a6000s are often full). These are
# plain sbatch flags -- NOT --export -- so they don't trigger user-env retrieval.
#   DFT_JOBNAME   distinct job name (also names the log via %x)
#   DFT_PARTITION e.g. project (h200 gpu2) or legacy (v100 gpu20/21/23)
#   DFT_NODELIST  pin a node, e.g. gpu2  (or gpu20)
#   DFT_EXCLUDE   e.g. exclude the 11GB 2080 Tis on legacy
SB_FLAGS=()
[[ -n "${DFT_JOBNAME:-}" ]]   && SB_FLAGS+=(--job-name="$DFT_JOBNAME")
[[ -n "${DFT_PARTITION:-}" ]] && SB_FLAGS+=(--partition="$DFT_PARTITION")
[[ -n "${DFT_NODELIST:-}" ]]  && SB_FLAGS+=(--nodelist="$DFT_NODELIST")
[[ -n "${DFT_EXCLUDE:-}" ]]   && SB_FLAGS+=(--exclude="$DFT_EXCLUDE")

echo "sbatch flags: ${SB_FLAGS[*]:-<none>}"
env -i HOME="$HOME" PATH="$PATH" USER="${USER:-}" \
  sbatch "${SB_FLAGS[@]}" rl/diffusion_ft/slurm/train_dppo.sbatch "$ENV_FILE"
