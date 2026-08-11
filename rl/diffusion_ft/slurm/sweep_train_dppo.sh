#!/bin/bash
# Sweep submitter for DPPO fine-tuning: fan out the equal sigma floor x the
# trajectory-KL target as independent sibling jobs (no dependencies, so they
# queue concurrently). Each arm is a full run with its own job name, log, and
# out_dir -- nothing is shared, so a bad arm cannot corrupt another.
#
# This does no work of its own: it only calls the tested single-run submitter
# (submit_train_dppo.sh) once per grid cell, which writes one timestamped env
# file per submission and passes its path positionally (never `sbatch --export`).
#
# Grid (override by exporting a space-separated list):
#   DFT_SIGMA_GRID      equal sample/log-prob floor      (default "0.0025 0.005 0.01")
#   DFT_TARGET_KL_GRID  exact Gaussian trajectory KL, nats (default "0.05 0.2")
# Any other DFT_* (DFT_INDICES, DFT_ITERS, DFT_EXPERT_WEIGHT, DFT_CKPT,
# DFT_PARTITION, DFT_NODELIST, DFT_EXCLUDE, ...) is inherited by every arm.
#
# Usage (from repo root):
#   # default 3x2 = 6 arms, overfit set:
#   bash rl/diffusion_ft/slurm/sweep_train_dppo.sh
#   # sigma-only sweep at one generous target while you calibrate kl_g:
#   DFT_TARGET_KL_GRID=1e9 bash rl/diffusion_ft/slurm/sweep_train_dppo.sh
#   # shorter arms on a specific node:
#   DFT_ITERS=30 DFT_NODELIST=gpu2 DFT_PARTITION=project \
#     bash rl/diffusion_ft/slurm/sweep_train_dppo.sh
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

SIGMA_GRID="${DFT_SIGMA_GRID:-0.0025 0.005 0.01}"
TARGET_KL_GRID="${DFT_TARGET_KL_GRID:-0.05 0.2}"
# Base out_dir; each arm gets a distinct subdir so checkpoints/logs never collide.
BASE_OUT="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# `1.5e-3` / `0.0025` -> `1p5e-3` / `0p0025` for use in a job name / path.
tag() { echo "$1" | sed 's/\./p/g; s/+//g'; }

n=0
echo "Sweep sigma=[$SIGMA_GRID] x target_kl=[$TARGET_KL_GRID], base_out=$BASE_OUT/$STAMP"
for sig in $SIGMA_GRID; do
  for kl in $TARGET_KL_GRID; do
    st="$(tag "$sig")"; kt="$(tag "$kl")"
    arm="s${st}_k${kt}"
    # Per-arm overrides consumed by submit_train_dppo.sh via the environment.
    # Everything else (indices, iters, ckpt, scheduling flags) is inherited.
    DFT_SIGMA_FLOOR="$sig" \
    DFT_TARGET_KL="$kl" \
    DFT_JOBNAME="dppo_${arm}" \
    DFT_OUT_DIR="$BASE_OUT/$STAMP/$arm" \
      bash rl/diffusion_ft/slurm/submit_train_dppo.sh
    n=$((n + 1))
    echo "  [$n] submitted arm $arm (sigma=$sig target_kl=$kl)"
  done
done
echo "Submitted $n sibling arms."
