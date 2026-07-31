#!/bin/bash
#SBATCH --job-name=es_oracle
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#
# Runs the Diffusion-ES experiment on a GPU node in mineuih's waymax_rs env.
# Submit from the repo root (/zfsauton2/home/yixiz/waymax_rs):
#
#   sbatch es_baseline/slurm_es.sbatch                      # -> run_arms.sh (binary + dense)
#   SCRIPT=es_baseline/repro_verify.sh sbatch es_baseline/slurm_es.sbatch   # Stage-0 reference only
#   SCRIPT=es_baseline/experiments/run_diverse_arms.sh sbatch es_baseline/slurm_es.sbatch
#
# Large run outputs default to scratch (see run_arms.sh OUT=).
# The inner scripts already select the waymax_rs python (flax 0.12.7) via the
# PYTHON env var, so no conda activation is required here.
set -euo pipefail

# Slurm may copy this script to /var/spool/slurm/ — submit from repo root.
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

mkdir -p /zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs

SCRIPT="${SCRIPT:-es_baseline/experiments/run_arms.sh}"
export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1   # stream python stdout instead of flushing only on exit
# Keep ES experiment artifacts on scratch unless the caller overrides OUT.
export OUT="${OUT:-/zfsauton/scratch/yixiz/waymax_rs/es_baseline/experiments/out}"

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}  script=$SCRIPT  root=$ROOT"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"
bash "$SCRIPT"
