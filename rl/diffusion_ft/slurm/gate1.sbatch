#!/bin/bash
#SBATCH --job-name=dft_gate1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# Gate 1 smoke test for the diffusion fine-tuning environment.
# `general` hosts only a6000/a5000 (>=24 GB), so the partition alone pins the GPU
# class away from the 11 GB 2080 Tis (which can OOM XLA with no traceback).
#
# Submit with the companion submitter (writes a timestamped env file and passes
# its path as $1):
#     bash rl/diffusion_ft/slurm/submit_gate1.sh
#
# Configuration travels in the positional env-file argument (never --export,
# which triggers user-env retrieval and requeue-hold on this cluster).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

# Source any positional argument that is a path (the env file); export any
# VAR=value argument directly.
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

echo "host=${SLURMD_NODENAME:-unknown}  gpu=${CUDA_VISIBLE_DEVICES:-unset}  root=$ROOT"
echo "DFT_CKPT=${DFT_CKPT:-<default>}"
echo "DFT_TFRECORD=${DFT_TFRECORD:-<default>}"
echo "DFT_INDICES=${DFT_INDICES:-<default>}"
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" -m rl.diffusion_ft.tests.test_gate1
