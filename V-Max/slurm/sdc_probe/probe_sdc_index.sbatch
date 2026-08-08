#!/bin/bash
#SBATCH --job-name=vmax-sdcprobe
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# Reads tf_example shards only -- no GPU. Asking for one would just queue this
# behind real training work.
#
# Configuration arrives as a positional argument: either a path to an env file
# (written by submit_sdc_probe.sh) or bare VAR=value pairs. Never via
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets the job requeued
# and held with `(user env retrieval failed requeued held)`.

# `set -euo pipefail` belongs in the job body, not only in a sourced preamble:
# if the source itself fails, the job must not sail on with the system python
# and then exit 0.
set -euo pipefail

for arg in "$@"; do
    if [ -f "${arg}" ]; then
        # shellcheck source=/dev/null
        . "${arg}"
    elif [[ "${arg}" == *=* ]]; then
        export "${arg?}"
    fi
done

WOMD_ROOT="${WOMD_ROOT:-/zfsauton/scratch/eshau/womd/tf_example}"
SHARDS="${SHARDS:-${WOMD_ROOT}/training/training_tfexample.tfrecord-*}"
NUM_SHARDS="${NUM_SHARDS:-20}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-64}"
OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/sdc_probe}"
PROBE_DIR="${PROBE_DIR:-${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs/V-Max/slurm/sdc_probe}}"
CONDA_SH="${CONDA_SH:-/zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-waymax}"

# The env's activate.d cuda-nvcc hooks read these without a default and abort
# under `set -u`.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
set +u
# shellcheck source=/dev/null
. "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""   # keep TensorFlow off any GPU it might see
export PYTHONUNBUFFERED=1

echo "Shards   : ${SHARDS}"
echo "Sampling : ${NUM_SHARDS} shard(s)"
echo "Threshold: max_num_objects=${MAX_NUM_OBJECTS}"
echo "Out dir  : ${OUT_DIR}"

mkdir -p "${OUT_DIR}"

python "${PROBE_DIR}/probe_sdc_index.py" \
    --shards "${SHARDS}" \
    --num_shards "${NUM_SHARDS}" \
    --max_num_objects "${MAX_NUM_OBJECTS}" \
    --out_dir "${OUT_DIR}"

echo "Done. Re-tabulate without re-reading shards from ${OUT_DIR}/rows.jsonl"
