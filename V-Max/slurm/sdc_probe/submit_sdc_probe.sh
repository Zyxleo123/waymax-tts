#!/bin/bash
# Submit the raw-WOMD SDC-index probe. This script only calls sbatch.
#
#   bash V-Max/slurm/sdc_probe/submit_sdc_probe.sh
#   NUM_SHARDS=100 MAX_NUM_OBJECTS=64 bash V-Max/slurm/sdc_probe/submit_sdc_probe.sh
#
# Configuration travels as a timestamped env file passed positionally -- job
# arguments go through the job record untouched, whereas `sbatch --export` would
# trip SLURM_GET_USER_ENV and get the job held.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/zfsauton/scratch/yixiz/waymax_rs/sdc_probe/env}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${ENV_DIR}/probe_${STAMP}.env"

WOMD_ROOT="${WOMD_ROOT:-/zfsauton/scratch/eshau/womd/tf_example}"
NUM_SHARDS="${NUM_SHARDS:-20}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-64}"
OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/sdc_probe/${STAMP}}"
PARTITION="${PARTITION:-cpu}"

mkdir -p "${ENV_DIR}" "${OUT_DIR}"

cat >"${ENV_FILE}" <<EOF
export WOMD_ROOT="${WOMD_ROOT}"
export SHARDS="${WOMD_ROOT}/training/training_tfexample.tfrecord-*"
export NUM_SHARDS="${NUM_SHARDS}"
export MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS}"
export OUT_DIR="${OUT_DIR}"
export PROBE_DIR="${HERE}"
EOF

echo "Env file : ${ENV_FILE}"
echo "Out dir  : ${OUT_DIR}"

# Submitting from inside an allocation would otherwise leak SLURM_* into the child.
for var in $(env | grep -o '^SLURM_[A-Z_]*' || true); do
    unset "${var}" || true
done

sbatch --partition="${PARTITION}" \
    --chdir="${HERE}" \
    "${HERE}/probe_sdc_index.sbatch" "${ENV_FILE}"
