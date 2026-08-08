#!/bin/bash
#SBATCH --job-name=ckpteval-convert
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/convert-%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/convert-%A_%a.out
#
# Convert raw WOMD scenario-proto shards 0 and 1 of one split into a single
# ScenarioMax tfexample tfrecord that V-Max can read.
#
#   array task 0 -> training split
#   array task 1 -> validation split
#
# Scenarios that trip OverpassException / NotEnoughValidObjectsException are
# silently filtered by ScenarioMax's postprocess worker; the counts are logged.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
PREAMBLE="${SUBMIT_DIR}/V-Max/slurm/_ckpt_eval_preamble.sh"
[ -f "${PREAMBLE}" ] || PREAMBLE=/zfsauton2/home/yixiz/waymax_rs/V-Max/slurm/_ckpt_eval_preamble.sh
# shellcheck source=/dev/null
. "${PREAMBLE}" "$@"

set -euo pipefail   # re-assert in the job body, not only in the preamble

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SPLIT=training;   SRC_SUBDIR=training;   SHARDS="${TRAIN_SHARDS}" ;;
    1) SPLIT=validation; SRC_SUBDIR=validation; SHARDS="${VALID_SHARDS}" ;;
    *) echo "Unexpected array task id ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

SRC_DIR="${WOMD_SCENARIO_ROOT}/${SRC_SUBDIR}"
FARM="${RAW_DIR}/${SPLIT}"
DST="${DATA_DIR}/${SPLIT}"
OUT_TFRECORD="${DST}/${SPLIT}.tfrecord"

echo "=== convert ${SPLIT} :: $(date) ==="
echo "src    : ${SRC_DIR}"
echo "shards : ${SHARDS}"
echo "dst    : ${OUT_TFRECORD}"

# Resumable: a finished conversion is not redone.
if [ -s "${OUT_TFRECORD}" ]; then
    echo "Already converted (${OUT_TFRECORD}, $(stat -c %s "${OUT_TFRECORD}") bytes) — skipping."
    ln -sfn "${OUT_TFRECORD}" "${DS_DIR}/${SPLIT}.tfrecord"
    exit 0
fi

# ── Symlink farm: get_waymo_scenarios() does an unsorted os.listdir(), so we
#    hand it a directory that contains exactly the shards we want. ────────────
rm -rf "${FARM}"
mkdir -p "${FARM}"
for shard in ${SHARDS}; do
    [ -f "${SRC_DIR}/${shard}" ] || ckpt_gate_fail "missing raw shard ${SRC_DIR}/${shard}"
    ln -s "${SRC_DIR}/${shard}" "${FARM}/${shard}"
done
echo "Farm contents:"; ls -l "${FARM}"

rm -rf "${DST}"
"${SCENARIOMAX_ROOT}/.venv/bin/scenariomax-convert" \
    --waymo_src "${FARM}" \
    --dst "${DST}" \
    --target_format tfexample \
    --tfrecord_name "${SPLIT}" \
    --shard 1 \
    --num_workers "${CONVERT_WORKERS}" \
    --log_level INFO

[ -s "${OUT_TFRECORD}" ] || ckpt_gate_fail "conversion produced no ${OUT_TFRECORD}"

# Short, stable dataset name. V-Max derives its output directory from the
# --path_dataset string, so a long absolute path would nest the results 6 dirs deep.
ln -sfn "${OUT_TFRECORD}" "${DS_DIR}/${SPLIT}.tfrecord"

echo "=== done ${SPLIT} :: $(date) — $(stat -c %s "${OUT_TFRECORD}") bytes ==="
