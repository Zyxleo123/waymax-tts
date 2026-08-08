#!/bin/bash
#SBATCH --job-name=ckpteval-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/report-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/report-%j.out
#
# Aggregate the sweep's per-episode CSVs into the clean-success / failure-mode
# table. Reads only saved artefacts, so it runs on CPU and can be rerun freely.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
PREAMBLE="${SUBMIT_DIR}/V-Max/slurm/_ckpt_eval_preamble.sh"
[ -f "${PREAMBLE}" ] || PREAMBLE=/zfsauton2/home/yixiz/waymax_rs/V-Max/slurm/_ckpt_eval_preamble.sh
# shellcheck source=/dev/null
. "${PREAMBLE}" "$@"

set -euo pipefail   # re-assert in the job body, not only in the preamble

ckpt_activate_conda
export PYTHONUNBUFFERED=1

python "${VMAX_ROOT}/slurm/aggregate_ckpt_eval.py" \
    --eval-out "${EVAL_OUT}" \
    --report-dir "${REPORT_DIR}" \
    --goal-col "${GOAL_RADIUS_COL}"

echo "Report: ${REPORT_DIR}/checkpoint_summary.txt"
