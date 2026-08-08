#!/bin/bash
# Submitter for the checkpoint sweep. Calls sbatch and nothing else.
#
#   bash V-Max/slurm/submit_ckpt_eval.sh            # convert -> sweep all ckpts -> report
#   SMOKE=1 bash V-Max/slurm/submit_ckpt_eval.sh    # convert -> model_14341120 only -> report
#
# Overridable: RUN_DIR, WORK_ROOT, CKPT_FILTER, PRIORITY_CKPT, BATCH_SIZE,
# GPU_PARTITION, GPU_GRES, SKIP_CONVERT=1 (datasets already built).
#
# Configuration reaches the jobs through a timestamped env FILE passed as a
# positional argument — never `sbatch --export`, which on this cluster sets
# SLURM_GET_USER_ENV=1 and gets the job requeued+held with
# "(user env retrieval failed requeued held)".

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/zfsauton2/home/yixiz/waymax_rs}"
SLURM_SCRIPTS="${REPO_ROOT}/V-Max/slurm"
WORK_ROOT="${WORK_ROOT:-/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01}"

GPU_PARTITION="${GPU_PARTITION:-general}"
GPU_GRES="${GPU_GRES:-gpu:1}"

STAMP="$(date +%Y%m%d-%H%M%S)"
ENV_DIR="${WORK_ROOT}/envs"
ENV_FILE="${ENV_DIR}/${STAMP}.env"
mkdir -p "${ENV_DIR}" "${WORK_ROOT}/logs"

if [ "${SMOKE:-0}" = "1" ]; then
    CKPT_FILTER="${CKPT_FILTER:-model_14341120.pkl}"
fi

{
    echo "# generated $(date) by submit_ckpt_eval.sh"
    echo "export REPO_ROOT='${REPO_ROOT}'"
    echo "export WORK_ROOT='${WORK_ROOT}'"
    echo "export RUN_DIR='${RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2}'"
    # Point a new WORK_ROOT at an existing DS_DIR to reuse converted data across runs.
    [ -n "${DS_DIR:-}" ] && echo "export DS_DIR='${DS_DIR}'"
    echo "export CKPT_FILTER='${CKPT_FILTER:-*.pkl}'"
    echo "export PRIORITY_CKPT='${PRIORITY_CKPT:-model_14341120.pkl}'"
    echo "export BATCH_SIZE='${BATCH_SIZE:-8}'"
    echo "export MAX_NUM_OBJECTS='${MAX_NUM_OBJECTS:-64}'"
    echo "export EVAL_SEED='${EVAL_SEED:-0}'"
    echo "export CONVERT_WORKERS='${CONVERT_WORKERS:-8}'"
    echo "export GOAL_RADIUS_COL='${GOAL_RADIUS_COL:-reached_goal}'"
} > "${ENV_FILE}"

echo "env file: ${ENV_FILE}"
cat "${ENV_FILE}"
echo

# Slurm variables inherited from an enclosing allocation confuse sbatch; drop them.
sb() { env $(compgen -v | grep '^SLURM_' | sed 's/^/-u /') sbatch "$@"; }

LOGS="${WORK_ROOT}/logs"

DEP=""
if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    CONVERT_ID=$(sb --parsable \
        --output="${LOGS}/convert-%A_%a.out" --error="${LOGS}/convert-%A_%a.out" \
        "${SLURM_SCRIPTS}/convert_shards.sbatch" "${ENV_FILE}")
    echo "convert  (array 0-1, cpu) : ${CONVERT_ID}"
    DEP="--dependency=afterok:${CONVERT_ID}"
else
    echo "convert                   : skipped (SKIP_CONVERT=1)"
fi

SWEEP_ID=$(sb --parsable ${DEP} \
    --partition="${GPU_PARTITION}" --gres="${GPU_GRES}" \
    --output="${LOGS}/sweep-%j.out" --error="${LOGS}/sweep-%j.out" \
    "${SLURM_SCRIPTS}/eval_ckpt_sweep.sbatch" "${ENV_FILE}")
echo "sweep    (gpu)            : ${SWEEP_ID}"

# afterany, not afterok: if the sweep hits the wall clock we still want the
# report over whatever checkpoints did finish.
REPORT_ID=$(sb --parsable --dependency=afterany:"${SWEEP_ID}" \
    --output="${LOGS}/report-%j.out" --error="${LOGS}/report-%j.out" \
    "${SLURM_SCRIPTS}/aggregate_ckpt_eval.sbatch" "${ENV_FILE}")
echo "report   (cpu)            : ${REPORT_ID}"

echo
echo "Logs   : ${WORK_ROOT}/logs/"
echo "Report : ${WORK_ROOT}/report/checkpoint_summary.txt"
