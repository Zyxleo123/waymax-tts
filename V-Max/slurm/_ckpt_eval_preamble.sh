# Shared preamble for the checkpoint-sweep evaluation jobs.
#
# Sourced by every .sbatch in this pipeline. Configuration arrives as positional
# arguments (NOT via `sbatch --export`, which triggers SLURM_GET_USER_ENV and gets
# the job requeued+held on this cluster):
#   - an argument that is a readable file is sourced (the submitter's env file)
#   - an argument of the form VAR=value is exported
#
# Must be self-sufficient under `env -i`.

set -euo pipefail

for _arg in "$@"; do
    if [ -f "${_arg}" ]; then
        # shellcheck source=/dev/null
        . "${_arg}"
    elif [[ "${_arg}" == *=* ]]; then
        export "${_arg?}"
    fi
done
unset _arg

# ── Defaults (every one overridable from the env file) ────────────────────────
export REPO_ROOT="${REPO_ROOT:-/zfsauton2/home/yixiz/waymax_rs}"
export VMAX_ROOT="${VMAX_ROOT:-${REPO_ROOT}/V-Max}"
export SCENARIOMAX_ROOT="${SCENARIOMAX_ROOT:-${REPO_ROOT}/ScenarioMax}"
export CONDA_SH="${CONDA_SH:-/zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-waymax}"

# Trained run whose checkpoints we sweep.
export RUN_DIR="${RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_repro/repro_sac_v2}"
export MODEL_DIR="${MODEL_DIR:-${RUN_DIR}/model}"
export HYDRA_CONFIG="${HYDRA_CONFIG:-${RUN_DIR}/.hydra/config.yaml}"

# Raw WOMD scenario-proto shards to convert (shard 0 and 1 of each split).
export WOMD_SCENARIO_ROOT="${WOMD_SCENARIO_ROOT:-/zfsauton/scratch/eshau/womd/scenario}"
export TRAIN_SHARDS="${TRAIN_SHARDS:-training.tfrecord-00000-of-01000 training.tfrecord-00001-of-01000}"
export VALID_SHARDS="${VALID_SHARDS:-validation.tfrecord-00000-of-00150 validation.tfrecord-00001-of-00150}"

# Workspace.
export WORK_ROOT="${WORK_ROOT:-/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01}"
export RAW_DIR="${RAW_DIR:-${WORK_ROOT}/raw}"        # symlink farms, one dir per split
export DATA_DIR="${DATA_DIR:-${WORK_ROOT}/data}"     # scenariomax-convert output
export DS_DIR="${DS_DIR:-${WORK_ROOT}/ds}"           # short stable names for eval paths
export SHIM_DIR="${SHIM_DIR:-${WORK_ROOT}/shim}"     # one fake run-dir per checkpoint
export EVAL_OUT="${EVAL_OUT:-${WORK_ROOT}/benchmark}"
export REPORT_DIR="${REPORT_DIR:-${WORK_ROOT}/report}"
export LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

# Eval knobs. MAX_NUM_OBJECTS must match training (repro_sac_v2 used 64).
export CKPT_FILTER="${CKPT_FILTER:-*.pkl}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-64}"
export EVAL_SEED="${EVAL_SEED:-0}"
export CONVERT_WORKERS="${CONVERT_WORKERS:-8}"
export GOAL_RADIUS_COL="${GOAL_RADIUS_COL:-reached_goal}"

mkdir -p "${RAW_DIR}" "${DATA_DIR}" "${DS_DIR}" "${SHIM_DIR}" "${EVAL_OUT}" "${REPORT_DIR}" "${LOG_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
ckpt_activate_conda() {
    # The env's activate.d hooks (cuda-nvcc) read NVCC_PREPEND_FLAGS et al. without
    # a default, so they abort under `set -u`. Pre-seed the one we know about and
    # relax nounset for the duration of activation, then restore it.
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
    set +u
    # shellcheck source=/dev/null
    . "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    set -u
}

ckpt_gate_fail() {
    # A failed decision gate must exit 0 — a non-zero exit strands every afterok
    # dependent in DependencyNeverSatisfied with no explanation in any log.
    echo "GATE FAILED: $*" >&2
    echo "Exiting 0 so downstream jobs still run and report what is missing." >&2
    exit 0
}
