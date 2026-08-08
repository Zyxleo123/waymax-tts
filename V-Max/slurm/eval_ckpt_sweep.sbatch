#!/bin/bash
#SBATCH --job-name=ckpteval-sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/sweep-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/ckpt_eval_shard01/logs/sweep-%j.out
#
# Evaluate every checkpoint of a V-Max run on both converted splits.
#
# V-Max's get_model_path() always picks model_final.pkl (or the highest-numbered
# checkpoint) out of a run's model/ directory — there is no "evaluate this exact
# file" flag. So we build one shim run-dir per checkpoint, each holding a symlink
# to that single .pkl plus the run's .hydra/config.yaml, and point --path_model
# at the shim.
#
# Resumable: a (checkpoint, split) pair whose evaluation_episodes.csv already
# exists is skipped, so hitting the wall clock means resubmitting, not restarting.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
PREAMBLE="${SUBMIT_DIR}/V-Max/slurm/_ckpt_eval_preamble.sh"
[ -f "${PREAMBLE}" ] || PREAMBLE=/zfsauton2/home/yixiz/waymax_rs/V-Max/slurm/_ckpt_eval_preamble.sh
# shellcheck source=/dev/null
. "${PREAMBLE}" "$@"

set -euo pipefail   # re-assert in the job body, not only in the preamble

ckpt_activate_conda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

echo "=== checkpoint sweep :: $(date) ==="
echo "run dir  : ${RUN_DIR}"
echo "filter   : ${CKPT_FILTER}"
echo "batch    : ${BATCH_SIZE}   objects: ${MAX_NUM_OBJECTS}   seed: ${EVAL_SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Gates ─────────────────────────────────────────────────────────────────────
[ -f "${HYDRA_CONFIG}" ] || ckpt_gate_fail "no hydra config at ${HYDRA_CONFIG}"
[ -d "${MODEL_DIR}" ]    || ckpt_gate_fail "no model dir at ${MODEL_DIR}"

# V-Max derives its output directory from the --path_dataset string, so we pass a
# short RELATIVE path ("ds/<split>.tfrecord") from ${WORK_ROOT}. When DS_DIR points
# somewhere else (reusing another run's converted data), mirror it into
# ${WORK_ROOT}/ds so the relative path still resolves — and gate on the path the
# eval will actually open, not on DS_DIR, so the two can never disagree.
LOCAL_DS="${WORK_ROOT}/ds"
mkdir -p "${LOCAL_DS}"

SPLITS=""
for split in training validation; do
    if [ "$(readlink -f "${DS_DIR}")" != "$(readlink -f "${LOCAL_DS}")" ] && [ -s "${DS_DIR}/${split}.tfrecord" ]; then
        ln -sfn "$(readlink -f "${DS_DIR}/${split}.tfrecord")" "${LOCAL_DS}/${split}.tfrecord"
    fi

    if [ -s "${LOCAL_DS}/${split}.tfrecord" ]; then
        SPLITS="${SPLITS} ${split}"
    else
        echo "WARNING: no dataset for '${split}' at ${LOCAL_DS}/${split}.tfrecord (DS_DIR=${DS_DIR}) — skipping that split." >&2
    fi
done
[ -n "${SPLITS}" ] && SPLITS="${SPLITS# }"
[ -n "${SPLITS}" ] || ckpt_gate_fail "no dataset for either split; nothing to evaluate"

# ── Checkpoint list: numeric order, with PRIORITY_CKPT (if any) evaluated first ──
mapfile -t ALL_CKPTS < <(
    cd "${MODEL_DIR}" && ls -1 ${CKPT_FILTER} 2>/dev/null |
        sed -E 's/^model_final\.pkl$/999999999999 &/; t; s/^model_([0-9]+)\.pkl$/\1 &/; t; s/^(.*)$/0 \1/' |
        sort -k1,1n -k2,2 | awk '{print $2}'
)
[ "${#ALL_CKPTS[@]}" -gt 0 ] || ckpt_gate_fail "no checkpoints matching '${CKPT_FILTER}' in ${MODEL_DIR}"

CKPTS=()
if [ -n "${PRIORITY_CKPT:-}" ] && [ -f "${MODEL_DIR}/${PRIORITY_CKPT}" ]; then
    for c in "${ALL_CKPTS[@]}"; do [ "${c}" = "${PRIORITY_CKPT}" ] && CKPTS+=("${c}"); done
fi
for c in "${ALL_CKPTS[@]}"; do [ "${c}" = "${PRIORITY_CKPT:-}" ] || CKPTS+=("${c}"); done

echo "splits      : ${SPLITS}"
echo "checkpoints : ${#CKPTS[@]} -> ${CKPTS[*]}"

cd "${WORK_ROOT}"

n_done=0; n_skip=0; n_fail=0
for ckpt in "${CKPTS[@]}"; do
    stem="${ckpt%.pkl}"

    # One shim run-dir holding exactly this checkpoint.
    shim="${SHIM_DIR}/${stem}"
    rm -rf "${shim}"
    mkdir -p "${shim}/model" "${shim}/.hydra"
    ln -sfn "${HYDRA_CONFIG}" "${shim}/.hydra/config.yaml"
    ln -sfn "${MODEL_DIR}/${ckpt}" "${shim}/model/${ckpt}"

    for split in ${SPLITS}; do
        # V-Max writes to <eval_name>/ai/<path_dataset>/<path_model>/<model_stem>/
        out="${EVAL_OUT}/ai/ds/${split}.tfrecord/${stem}/${stem}"
        if [ -s "${out}/evaluation_episodes.csv" ]; then
            echo "-- skip ${ckpt} / ${split} (already evaluated)"
            n_skip=$((n_skip + 1))
            continue
        fi

        echo "== eval ${ckpt} / ${split} :: $(date +%H:%M:%S)"
        t0=$(date +%s)
        if python -m vmax.scripts.evaluate.evaluate \
                --sdc_actor ai \
                --src_dir "${SHIM_DIR}" \
                --path_model "${stem}" \
                --path_dataset "ds/${split}.tfrecord" \
                --eval_name "${EVAL_OUT}" \
                --max_num_objects "${MAX_NUM_OBJECTS}" \
                --batch_size "${BATCH_SIZE}" \
                --seed "${EVAL_SEED}"; then
            echo "   ok in $(( $(date +%s) - t0 ))s -> ${out}"
            n_done=$((n_done + 1))
        else
            # One bad checkpoint must not abort the other 25.
            echo "   FAILED: ${ckpt} / ${split} (continuing)" >&2
            n_fail=$((n_fail + 1))
        fi
    done
done

echo "=== sweep finished :: $(date) — evaluated ${n_done}, skipped ${n_skip}, failed ${n_fail} ==="

# A sweep that evaluated nothing must not exit 0 — otherwise a total failure looks
# identical to a completed run in `myacct`, and the report job quietly writes an
# empty table. The report is chained with afterany, so it still runs and says so.
if [ "${n_done}" -eq 0 ] && [ "${n_fail}" -gt 0 ]; then
    echo "ERROR: every evaluation failed; see the tracebacks above." >&2
    exit 1
fi
