#!/bin/bash
#SBATCH --job-name=ckpteval-render
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# Render every FAILING scenario of one checkpoint to mp4, on both splits.
#
# Rendering forces batch_dims=(1,) whereas the metrics sweep used (8,1), and the
# dataset is shuffled (shuffle_seed=0) — so scenario_index is only guaranteed
# comparable within one batch layout. This job therefore re-runs the metrics pass
# at batch_size=1 and selects the failures from *that* CSV, instead of trusting the
# batch-8 indexes to line up.
#
# Output (flat, one dir per split, failure mode in the filename):
#   ${VIDEO_ROOT}/<ckpt>/<split>/<bucket>__scenario<idx>.mp4

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

CKPT="${RENDER_CKPT:?set RENDER_CKPT, e.g. model_15365120.pkl}"
STEM="${CKPT%.pkl}"
# Videos live on scratch, not in the repo — they are regenerable artefacts.
VIDEO_ROOT="${VIDEO_ROOT:-${WORK_ROOT}/selected}"
RENDER_OUT="${WORK_ROOT}/render"          # batch-1 metrics + raw mp4 tree
mkdir -p "${VIDEO_ROOT}" "${RENDER_OUT}"

echo "=== render failures :: $(date) ==="
echo "run    : ${RUN_DIR}"
echo "ckpt   : ${CKPT}"
echo "videos : ${VIDEO_ROOT}/${STEM}/"

[ -f "${MODEL_DIR}/${CKPT}" ] || ckpt_gate_fail "no checkpoint ${MODEL_DIR}/${CKPT}"
[ -f "${HYDRA_CONFIG}" ]      || ckpt_gate_fail "no hydra config at ${HYDRA_CONFIG}"

# Shim run-dir holding exactly this checkpoint (V-Max always picks the newest
# .pkl in model/, so it must be alone).
shim="${SHIM_DIR}/${STEM}"
rm -rf "${shim}"; mkdir -p "${shim}/model" "${shim}/.hydra"
ln -sfn "${HYDRA_CONFIG}" "${shim}/.hydra/config.yaml"
ln -sfn "${MODEL_DIR}/${CKPT}" "${shim}/model/${CKPT}"

# Mirror the datasets so the relative path used below resolves (see eval_ckpt_sweep).
LOCAL_DS="${WORK_ROOT}/ds"; mkdir -p "${LOCAL_DS}"
for split in training validation; do
    if [ "$(readlink -f "${DS_DIR}")" != "$(readlink -f "${LOCAL_DS}")" ] && [ -s "${DS_DIR}/${split}.tfrecord" ]; then
        ln -sfn "$(readlink -f "${DS_DIR}/${split}.tfrecord")" "${LOCAL_DS}/${split}.tfrecord"
    fi
done

cd "${WORK_ROOT}"

for split in training validation; do
    [ -s "ds/${split}.tfrecord" ] || { echo "no dataset for ${split}, skipping" >&2; continue; }

    base="${RENDER_OUT}/ai/ds/${split}.tfrecord/${STEM}/${STEM}"
    manifest="${VIDEO_ROOT}/${STEM}/${split}_failures.csv"
    dest="${VIDEO_ROOT}/${STEM}/${split}"
    mkdir -p "$(dirname "${manifest}")"

    # ── 1. metrics pass at batch_size 1 (same batch layout as rendering) ──────
    if [ -s "${base}/evaluation_episodes.csv" ]; then
        echo "-- ${split}: batch-1 metrics already present, reusing"
    else
        echo "== ${split}: batch-1 metrics pass :: $(date +%H:%M:%S)"
        python -m vmax.scripts.evaluate.evaluate \
            --sdc_actor ai --src_dir "${SHIM_DIR}" --path_model "${STEM}" \
            --path_dataset "ds/${split}.tfrecord" --eval_name "${RENDER_OUT}" \
            --max_num_objects "${MAX_NUM_OBJECTS}" --batch_size 1 --seed "${EVAL_SEED}"
    fi
    [ -s "${base}/evaluation_episodes.csv" ] || ckpt_gate_fail "no batch-1 CSV for ${split}"

    # ── 2. pick the failures ─────────────────────────────────────────────────
    IDX="$(python "${VMAX_ROOT}/slurm/select_failures.py" --mode indexes \
             --csv "${base}/evaluation_episodes.csv" --manifest "${manifest}" \
             --goal-col "${GOAL_RADIUS_COL}")"
    n_fail=$(wc -w <<< "${IDX}")
    echo "== ${split}: ${n_fail} failing scenarios -> ${manifest}"
    if [ "${n_fail}" -eq 0 ]; then
        echo "   nothing to render for ${split}"
        continue
    fi

    # ── 3. render just those ─────────────────────────────────────────────────
    echo "== ${split}: rendering :: $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    python -m vmax.scripts.evaluate.evaluate \
        --sdc_actor ai --src_dir "${SHIM_DIR}" --path_model "${STEM}" \
        --path_dataset "ds/${split}.tfrecord" --eval_name "${RENDER_OUT}" \
        --max_num_objects "${MAX_NUM_OBJECTS}" --batch_size 1 --seed "${EVAL_SEED}" \
        --render true --scenario_indexes ${IDX}

    # ── 4. flatten + label ───────────────────────────────────────────────────
    python "${VMAX_ROOT}/slurm/select_failures.py" --mode label \
        --csv "${base}/evaluation_episodes.csv" --manifest "${manifest}" \
        --mp4-dir "${base}/mp4" --out-dir "${dest}"
    du -sh "${dest}" || true
done

echo "=== done :: $(date) ==="
find "${VIDEO_ROOT}/${STEM}" -name '*.mp4' | wc -l | xargs echo "total videos:"
