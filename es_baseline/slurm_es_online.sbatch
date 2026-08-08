#!/bin/bash
#SBATCH --job-name=es_online
#SBATCH --partition=general
#SBATCH --nodelist=gpu26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#
# Re-run of the online-SAC ES arm after job 25871 died silently (FAILED 1:0,
# MaxRSS 7.2G of a 19.5G host limit, no traceback, no diagnostics json).
#
# The likely cause is GPU memory, not host memory: 25871 landed on gpu17, an
# rtx_2080_ti with 11 GB, while the three passing smoke runs were on gpu27, an
# a6000 with 48 GB — and the real arm is 12 worlds x k=64 against the smokes'
# 1 world x k=16. A CUDA/XLA allocation failure can take the process down without
# a Python traceback, which matches the silent death.
#
# Node picked from `myfree` (2026-08-08 re-run): gpu26, partition general, a6000,
# 2/8 GPUs free, AVAILABLE (gpu31 is now full). This re-run also carries the
# runner.py scene-relabel fix: the previous online results were mislabeled because
# load_scenario_state_batch_fast returns the batch sorted while runner.py labeled
# it in requested (failure-set-first) order.
# The a6000 is the point — do not let this fall back onto a 2080 Ti node.
#
# Remaining differences from slurm_es.sbatch, kept as cheap discriminators:
#   * --mem=64G --cpus-per-task=8  -> if it still dies, host memory is ruled out
#   * two heartbeats, one on scratch and one on node-local /tmp. If the scratch
#     heartbeat freezes while the local one keeps ticking, the filesystem is the
#     culprit and the missing log tail was never writable in the first place.
#
# Submit from the repo root (/zfsauton2/home/yixiz/waymax_rs), on rhea:
#
#   SCRIPT=es_baseline/experiments/run_sac_online_arms.sh sbatch es_baseline/slurm_es_online.sbatch
#
# Cheap end-to-end variant (4 worlds, 4 scenes) before committing to the full list:
#
#   SCRIPT=es_baseline/experiments/run_sac_online_arms.sh MAX_WORLDS=4 SHARDS=tf00000 \
#     SCEN_IDX_tf00000="71,92,113,134" sbatch es_baseline/slurm_es_online.sbatch

# set -e lives in the job body, not only in a sourced preamble: a failed source
# must not leave the job running the system python and still exiting 0.
set -euo pipefail

# Slurm copies this script to /var/spool/slurm/, so it cannot find the repo
# relative to itself — submit from the repo root.
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

LOGDIR=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs
mkdir -p "$LOGDIR"

SCRIPT="${SCRIPT:-es_baseline/experiments/run_sac_online_arms.sh}"
export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1   # stream python stdout instead of flushing only on exit
export OUT="${OUT:-/zfsauton/scratch/yixiz/waymax_rs/es_baseline/experiments/out}"
mkdir -p "$OUT/_heartbeat"

JOB="${SLURM_JOB_ID:-manual}"
HB_SCRATCH="$OUT/_heartbeat/${JOB}.scratch.hb"
HB_LOCAL="/tmp/es_online_${JOB}.local.hb"

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}  script=$SCRIPT  root=$ROOT"
echo "job=$JOB  mem=${SLURM_MEM_PER_NODE:-unset}M  cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "heartbeat(scratch)=$HB_SCRATCH"
echo "heartbeat(local)  =$HB_LOCAL   (on $(hostname), fetch it if the scratch one stalls)"
df -h /zfsauton/scratch | tail -1

# Both heartbeats tick from the same loop, so a divergence between them is a
# property of the filesystem and not of the job's progress. Failures are
# swallowed here (|| true) so a dead scratch mount cannot kill the loop itself.
heartbeat() {
  while true; do
    ts="$(date -Is)"
    { echo "$ts rss=$(ps -o rss= -p $$ 2>/dev/null | tr -d ' ')" >>"$HB_LOCAL"; } 2>/dev/null || true
    { echo "$ts" >>"$HB_SCRATCH"; } 2>/dev/null || true
    sleep 20
  done
}
heartbeat &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

marker() {  # marker <name>: a dated file on scratch, independent of stdout
  { date -Is >"$OUT/_heartbeat/${JOB}.$1"; } 2>/dev/null \
    || echo "WARNING: could not write marker $1 to $OUT/_heartbeat (scratch write failing)"
}

marker started
"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"
marker jax_ok

# The arms script is allowed to fail without taking the shell down, so that the
# exit status is reported explicitly rather than inferred from a truncated log.
rc=0
bash "$SCRIPT" || rc=$?
marker "script_exit_${rc}"

echo "=== $SCRIPT exited rc=$rc at $(date -Is) ==="
if [[ "$rc" -ne 0 ]]; then
  echo "markers written:"
  ls -l "$OUT/_heartbeat/" | grep "$JOB" || true
  echo "last scratch heartbeat: $(tail -1 "$HB_SCRATCH" 2>/dev/null || echo UNREADABLE)"
  echo "last local   heartbeat: $(tail -1 "$HB_LOCAL" 2>/dev/null || echo UNREADABLE)"
fi
# Preserve the local heartbeat next to the scratch one while the node is still ours.
cp "$HB_LOCAL" "$OUT/_heartbeat/${JOB}.local.hb" 2>/dev/null || true
exit "$rc"
