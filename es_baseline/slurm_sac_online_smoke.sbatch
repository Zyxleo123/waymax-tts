#!/bin/bash
#SBATCH --job-name=sac_online_smoke
#SBATCH --partition=debug
#SBATCH --nodelist=gpu27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs/slurm-%x-%j.out
#
# Smoke test for the online SAC init roller (experiments/sac_online.py):
# loads the V-Max SAC checkpoint, rolls K episodes from a spliced ES state, and
# asserts the rollouts anchor on the live ego pose and are actually diverse.
#
#   cd /zfsauton2/home/yixiz/waymax_rs
#   sbatch es_baseline/slurm_sac_online_smoke.sbatch
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p /zfsauton/scratch/yixiz/waymax_rs/es_baseline/logs

# Same interpreter as the ES arms: it has flax>=0.12 (nnx) for the diffusion policy,
# and now imports V-Max too (see the optional-dependency guards in V-Max/vmax).
PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
SAC_RUN_DIR="${SAC_RUN_DIR:-/zfsauton/scratch/yixiz/waymax_rs/vmax_womd_raw_parity/womd_raw_parity_sac_lq}"
SAC_MODEL="${SAC_MODEL:-$SAC_RUN_DIR/model/model_best.pkl}"   # step 1797120
K="${K:-16}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-50}"
SCEN="${SCEN:-71}"
TIMESTEPS="${TIMESTEPS:-10,40}"
ACTION_NOISE="${ACTION_NOISE:-0,0.05,0.1,0.2}"   # swept in one job; first entry drives the checks

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "run_dir=$SAC_RUN_DIR  model=$SAC_MODEL  k=$K  steps=$ROLLOUT_STEPS  scen=$SCEN"

"$PYTHON" es_baseline/experiments/smoke_sac_online.py \
  --sac_run_dir "$SAC_RUN_DIR" \
  --sac_model "$SAC_MODEL" \
  --scenario_index "$SCEN" \
  --k "$K" \
  --rollout_steps "$ROLLOUT_STEPS" \
  --timesteps "$TIMESTEPS" \
  --action_noise "$ACTION_NOISE"
