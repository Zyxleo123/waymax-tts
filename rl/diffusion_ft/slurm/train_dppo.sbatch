#!/bin/bash
#SBATCH --job-name=dft_dppo
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#SBATCH --error=/zfsauton2/home/yixiz/waymax_rs/logs/dft/slurm-%x-%j.out
#
# DPPO fine-tuning of the diffusion policy. `general` hosts only a6000/a5000
# (>=24 GB), pinning the GPU class away from the 11 GB 2080 Tis.
#
# Submit via the companion submitter (writes a timestamped env file, passes its
# path as $1). Never use `sbatch --export` (requeue-hold on this cluster).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"
mkdir -p "$ROOT/logs/dft"

for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    export "${arg?}"
  fi
done

export PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/zfsauton/scratch/yixiz/waymax_rs/.jax_compilation_cache}"

# Config (overridable via the env file / positional VAR=value args).
DFT_INDICES="${DFT_INDICES:-0,1,2,3,4,5,6,7}"
DFT_ITERS="${DFT_ITERS:-50}"
DFT_EXPERT_WEIGHT="${DFT_EXPERT_WEIGHT:-0.0}"
DFT_K_TRAINABLE="${DFT_K_TRAINABLE:-10}"
DFT_PREFIX_LEN="${DFT_PREFIX_LEN:-5}"
DFT_ACTOR_LR="${DFT_ACTOR_LR:-3e-5}"
DFT_PPO_CLIP="${DFT_PPO_CLIP:-0.1}"
DFT_GAE_LAMBDA="${DFT_GAE_LAMBDA:-1.0}"   # 1.0 = Monte-Carlo advantage
DFT_PPO_EPOCHS="${DFT_PPO_EPOCHS:-4}"
DFT_VALUE_EPOCHS="${DFT_VALUE_EPOCHS:-10}"
DFT_PG_SCALE="${DFT_PG_SCALE:-0}"   # 0 = auto (channels*prefix_len)
DFT_SEED="${DFT_SEED:-0}"
DFT_OUT_DIR="${DFT_OUT_DIR:-/zfsauton/scratch/yixiz/waymax_rs/diffusion_ft/dppo}"
DFT_CKPT="${DFT_CKPT:-/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260616_013642/latest}"
DFT_TFRECORD="${DFT_TFRECORD:-/zfsauton/scratch/eshau/womd/tf_example/training/training_tfexample.tfrecord-00000-of-01000}"

echo "host=${SLURMD_NODENAME:-unknown} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "indices=$DFT_INDICES iters=$DFT_ITERS expert_weight=$DFT_EXPERT_WEIGHT k=$DFT_K_TRAINABLE"
echo "actor_lr=$DFT_ACTOR_LR ppo_clip=$DFT_PPO_CLIP gae_lambda=$DFT_GAE_LAMBDA"
echo "ppo_epochs=$DFT_PPO_EPOCHS value_epochs=$DFT_VALUE_EPOCHS pg_scale=$DFT_PG_SCALE"

# The out_dir lives on a quota'd filesystem; job 27279 died mid-run on
# `OSError: [Errno 122] Disk quota exceeded` writing train_log.jsonl. Fail fast
# and loudly here rather than 27 iterations in.
mkdir -p "$DFT_OUT_DIR"
if ! (echo probe > "$DFT_OUT_DIR/.write_probe" && rm -f "$DFT_OUT_DIR/.write_probe"); then
  echo "FATAL: cannot write to DFT_OUT_DIR=$DFT_OUT_DIR (quota?). Not starting." >&2
  exit 1
fi

"$PYTHON" -c "import jax; print('jax devices:', jax.devices())"

"$PYTHON" -m rl.diffusion_ft.train --method dppo \
  --ckpt "$DFT_CKPT" --tfrecord "$DFT_TFRECORD" --indices "$DFT_INDICES" \
  --iters "$DFT_ITERS" --expert_weight "$DFT_EXPERT_WEIGHT" \
  --k_trainable "$DFT_K_TRAINABLE" --prefix_len "$DFT_PREFIX_LEN" \
  --actor_lr "$DFT_ACTOR_LR" --ppo_clip "$DFT_PPO_CLIP" \
  --gae_lambda "$DFT_GAE_LAMBDA" --ppo_epochs "$DFT_PPO_EPOCHS" \
  --value_epochs "$DFT_VALUE_EPOCHS" --pg_scale "$DFT_PG_SCALE" \
  --seed "$DFT_SEED" --out_dir "$DFT_OUT_DIR"
