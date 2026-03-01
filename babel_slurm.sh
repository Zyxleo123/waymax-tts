#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate es

LR="${LR:-0.001}"
WANDB_NAME="${WANDB_NAME:-womd_lr-${LR}}"

CUDA_VISIBLE_DEVICES=0 python -m train.train_diffusion \
  --tfrecord_path /data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example/training/training_tfexample.tfrecord@1000 \
  --seed 42 \
  --batch_size 2048 \
  --shuffle_buffer_size 2048 \
  --dataset_num_shards 2 \
  --epochs 250 \
  --steps_per_epoch 100 \
  --save_every 50 \
  --save_dir /data/user_data/eshau/checkpoints \
  --warmup_steps 1500 \
  --lr "${LR}" \
  --ema_update_every 1 \
  --hidden_dim 256 \
  --cond_dim 256 \
  --target_dim 5 \
  --predict_horizon 25 \
  --model_dt 0.2 \
  --log_every 1 \
  --pbar_every 1 \
  --log_jsonl_path ./train_logs.jsonl \
  --wandb_project waymax \
  --wandb_name "${WANDB_NAME}" \
  --wandb_mode online \
  --jax_compilation_cache_dir .jax_compilation_cache
