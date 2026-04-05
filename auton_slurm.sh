#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate es

LR="${LR:-0.001}"
WANDB_NAME="${WANDB_NAME:-womd_lr-${LR}}"

CUDA_VISIBLE_DEVICES=0 python -m train.train_diffusion \
  --tfrecord_path /zfsauton/datasets/womd/tf_example/training/training_tfexample.tfrecord@1000 \
  --seed 42 \
  --batch_size 256 \
  --shuffle_buffer_size 256 \
  --dataset_num_shards 2 \
  --epochs 500 \
  --steps_per_epoch 1000 \
  --save_every 10 \
  --save_dir /zfsauton/scratch/mineuih/waymax_ed/checkpoints \
  --warmup_steps 1500 \
  --lr 0.0001 \
  --ema_update_every 1 \
  --hidden_dim 256 \
  --cond_dim 256 \
  --target_dim 5 \
  --predict_horizon 25 \
  --model_dt 0.2 \
  --log_every 1 \
  --pbar_every 1 \
  --log_jsonl_path ./train_logs.jsonl \
  --wandb_project waymax_diffusion \
  --wandb_name waymax_diffusion \
  --wandb_mode online \
  --jax_compilation_cache_dir .jax_compilation_cache
