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

LR="${LR:-1e-6}"
WANDB_NAME="${WANDB_NAME:-finetuning_overtake-${LR}}"

CUDA_VISIBLE_DEVICES=0 python -m train.train_diffusion \
  --tfrecord_path /data/user_data/mineuih/waymax_rs/finetuning_data/overtake/finetuning_tfexample.tfrecord@1 \
  --seed 42 \
  --batch_size 128 \
  --shuffle_buffer_size 2048 \
  --dataset_num_shards 2 \
  --epochs 1000 \
  --steps_per_epoch 1000 \
  --save_every 10 \
  --save_dir /data/user_data/mineuih/waymax_rs/checkpoints \ \
  --warmup_steps 0 \
  --lr 1e-6 \
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
  --wandb_name finetuning_overtake-1e-5 \
  --wandb_mode online \
  --jax_compilation_cache_dir .jax_compilation_cache \
  # --resume_path /data/user_data/mineuih/checkpoints/diffusion_lr-0p0001_20260301_215223/epoch_0420
