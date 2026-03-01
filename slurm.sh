#!/bin/bash
#SBATCH --job-name=womd          # Job name
#SBATCH --partition=general                  # Partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (1 shared-memory job)
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32              # 128 CPU cores for this task
#SBATCH --mem=256G                       # Total memory
#SBATCH --time=24:00:00                 # Walltime (HH:MM:SS)
#SBATCH --output=womd_%j.out        # Standard output
#SBATCH --error=womd_%j.err         # Standard error
# #SBATCH --account=eshau               # Uncomment & set if your cluster needs an account

source ~/.bashrc
conda activate es

CUDA_VISIBLE_DEVICES=0 python -m train.train_diffusion \
  --tfrecord_path /data/datasets/waymo/waymo-open-dataset-v1.3.1/tf_example/training/training_tfexample.tfrecord@1000 \
  --seed 42 \
  --batch_size 4096 \
  --shuffle_buffer_size 4096 \
  --dataset_num_shards 2 \
  --epochs 500 \
  --steps_per_epoch 64 \
  --save_every 10 \
  --save_dir ./train/checkpoints \
  --warmup_steps 960 \
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
  --wandb_project waymax \
  --wandb_mode online \
  --jax_compilation_cache_dir .jax_compilation_cache


