#!/bin/sh

source ~/.bashrc
conda activate es

CUDA_VISIBLE_DEVICES=0 python -m train.train_diffusion \
  --tfrecord_path /zfsauton/datasets/WOMD/tf_example/tf_example/training/training_tfexample.tfrecord@1000 \
  --seed 42 \
  --batch_size 2048 \
  --shuffle_buffer_size 2048 \
  --dataset_num_shards 2 \
  --epochs 500 \
  --steps_per_epoch 50 \
  --save_every 100 \
  --save_dir /zfsauton/scratch/eshau/checkpoints \
  --warmup_steps 2000 \
  --lr 0.00001 \
  --ema_update_every 4 \
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