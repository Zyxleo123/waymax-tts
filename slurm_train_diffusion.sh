#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"

source ~/.bashrc
conda activate waymax_rs

python -m train_diffusion.train_diffusion_policy --pretrained_ckpt "${PRETRAINED_CKPT}"