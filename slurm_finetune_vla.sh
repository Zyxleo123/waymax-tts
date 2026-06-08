#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate waymax_rs

SCENE_TOKENIZER_CKPT_PATH="${SCENE_TOKENIZER_CKPT_PATH:-}"

python -m train_vla.finetuning_vla --add_eos --scene_tokenizer_ckpt "${SCENE_TOKENIZER_CKPT_PATH}"