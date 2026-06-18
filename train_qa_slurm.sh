#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate waymax_rs

LR="${LR:-0.0001}"

python -m vla.train_qa