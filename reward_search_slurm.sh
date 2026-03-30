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

BASELINE_FLAG=""
if [[ "${USE_BASELINE:-true}" == "true" ]]; then
    BASELINE_FLAG="--use_baseline"
fi

PYTHONWARNINGS=ignore python scripts/main.py \
    --task ${TASK:-overtake} \
    --num_worlds ${NUM_WORLDS:-16} \
    --use_candidates \
    ${BASELINE_FLAG} \