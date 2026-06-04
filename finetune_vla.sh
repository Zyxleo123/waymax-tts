#!/usr/bin/env bash

set -euo pipefail

CKPT_PATH="/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_vla/pretrain_vla_gemma_20260603_012414/checkpoints/step_00010000.pt"
CHECK_INTERVAL_SECONDS=60

echo "Waiting for checkpoint: ${CKPT_PATH}"

while true; do
	if [[ -f "${CKPT_PATH}" ]]; then
		echo "Checkpoint found. Starting finetuning."
		python -m train_vla.finetuning_vla --scene_tokenizer_ckpt "${CKPT_PATH}" --add_eos
		break
	fi

	echo "Checkpoint not found yet. Sleeping for ${CHECK_INTERVAL_SECONDS}s."
	sleep "${CHECK_INTERVAL_SECONDS}"
done
