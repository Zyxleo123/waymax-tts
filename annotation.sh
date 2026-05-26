#!/bin/bash
source ~/.bashrc
conda activate waymax_rs

DEVICE="${1:-0}"
OUTPUT_DIR=/zfsauton/scratch/mineuih/waymax_rs/annotations/

mkdir -p "$OUTPUT_DIR"

count_json_files() {
	find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.jsonl' | wc -l
}

current_count=$(count_json_files)
target_count=1000

while [[ "$current_count" -lt "$target_count" ]]; do
	echo "Current JSONL count: $current_count / $target_count"
	# export CUDA_VISIBLE_DEVICES="$DEVICE"
	# echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
	python -m language.manual_annotation.annotation \
		--tfrecord_dir /zfsauton/scratch/mineuih/womd/training/ \
		--lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs/ \
		--output_dir "$OUTPUT_DIR" \
		--max_tfrecords 5 \
		--cpu

	new_count=$(count_json_files)
	if [[ "$new_count" -le "$current_count" ]]; then
		echo "JSONL count did not increase after a run: still $new_count"
		exit 1
	fi
	current_count="$new_count"
done

echo "Done. JSONL count reached $current_count."