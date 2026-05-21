#!/bin/bash
source ~/.bashrc
conda activate waymax_rs

DEVICE="${1:-0}"
OUTPUT_DIR=/zfsauton/scratch/mineuih/waymax_rs/qa_dataset_new/

mkdir -p "$OUTPUT_DIR"

count_json_files() {
	find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' | wc -l
}

current_count=$(count_json_files)
target_count=1000

while [[ "$current_count" -lt "$target_count" ]]; do
	echo "Current JSON count: $current_count / $target_count"
	export CUDA_VISIBLE_DEVICES="$DEVICE"
	echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
	python language/generate_qa_dataset.py \
		--tfrecord_dir /zfsauton/scratch/mineuih/womd/training/ \
		--lane_graph_dir /zfsauton/scratch/mineuih/waymax_rs/lane_graphs/ \
		--output_dir "$OUTPUT_DIR" \
		--max_tfrecords 5

	new_count=$(count_json_files)
	if [[ "$new_count" -le "$current_count" ]]; then
		echo "JSON count did not increase after a run: still $new_count"
		exit 1
	fi
	current_count="$new_count"
done

echo "Done. JSON count reached $current_count."