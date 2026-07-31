#!/bin/bash
#SBATCH --job-name=es_smx_build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/ScenarioMaxWaymoES/logs/slurm-%x-%j.out
#
# Path 2 step 1: map ES oracle scenes -> ScenarioMax shard (CPU, no GPU, no TF).
# All artifacts + logs under scratch OUT_DIR.
#
#   sbatch es_baseline/slurm_build_es_scenariomax.sbatch
set -euo pipefail

# Slurm may copy this script to /var/spool/slurm/ — do not derive paths from BASH_SOURCE.
ROOT="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/waymax_rs}"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/ScenarioMaxWaymoES}"
case "$OUT_DIR" in
  /zfsauton/scratch/*|/scratch/*) ;;
  *) echo "ERROR: OUT_DIR must be on scratch, got: $OUT_DIR"; exit 1 ;;
esac
mkdir -p "$OUT_DIR/logs"

# Any python with numpy is fine — script does not import tensorflow.
PYTHON="${PYTHON:-/zfsauton/scratch/yixiz/miniconda3/envs/waymax/bin/python}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

echo "host=$(hostname)  out=$OUT_DIR  python=$PYTHON"
"$PYTHON" es_baseline/experiments/build_es_scenariomax.py --out-dir "$OUT_DIR"

echo "done: $OUT_DIR/es_scenes.tfrecord"
echo "manifest: $OUT_DIR/es_scenes_manifest.json"
ls -lh "$OUT_DIR/es_scenes.tfrecord" "$OUT_DIR/es_scenes_manifest.json"
