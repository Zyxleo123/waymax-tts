#!/usr/bin/env bash
# Stage-0 reference arm: run the vendored Diffusion-ES on the 24 failure scenes
# + control successes, exactly at the ES defaults that generated the failure set.
#
# Behavior-preservation is already proven statically (VENDOR_MANIFEST.sha256);
# this run is a smoke test AND establishes the reference arm all later experiments
# are paired against.
#
# REQUIRED: set CKPT to the diffusion checkpoint used for the ES failure run.
#   export CKPT=/path/to/diffusion/checkpoint
# Optional: SEED (default 0), OUT (default es_baseline/repro/out)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- environment: this diffusion pipeline needs flax with nnx.List (>=0.12).
# mineuih's env 'waymax_rs' (flax 0.12.7) has it; yixiz's 'waymax' (flax 0.10.6)
# does NOT. That env lives on shared scratch (mounted on every node), so we drive
# the run with its absolute python directly -- no fragile cross-user activation.
# Override with PYTHON=/path/to/env/bin/python if needed.
PYTHON="${PYTHON:-/zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python}"
# best-effort activation for CUDA/env vars; harmless if unavailable
CONDA_BASE="${CONDA_BASE:-/opt/miniconda3}"
# conda.sh trips set -e/-u; isolate it. Not required for correctness (we drive via
# the absolute PYTHON and JAX finds the GPU without activation).
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  set +eu
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate /zfsauton/scratch/mineuih/conda_envs/waymax_rs 2>/dev/null || true
  set -eu
fi
"$PYTHON" -c "import flax.nnx as nnx; assert hasattr(nnx,'List')" 2>/dev/null || {
  echo "ERROR: $PYTHON lacks flax nnx.List. Point PYTHON at an env with flax>=0.12"
  echo "       (e.g. /zfsauton/scratch/mineuih/conda_envs/waymax_rs/bin/python)."; exit 1; }

CKPT="${CKPT:-/zfsauton/scratch/mineuih/waymax_rs/checkpoints/pretrain_diffusion/pretrain_diffusion_without_subgoal_20260711_141745/latest}"
SEED="${SEED:-0}"
OUT="${OUT:-$HERE/repro/out}"
LANE_GRAPH_DIR="${LANE_GRAPH_DIR:-/zfsauton/scratch/mineuih/waymax_rs/lane_graphs}"
mkdir -p "$OUT"

# failures + controls per tfrecord (failures first, then controls)
IDX_00000="71,92,113,134,187,252,315,377,0,1,2,3"
IDX_00001="45,62,147,158,200,243,289,360,463,0,1,2,3"
IDX_00002="0,13,20,24,70,94,97,1,2,3,4"

run_one () {
  local tfdir="$1" idx="$2" tag="$3"
  echo "=== ES on $tag : scenarios $idx ==="
  "$PYTHON" "$HERE/experiments/run_baseline.py" \
    --planner es \
    --diffusion_ckpt_path "$CKPT" \
    --tfrecord_dir "$tfdir" \
    --split training \
    --lane_graph_dir "$LANE_GRAPH_DIR" \
    --exp_dir "$OUT" \
    --tag "repro_$tag" \
    --seed "$SEED" \
    --num_worlds 12 \
    --num_scenarios 12 \
    --population_size 64 \
    --resample_timesteps 3 \
    --elite_size 4 \
    --num_es_iterations 3 \
    --start_timestep 10 \
    --replan_interval_steps 10 \
    --scenario_indices "$idx" \
    --save_trajectory \
    --visualize_mode none
}

run_one "$HERE/repro/tf00000" "$IDX_00000" "tf00000"
run_one "$HERE/repro/tf00001" "$IDX_00001" "tf00001"
run_one "$HERE/repro/tf00002" "$IDX_00002" "tf00002"

echo
echo "=== Stage-0 summary (compare against original failure/control labels) ==="
"$PYTHON" "$HERE/repro_compare.py" --out "$OUT"
