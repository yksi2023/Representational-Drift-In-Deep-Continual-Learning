#!/usr/bin/env bash
#SBATCH --job-name=cnn_aggregate
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Aggregate multi-seed CNN experiments into per-method averaged plots.
#
# Produces per-method plots averaged across seeds, including accuracy/model
# similarity matrices, sample similarity (curves + CKA matrix + averaged
# sample-sim heatmaps), reference drift, and gap drift.
#
# Submit:
#   sbatch analysis_cnn_agg.sh 1
#   sbatch --partition gpu_5090 analysis_cnn_agg.sh 1 --methods replay,ewc
#
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash analysis_cnn_agg.sh <i> [--methods m1,m2,...] [--layers l1,l2,...]"
    exit 1
fi
IDX="$1"; shift

METHODS="normal,replay,ewc,lwf,gpm"
LAYERS="layer1,layer2,layer3,layer4"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --methods) METHODS="$2"; shift 2 ;;
        --layers)  LAYERS="$2";  shift 2 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_cnn_"
OUTPUT_DIR="${EXP_ROOT}/exp${IDX}_aggregate_report"

cd "${WORK_DIR}"

echo "=== CNN Aggregate Analysis ==="
echo "  IDX=${IDX}, Methods=(${METHODS}), Layers=(${LAYERS})"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python aggregate_seeds.py \
    --exp_root "${EXP_ROOT}" \
    --prefix "${PREFIX}" \
    --methods "${METHODS}" \
    --layers "${LAYERS}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
