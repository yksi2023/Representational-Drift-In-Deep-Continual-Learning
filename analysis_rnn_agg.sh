#!/usr/bin/env bash
#SBATCH --job-name=rnn_aggregate
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Aggregate multi-seed RNN experiments into per-method averaged plots.
# No prior analysis_rnn.sh needed — reads representations/*.npz directly.
#
# Produces per method (averaged across seeds):
#   accuracy_matrix.pdf, pearson_matrix_<probe>.pdf, vector_drift_<probe>.pdf,
#   temporal_correlation/cross_checkpoint_pearson_<probe>[_fix1|_stim1_go1].pdf
#
# Submit:
#   sbatch analysis_rnn_agg.sh 1
#   sbatch analysis_rnn_agg.sh 1 --methods normal,replay --probe fdgo
#
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash analysis_rnn_agg.sh <i> [--methods m1,m2,...] [--probe TASK]"
    exit 1
fi
IDX="$1"; shift

METHODS="normal,replay"
PROBE="fdgo"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --methods)    METHODS="$2"; shift 2 ;;
        --probe)      PROBE="$2";   shift 2 ;;
        --output_dir) EXTRA_ARGS+=("--output_dir" "$2"); shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_rnn_"
OUTPUT_DIR="${EXP_ROOT}/exp${IDX}_aggregate_report"

cd "${WORK_DIR}"

echo "=== RNN Aggregate Analysis ==="
echo "  IDX=${IDX}, Methods=(${METHODS}), Probe=${PROBE}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python aggregate_seeds.py \
    --exp_root "${EXP_ROOT}" \
    --prefix "${PREFIX}" \
    --methods "${METHODS}" \
    --probe "${PROBE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
