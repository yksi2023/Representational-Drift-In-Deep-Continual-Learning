#!/usr/bin/env bash
# Train all CL methods for CNN on CIFAR-100 using a from-scratch
# CIFAR-stem ResNet18 with GroupNorm + Weight Standardization + Zero-gamma.
# Results: cnn/experiments/exp<i>_cnn_<method>/
# Timing:  cnn/experiments/exp<i>_cnn_timing.txt
# Usage:   bash cifar_cnn.sh <i>
set -euo pipefail
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash cifar_cnn.sh <i>   # e.g. 'bash cifar_cnn.sh 2' -> exp2_cnn_*"
    exit 1
fi
IDX="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
TIMING="${EXP_ROOT}/exp${IDX}_cnn_timing.txt"
mkdir -p "${EXP_ROOT}"; : > "${TIMING}"

# CIFAR-100 (100 classes, 10 tasks x 10 classes), from-scratch CIFAR-stem ResNet18 (GN+WS+Zero-gamma).
COMMON=(
    --dataset cifar100
    --model resnet18_cifar_gn
    --increment 10
    --epochs 100
    --batch_size 128
    --optimizer sgd
    --lr 0.1
    --momentum 0.9
    --scheduler cosine      # cosine anneal over the 100 epochs of each task
    --patience 100          # effectively disable early stopping for from-scratch runs
    --amp
    --channels_last
)

cd "${WORK_DIR}"

run_one() {
    local name="$1"; shift
    local save_dir="${EXP_ROOT}/exp${IDX}_cnn_${name}"
    echo ""
    echo "==> [exp${IDX}] Training: ${name} -> ${save_dir}"
    local t0; t0=$(date +%s)
    python run_experiment.py "${COMMON[@]}" --save_dir "${save_dir}" "$@"
    printf '%s: %ds\n' "${name}" $(( $(date +%s) - t0 )) | tee -a "${TIMING}"
}

T0=$(date +%s)

run_one normal    --method normal
run_one replay    --method replay --memory_per_class 200
run_one ewc       --method ewc    --ewc_lambda 500.0
run_one lwf       --method lwf    --lwf_lambda 1.0 --lwf_temperature 2.0
run_one gpm       --method gpm

printf 'TOTAL: %ds\n' $(( $(date +%s) - T0 )) | tee -a "${TIMING}"
echo "Done. Run 'bash analysis_cnn.sh ${IDX}' for drift analysis."
