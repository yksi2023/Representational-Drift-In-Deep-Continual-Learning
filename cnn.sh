#!/usr/bin/env bash
# Train all CL methods for CNN (TinyImageNet + pretrained ResNet18).
# Results: cnn/experiments/exp_cnn_<method>/
# Timing:  cnn/experiments/exp_cnn_timing.txt
# Usage:   bash cnn.sh
set -euo pipefail
source activate drift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
TIMING="${EXP_ROOT}/exp_cnn_timing.txt"
mkdir -p "${EXP_ROOT}"; : > "${TIMING}"

COMMON=(
    --dataset tiny_imagenet
    --increment 10
    --epochs 50
    --batch_size 512
    --optimizer sgd
    --lr 0.4
    --patience 10
    --use_pretrained_resnet
    --freeze_until layer2
    --channels_last
    --amp
)

cd "${WORK_DIR}"

run_one() {
    local name="$1"; shift
    local save_dir="${EXP_ROOT}/exp_cnn_${name}"
    echo ""
    echo "==> Training: ${name} -> ${save_dir}"
    local t0; t0=$(date +%s)
    python run_experiment.py "${COMMON[@]}" --save_dir "${save_dir}" "$@"
    printf '%s: %ds\n' "${name}" $(( $(date +%s) - t0 )) | tee -a "${TIMING}"
}

T0=$(date +%s)

run_one normal    --method normal
run_one replay    --method replay --memory_size 20000
run_one ewc       --method ewc    --ewc_lambda 1000.0
run_one lwf       --method lwf    --lwf_lambda 1.0 --lwf_temperature 2.0
run_one gpm       --method gpm

printf 'TOTAL: %ds\n' $(( $(date +%s) - T0 )) | tee -a "${TIMING}"
echo "Done. Run 'bash analysis_cnn.sh' for drift analysis."
