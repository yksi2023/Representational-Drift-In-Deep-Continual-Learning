#!/usr/bin/env bash
# Train all CL methods for BiT-S R50x1 (IN1k pretrained) on ImageNet-21k-P200.
# Protocol: 20 tasks x 5 classes = 100 classes (out of 200 available).
# Hardware target: single RTX 4090 (24 GB VRAM) with AMP + channels_last.
#
# Prereqs:
#   pretrained_weights/BiT-S-R50x1.npz                 (convert_bit_tf_to_npz.py)
#   cnn/data/imagenet21k_p200/{train,val,test}/<wnid>/ (prepare_imagenet21k_p200.py)
#
# Results: cnn/experiments/exp<i>_cnn_bit_in21kp200_<method>/
# Timing : cnn/experiments/exp<i>_cnn_bit_in21kp200_timing.txt
# Usage  : bash run_imagenet21k_p200.sh <i>
set -euo pipefail
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash run_imagenet21k_p200.sh <i>"
    exit 1
fi
IDX="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
TIMING="${EXP_ROOT}/exp${IDX}_cnn_bit_in21kp200_timing.txt"
mkdir -p "${EXP_ROOT}"; : > "${TIMING}"

COMMON=(
    --dataset imagenet21k_p200
    --model bit_s_r50x1_in1k
    --num_classes 200
    --increment 10
    --img_size 224
    --epochs 30
    --batch_size 128
    --optimizer sgd
    --lr 0.01
    --patience 5
    --channels_last
    --scheduler cosine
)

cd "${WORK_DIR}"

run_one() {
    local name="$1"; shift
    local save_dir="${EXP_ROOT}/exp${IDX}_cnn_bit_in21kp200_${name}"
    echo ""
    echo "==> [exp${IDX}] Training: ${name} -> ${save_dir}"
    local t0; t0=$(date +%s)
    python run_experiment.py "${COMMON[@]}" --save_dir "${save_dir}" "$@"
    printf '%s: %ds\n' "${name}" $(( $(date +%s) - t0 )) | tee -a "${TIMING}"
}

T0=$(date +%s)

# Replay budget: 20 exemplars / class x 100 classes = 2000 total (iCaRL-style).
run_one normal    --method normal
run_one replay    --method replay --memory_per_class 200
run_one ewc       --method ewc    --ewc_lambda 1e6
run_one lwf       --method lwf    --lwf_lambda 1.0 --lwf_temperature 2.0
run_one gpm       --method gpm

printf 'TOTAL: %ds\n' $(( $(date +%s) - T0 )) | tee -a "${TIMING}"
echo "Done. Run 'bash analysis_cnn.sh ${IDX}' for drift analysis."
