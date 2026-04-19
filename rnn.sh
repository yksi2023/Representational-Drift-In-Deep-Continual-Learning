#!/usr/bin/env bash
# Train all CL methods for RNN (Yang et al. sequential learning tasks).
# Results: rnn/experiments/exp<i>_rnn_<method>/
# Timing:  rnn/experiments/exp<i>_rnn_timing.txt
# Usage:   bash rnn.sh <i>
set -euo pipefail
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash rnn.sh <i>   # e.g. 'bash rnn.sh 1' -> exp1_rnn_*"
    exit 1
fi
IDX="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
TIMING="${EXP_ROOT}/exp${IDX}_rnn_timing.txt"
mkdir -p "${EXP_ROOT}"; : > "${TIMING}"

COMMON=(
    --hidden_size 256
    --sigma_rec 0.05
    --activation softplus
    --w_rec_init diag
    --num_iterations 5000
    --batch_size 1024
    --lr 0.001
    --train_pool_size 50
    --seed 0
)

cd "${WORK_DIR}"

run_one() {
    local name="$1"; shift
    local save_dir="${EXP_ROOT}/exp${IDX}_rnn_${name}"
    echo ""
    echo "==> [exp${IDX}] Training: ${name} -> ${save_dir}"
    local t0; t0=$(date +%s)
    python run_experiment.py "${COMMON[@]}" --save_dir "${save_dir}" "$@"
    printf '%s: %ds\n' "${name}" $(( $(date +%s) - t0 )) | tee -a "${TIMING}"
}

T0=$(date +%s)

run_one normal   --method normal
run_one replay   --method replay --memory_per_task 50 --replay_num_tasks 1
run_one ewc      --method ewc --ewc_lambda 100.0 --fisher_samples 200
run_one lwf      --method lwf --lwf_lambda 1.0 --lwf_temperature 2.0
run_one hypernet --method hypernet --hnet_beta 0.5 --hnet_chunks 10 --hnet_hidden 128

printf 'TOTAL: %ds\n' $(( $(date +%s) - T0 )) | tee -a "${TIMING}"
echo "Done. Run 'bash analysis_rnn.sh ${IDX}' for drift analysis."
