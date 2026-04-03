#!/usr/bin/env bash
# ==============================================================================
# Run all RNN sequential learning experiments (one per CL method).
# Results are saved under rnn/experiments/<method_name>/.
# A timing summary is printed at the end and saved to rnn/experiments/timing.txt.
#
# Usage:  bash run_rnn_experiments.sh
# ==============================================================================
set -euo pipefail
source activate drift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RNN_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${RNN_DIR}/experiments"
TIMING_FILE="${EXP_ROOT}/exp2_rnn_timing.txt"

mkdir -p "${EXP_ROOT}"
> "${TIMING_FILE}"   # clear previous timing

# ── Shared hyper-parameters ──────────────────────────────────────────────────
COMMON_ARGS=(
    --hidden_size 256
    --sigma_rec 0.05
    --activation softplus
    --w_rec_init diag
    --num_iterations 20000
    --batch_size 256
    --lr 0.001
    --train_pool_size 50
    --seed 0
)

# ── Helper: run one experiment and record wall time ──────────────────────────
run_one() {
    local name="$1"; shift
    local save_dir="${EXP_ROOT}/exp2_rnn_${name}"
    echo ""
    echo "================================================================"
    echo "  Starting: ${name}"
    echo "  Save dir: ${save_dir}"
    echo "================================================================"

    local start end elapsed
    start=$(date +%s)

    python run_experiment.py "${COMMON_ARGS[@]}" --save_dir "${save_dir}" "$@"

    end=$(date +%s)
    elapsed=$(( end - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    local msg="${name}: ${mins}m ${secs}s (${elapsed}s total)"
    echo "${msg}"
    echo "${msg}" >> "${TIMING_FILE}"
}

# ── Switch to rnn/ so relative imports work ──────────────────────────────────
cd "${RNN_DIR}"

TOTAL_START=$(date +%s)

# 1. Replay
run_one "replay" \
    --method replay \
    --memory_per_task 50 \
    --replay_num_tasks 1

# 2. HyperNet
run_one "hypernet" \
    --method hypernet \
    --hnet_beta 0.01 \
    --hnet_chunks 10 \
    --hnet_hidden 128

# 3. EWC
run_one "ewc" \
    --method ewc \
    --ewc_lambda 100.0 \
    --fisher_samples 200

# 4. Baseline (no CL regularization)
run_one "normal" \
    --method normal

# 5. LwF
run_one "lwf" \
    --method lwf \
    --lwf_lambda 1.0 \
    --lwf_temperature 2.0

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "================================================================"
echo "  All experiments finished!"
echo "  Total wall time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "================================================================"
echo ""
echo "── Timing summary ──"
cat "${TIMING_FILE}"
echo "TOTAL: ${TOTAL_MINS}m ${TOTAL_SECS}s" >> "${TIMING_FILE}"
echo ""
echo "Results saved under: ${EXP_ROOT}/"
echo "Timing log: ${TIMING_FILE}"
