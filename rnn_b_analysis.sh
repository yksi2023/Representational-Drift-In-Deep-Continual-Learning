#!/usr/bin/env bash
#SBATCH --job-name=expB_rnn_analysis
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Experiment B (RNN): minimal per-run drift analysis plus lambda comparison.
#
# Submit:
#   sbatch rnn_b_analysis.sh 1
#   sbatch rnn_b_analysis.sh 1 --force
#   sbatch rnn_b_analysis.sh 1 --report-only
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: sbatch [slurm opts] rnn_b_analysis.sh <i> [--force | --report-only] [arm_patterns ...]"
    exit 1
fi
IDX="$1"; shift

JOBS_PER_GPU=1
FORCE=false
REPORT_ONLY=false
PATTERNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs_per_gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --report-only) REPORT_ONLY=true; shift ;;
        *) PATTERNS+=("$1"); shift ;;
    esac
done

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
N_SLOTS=$((N_GPUS * JOBS_PER_GPU))

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}b_rnn_"
REPORT_DIR="${EXP_ROOT}/rnn_experiment_b_exp${IDX}_report"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"
cd "${WORK_DIR}"

matches_pattern() {
    local base="$1"
    if [ ${#PATTERNS[@]} -eq 0 ]; then return 0; fi
    local pattern
    for pattern in "${PATTERNS[@]}"; do
        if [[ "${base}" == *"${pattern}"* ]]; then return 0; fi
    done
    return 1
}

shopt -s nullglob
dirs=("${EXP_ROOT}/${PREFIX}"*/)
shopt -u nullglob
if [ ${#dirs[@]} -eq 0 ]; then
    echo "No experiment dirs matched ${EXP_ROOT}/${PREFIX}*"
    exit 0
fi

declare -a SLOT_PIDS=()
for ((slot=0; slot<N_SLOTS; slot++)); do SLOT_PIDS+=(0); done

wait_for_slot() {
    while true; do
        for ((slot=0; slot<N_SLOTS; slot++)); do
            if [ "${SLOT_PIDS[$slot]}" -eq 0 ]; then AVAIL_SLOT=$slot; return; fi
            if ! kill -0 "${SLOT_PIDS[$slot]}" 2>/dev/null; then
                wait "${SLOT_PIDS[$slot]}" 2>/dev/null || true
                SLOT_PIDS[$slot]=0; AVAIL_SLOT=$slot; return
            fi
        done
        sleep 1
    done
}

wait_all() {
    for ((slot=0; slot<N_SLOTS; slot++)); do
        if [ "${SLOT_PIDS[$slot]}" -ne 0 ]; then
            wait "${SLOT_PIDS[$slot]}" 2>/dev/null || true
            SLOT_PIDS[$slot]=0
        fi
    done
}

if [ "${REPORT_ONLY}" = false ]; then
    n_run=0
    n_skip=0
    for dir in "${dirs[@]}"; do
        dir="${dir%/}"
        base="$(basename "${dir}")"
        if ! matches_pattern "${base}"; then continue; fi
        if [ "${FORCE}" = false ] && [ -f "${dir}/drift_analysis/reference_drift_metrics.json" ]; then
            echo "[skip] ${base}"
            n_skip=$((n_skip + 1))
            continue
        fi

        wait_for_slot
        slot=${AVAIL_SLOT}
        gpu_id=$((slot % N_GPUS))
        log_file="${LOG_DIR}/analysis_${base}.log"
        echo "[gpu${gpu_id} slot${slot}] analyze: ${base}"
        (
            export CUDA_VISIBLE_DEVICES=${gpu_id}
            n_cores=$(nproc)
            per_job=$((n_cores / N_SLOTS))
            export OMP_NUM_THREADS=$((per_job > 1 ? per_job : 1))
            export MKL_NUM_THREADS=${OMP_NUM_THREADS}
            export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
            export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
            python analyze_drift.py \
                --exp_dir "${dir}" \
                --skip_model_sim \
                --skip_sample_sim \
                --skip_temporal_sim \
                --skip_vector_drift \
                --skip_subspace_drift \
                --skip_performance \
                --skip_umap \
                > "${log_file}" 2>&1
        ) &
        SLOT_PIDS[${slot}]=$!
        n_run=$((n_run + 1))
    done
    wait_all
    echo "Analyzed ${n_run} run(s), skipped ${n_skip} already done."
fi

echo "==> [exp${IDX}b RNN] Building comparison report..."
python compare_experiment_b.py \
    --exp_root "${EXP_ROOT}" \
    --glob "${PREFIX}*" \
    --out_dir "${REPORT_DIR}"
echo "Done. Results in: ${REPORT_DIR}"
