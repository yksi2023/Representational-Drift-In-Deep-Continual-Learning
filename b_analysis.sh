#!/usr/bin/env bash
#SBATCH --job-name=expB_analysis
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Experiment B (CNN) -- parallel analysis + report.
#
# Drift analysis on exp<i>b_cnn_* runs, then the full comparison report.
# Only runs the analysis steps needed by the report (skips UMAP, distributions,
# sample similarity, etc.).
#
# Submit:
#   sbatch b_analysis.sh 1
#   sbatch --partition gpu_5090 --gpus 8 b_analysis.sh 1 --force
#   sbatch b_analysis.sh 1 --report-only
#
# Options:
#   --jobs_per_gpu M   Oversubscription factor (default: 1)
#   --force            Re-analyze even if drift_analysis/ exists
#   --report-only      Skip drift analysis, rebuild report only
#   arm_patterns       Optional substrings to filter dirs (e.g. l0p3 l2 l5)
#
# Skip rule: a run is considered analyzed when drift_analysis/metrics.json exists.
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

if [ $# -lt 1 ]; then
    echo "Usage: sbatch [slurm opts] b_analysis.sh <i> [--force | --report-only] [arm_patterns ...]"
    exit 1
fi
IDX="$1"; shift

JOBS_PER_GPU=1
FORCE=false
REPORT_ONLY=false
PATTERNS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs_per_gpu)  JOBS_PER_GPU="$2"; shift 2 ;;
        --force)         FORCE=true; shift ;;
        --report-only)   REPORT_ONLY=true; shift ;;
        *)               PATTERNS+=("$1"); shift ;;
    esac
done

# Detect GPUs from SLURM allocation (fallback: nvidia-smi)
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
N_SLOTS=$((N_GPUS * JOBS_PER_GPU))

ANALYSIS_LAYERS="layer1,layer2,layer3,layer4"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}b_cnn_"
REPORT_DIR="${EXP_ROOT}/exp${IDX}b_report"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${WORK_DIR}"

matches_pattern() {
    local base="$1"
    if [ ${#PATTERNS[@]} -eq 0 ]; then
        return 0
    fi
    local p
    for p in "${PATTERNS[@]}"; do
        if [[ "${base}" == *"${p}"* ]]; then
            return 0
        fi
    done
    return 1
}

is_analyzed() {
    [ -f "$1/drift_analysis/metrics.json" ]
}

shopt -s nullglob
dirs=("${EXP_ROOT}/${PREFIX}"*/)
shopt -u nullglob
if [ ${#dirs[@]} -eq 0 ]; then
    echo "No experiment dirs matched ${EXP_ROOT}/${PREFIX}*"
    exit 0
fi

# --------------- parallel dispatcher ---------------
# N_SLOTS = N_GPUS * JOBS_PER_GPU concurrent processes.
# Slot s maps to GPU (s % N_GPUS).
declare -a SLOT_PIDS=()
for ((s=0; s<N_SLOTS; s++)); do SLOT_PIDS+=(0); done

wait_for_slot() {
    while true; do
        for ((s=0; s<N_SLOTS; s++)); do
            if [ "${SLOT_PIDS[$s]}" -eq 0 ]; then
                AVAIL_SLOT=$s; return
            fi
            if ! kill -0 "${SLOT_PIDS[$s]}" 2>/dev/null; then
                wait "${SLOT_PIDS[$s]}" 2>/dev/null || true
                SLOT_PIDS[$s]=0
                AVAIL_SLOT=$s; return
            fi
        done
        sleep 1
    done
}

wait_all() {
    for ((s=0; s<N_SLOTS; s++)); do
        if [ "${SLOT_PIDS[$s]}" -ne 0 ]; then
            wait "${SLOT_PIDS[$s]}" 2>/dev/null || true
            SLOT_PIDS[$s]=0
        fi
    done
}

# --------------- analysis ---------------
if [ "${REPORT_ONLY}" = false ]; then
    n_run=0
    n_skip=0
    for d in "${dirs[@]}"; do
        d="${d%/}"
        base="$(basename "${d}")"
        if ! matches_pattern "${base}"; then
            continue
        fi
        if [ "${FORCE}" = false ] && is_analyzed "${d}"; then
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
            # Limit CPU threads per analysis job to avoid oversubscription across N_SLOTS processes.
            n_cores=$(nproc)
            per_job=$(( n_cores / N_SLOTS ))
            export OMP_NUM_THREADS=$(( per_job > 1 ? per_job : 1 ))
            export MKL_NUM_THREADS=${OMP_NUM_THREADS}
            export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
            export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
            export _DATALOADER_NUM_WORKERS=$(( per_job / 2 > 2 ? per_job / 2 : 2 ))
            python analyze_drift.py \
                --ckpt_dir "${d}" \
                --layers "${ANALYSIS_LAYERS}" \
                --skip_umap \
                --skip_distributions \
                --skip_sample_sim \
                --skip_model_sim \
                --skip_gap_drift \
                --skip_performance \
                > "${log_file}" 2>&1
        ) &
        SLOT_PIDS[${slot}]=$!
        n_run=$((n_run + 1))
    done
    wait_all
    echo ""
    echo "Analyzed ${n_run} run(s), skipped ${n_skip} already-done."
fi

echo ""
echo "==> [exp${IDX}b] Building comparison report..."
python compare_experiment_b.py \
    --exp_root "${EXP_ROOT}" \
    --glob "${PREFIX}*" \
    --out_dir "${REPORT_DIR}"
echo "Done. Results in: ${REPORT_DIR}"
