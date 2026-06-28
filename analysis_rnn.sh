#!/usr/bin/env bash
#SBATCH --job-name=rnn_analysis
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Drift analysis on every rnn/experiments/exp<i>_rnn_*/ directory.
# Multi-GPU parallel dispatch with skip logic.
#
# Submit:
#   sbatch analysis_rnn.sh 1
#   sbatch --gpus 8 analysis_rnn.sh 1 --force
#   sbatch analysis_rnn.sh 1 --skip_umap --methods normal,replay
#
# Options:
#   --force              Re-analyze even if drift_analysis/ exists
#   --methods M          Comma-separated methods to analyze (default: all)
#   --jobs_per_gpu N     Oversubscription factor (default: 1)
#   --skip_umap          Skip direction UMAP analysis
#   --n_directions N     Number of stimulus directions for UMAP (default: 8)
#   --n_trials_per_dir N Repetitions per direction for UMAP (default: 40)
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

# --------------- argument parsing ---------------
INDICES=()
METHODS_STR=""
FORCE=false
JOBS_PER_GPU=1
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --force)           FORCE=true; shift ;;
        --methods)         METHODS_STR="$2"; shift 2 ;;
        --jobs_per_gpu)    JOBS_PER_GPU="$2"; shift 2 ;;
        --skip_umap)       EXTRA_ARGS+=("$1"); shift ;;
        --n_directions)    EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --n_directions=*)  EXTRA_ARGS+=("$1"); shift ;;
        --n_trials_per_dir)   EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --n_trials_per_dir=*) EXTRA_ARGS+=("$1"); shift ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                INDICES+=("$1"); shift
            else
                echo "Unknown arg: $1"; exit 1
            fi ;;
    esac
done

if [ ${#INDICES[@]} -eq 0 ]; then
    echo "Usage: sbatch [slurm opts] analysis_rnn.sh <i> [<i2> ...] [--force] [--methods M] [options]"
    exit 1
fi

METHODS=()
if [ -n "${METHODS_STR}" ]; then
    IFS=',' read -ra METHODS <<< "${METHODS_STR}"
fi

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
N_SLOTS=$((N_GPUS * JOBS_PER_GPU))

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/rnn"
LOG_DIR="${WORK_DIR}/experiments/logs"
mkdir -p "${LOG_DIR}"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Collect all matching directories
# ------------------------------------------------------------------
all_dirs=()
for IDX in "${INDICES[@]}"; do
    PREFIX="exp${IDX}_rnn_"
    shopt -s nullglob
    dirs=("${WORK_DIR}/experiments/${PREFIX}"*/)
    shopt -u nullglob
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No directories matched: ${WORK_DIR}/experiments/${PREFIX}*"
    else
        all_dirs+=("${dirs[@]}")
    fi
done

if [ ${#all_dirs[@]} -eq 0 ]; then
    echo "No experiment directories matched any provided indices."
    exit 1
fi

is_analyzed() {
    [ -d "$1/drift_analysis" ]
}

echo "=== RNN Drift Analysis ==="
echo "  Indices: ${INDICES[*]}, GPUs=${N_GPUS}, Slots=${N_SLOTS}"
echo "  Directories: ${#all_dirs[@]}"
echo ""

# --------------- parallel dispatcher ---------------
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
        sleep 2
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

# --------------- analysis loop ---------------
n_run=0
n_skip=0

for d in "${all_dirs[@]}"; do
    d="${d%/}"
    base="$(basename "${d}")"

    # Filter by method if --methods specified
    if [ ${#METHODS[@]} -gt 0 ]; then
        match=false
        for m in "${METHODS[@]}"; do
            if [[ "${base}" == *"_${m}_"* ]]; then
                match=true; break
            fi
        done
        if [ "${match}" = false ]; then
            continue
        fi
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
    echo "[gpu${gpu_id} slot${slot}] ${base}"

    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per_slot=$(( n_cores / N_SLOTS ))
        export OMP_NUM_THREADS=$(( per_slot > 1 ? per_slot : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        python analyze_drift.py --exp_dir "${d}" --skip_sample_sim "${EXTRA_ARGS[@]}" \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    SLOT_PIDS[${slot}]=$!
    n_run=$((n_run + 1))
done

wait_all
echo ""
echo "Done. Analyzed=${n_run}, Skipped=${n_skip}"
