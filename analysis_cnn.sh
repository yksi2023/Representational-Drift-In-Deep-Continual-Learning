#!/usr/bin/env bash
#SBATCH --job-name=cnn_analysis
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Drift analysis on every cnn/experiments/exp<i>_cnn_*/ directory.
# Multi-GPU parallel dispatch with skip logic.
#
# Submit:
#   sbatch analysis_cnn.sh 1
#   sbatch --gpus 2 analysis_cnn.sh 1 --force
#   sbatch analysis_cnn.sh 1 --layers layer3,layer4
#
# Options:
#   --force      Re-analyze even if drift_analysis/metrics.json exists
#   --layers L   Override layer names (comma-separated)
#   --methods M  Comma-separated methods to analyze (e.g. replay,ewc). Default: all
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

# --------------- argument parsing ---------------
INDICES=()
LAYERS_OVERRIDE=""
METHODS_STR=""
FORCE=false
JOBS_PER_GPU=1

while [ $# -gt 0 ]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        --layers)        LAYERS_OVERRIDE="$2"; shift 2 ;;
        --methods)       METHODS_STR="$2"; shift 2 ;;
        --jobs_per_gpu)  JOBS_PER_GPU="$2"; shift 2 ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                INDICES+=("$1"); shift
            else
                echo "Unknown arg: $1"; exit 1
            fi ;;
    esac
done

if [ ${#INDICES[@]} -eq 0 ]; then
    echo "Usage: sbatch [slurm opts] analysis_cnn.sh <i> [<i2> ...] [--force] [--layers L] [--methods M]"
    exit 1
fi

# Parse methods filter
METHODS=()
if [ -n "${METHODS_STR}" ]; then
    IFS=',' read -ra METHODS <<< "${METHODS_STR}"
fi

# Detect GPUs from SLURM allocation (fallback: nvidia-smi)
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
N_SLOTS=$((N_GPUS * JOBS_PER_GPU))

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Collect all matching directories
# ------------------------------------------------------------------
all_dirs=()
for IDX in "${INDICES[@]}"; do
    PREFIX="exp${IDX}_cnn_"
    shopt -s nullglob
    dirs=("${EXP_ROOT}/${PREFIX}"*/)
    shopt -u nullglob
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No directories matched: ${EXP_ROOT}/${PREFIX}*"
    else
        all_dirs+=("${dirs[@]}")
    fi
done

if [ ${#all_dirs[@]} -eq 0 ]; then
    echo "No experiment directories matched any provided indices."
    exit 1
fi

# Pick default probe layers based on the model in experiment_config.json.
default_layers_for() {
    case "$1" in
        resnet18_pretrained)
            echo "backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4" ;;
        resnet18_tiny|resnet18_cifar_gn|resnet18_tiny_gn)
            echo "layer1,layer2,layer3,layer4" ;;
        bit_s_r50x1_in1k)
            echo "backbone.stages.0,backbone.stages.1,backbone.stages.2,backbone.stages.3,backbone.norm" ;;
        *)
            echo "" ;;
    esac
}

is_analyzed() {
    [ -f "$1/drift_analysis/metrics.json" ]
}

echo "=== CNN Drift Analysis ==="
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

    cfg="${d}/experiment_config.json"
    if [ -n "${LAYERS_OVERRIDE}" ]; then
        layers="${LAYERS_OVERRIDE}"
    elif [ -f "${cfg}" ]; then
        model=$(python -c "import json; print(json.load(open('${cfg}')).get('model', ''))")
        layers=$(default_layers_for "${model}")
        if [ -z "${layers}" ]; then
            echo "[skip] ${base} (unknown model '${model}')"
            continue
        fi
    else
        echo "[skip] ${base} (no config)"
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
        python analyze_drift.py \
            --ckpt_dir "${d}" \
            --layers "${layers}" \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    SLOT_PIDS[${slot}]=$!
    n_run=$((n_run + 1))
done

wait_all
echo ""
echo "Done. Analyzed=${n_run}, Skipped=${n_skip}"
