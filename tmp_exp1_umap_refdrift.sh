#!/usr/bin/env bash
#SBATCH --job-name=exp1_umap_ref
#SBATCH --partition=gpu_4090
#SBATCH --gpus=2
#SBATCH --output=logs/%x_%j.out
#
# Temporary: run only UMAP + reference drift for CNN exp1 all seeds/methods.
# Then run aggregate to get reference_drift.pdf.
#
# Submit:
#   sbatch tmp_exp1_umap_refdrift.sh
#   sbatch tmp_exp1_umap_refdrift.sh --methods normal,ewc
#   sbatch tmp_exp1_umap_refdrift.sh --force
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

IDX=1
METHODS=""
FORCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --methods) METHODS="$2"; shift 2 ;;
        --force)   FORCE=true; shift ;;
        *)         shift ;;
    esac
done

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_cnn_"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${WORK_DIR}"

# Discover all matching dirs
shopt -s nullglob
dirs=("${EXP_ROOT}/${PREFIX}"*/)
shopt -u nullglob

if [ ${#dirs[@]} -eq 0 ]; then
    echo "No dirs matching ${EXP_ROOT}/${PREFIX}*"
    exit 0
fi

# Filter by methods if specified
filtered=()
for d in "${dirs[@]}"; do
    d="${d%/}"
    base="$(basename "${d}")"
    if [ -n "${METHODS}" ]; then
        match=false
        IFS=',' read -ra mlist <<< "${METHODS}"
        for m in "${mlist[@]}"; do
            if [[ "${base}" == *"${m}"* ]]; then
                match=true; break
            fi
        done
        if [ "${match}" = false ]; then
            continue
        fi
    fi
    filtered+=("${d}")
done

if [ ${#filtered[@]} -eq 0 ]; then
    echo "No dirs after method filter."
    exit 0
fi

echo "Found ${#filtered[@]} dirs to analyze"

# Parallel dispatch
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
declare -a SLOT_PIDS=()
for ((s=0; s<N_GPUS; s++)); do SLOT_PIDS+=(0); done

wait_for_slot() {
    while true; do
        for ((s=0; s<N_GPUS; s++)); do
            if [ "${SLOT_PIDS[$s]}" -eq 0 ] || ! kill -0 "${SLOT_PIDS[$s]}" 2>/dev/null; then
                wait "${SLOT_PIDS[$s]}" 2>/dev/null || true
                SLOT_PIDS[$s]=0
                AVAIL_SLOT=$s; return
            fi
        done
        sleep 1
    done
}

wait_all() {
    for ((s=0; s<N_GPUS; s++)); do
        [ "${SLOT_PIDS[$s]}" -ne 0 ] && wait "${SLOT_PIDS[$s]}" 2>/dev/null || true
    done
}

n_run=0
n_skip=0
for d in "${filtered[@]}"; do
    base="$(basename "${d}")"

    # Skip if already analyzed (metrics.json exists) unless --force
    if [ "${FORCE}" = false ] && [ -f "${d}/drift_analysis/metrics.json" ]; then
        echo "[skip] ${base}"
        n_skip=$((n_skip + 1))
        continue
    fi

    # Determine layers from config
    cfg="${d}/experiment_config.json"
    if [ ! -f "${cfg}" ]; then
        echo "[skip] ${base} (no config)"
        continue
    fi
    model=$(python -c "import json; print(json.load(open('${cfg}')).get('model', ''))")
    case "${model}" in
        resnet18*) layers="layer1,layer2,layer3,layer4" ;;
        *)         layers="layer1,layer2,layer3,layer4" ;;
    esac

    wait_for_slot
    slot=${AVAIL_SLOT}
    gpu_id=$((slot % N_GPUS))
    log_file="${LOG_DIR}/analysis_${base}.log"
    echo "[gpu${gpu_id}] ${base}"

    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per=$(( n_cores / N_GPUS ))
        export OMP_NUM_THREADS=$(( per > 1 ? per : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

        # Only run: reference drift (step 1, always runs) + UMAP (step 4b)
        # Skip everything else
        python analyze_drift.py \
            --ckpt_dir "${d}" \
            --layers "${layers}" \
            --skip_sample_sim \
            --skip_distributions \
            --skip_model_sim \
            --skip_gap_drift \
            --skip_performance \
            --skip_health \
            --skip_subspace_overlap \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    SLOT_PIDS[${slot}]=$!
    n_run=$((n_run + 1))
done

wait_all
echo ""
echo "Analysis done. Run=${n_run}, Skipped=${n_skip}"

# ── Aggregate ──
echo ""
echo "==> Running aggregate (reference drift)..."

# Determine which methods exist
if [ -n "${METHODS}" ]; then
    AGG_METHODS="${METHODS}"
else
    # Auto-detect methods from directory names
    AGG_METHODS=$(python -c "
import os, re, sys
prefix = '${PREFIX}'
root = '${EXP_ROOT}'
methods = set()
for d in os.listdir(root):
    if d.startswith(prefix) and os.path.isdir(os.path.join(root, d)):
        rest = d[len(prefix):]
        m = re.match(r'(\w+?)_seed', rest)
        if m:
            methods.add(m.group(1))
print(','.join(sorted(methods)))
")
fi

echo "Methods: ${AGG_METHODS}"

python aggregate_seeds.py \
    --exp_root "${EXP_ROOT}" \
    --prefix "${PREFIX}" \
    --methods "${AGG_METHODS}" \
    --layers layer3,layer4 \
    --output_dir "${EXP_ROOT}/exp${IDX}_aggregate_report"

echo ""
echo "Done. Check ${EXP_ROOT}/exp${IDX}_aggregate_report/"
