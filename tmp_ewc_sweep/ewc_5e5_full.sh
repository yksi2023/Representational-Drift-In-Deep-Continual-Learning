#!/usr/bin/env bash
#SBATCH --job-name=ewc_5e5
#SBATCH --partition=gpu_4090
#SBATCH --gpus=2
#SBATCH --output=logs/%x_%j.out
#
# EWC lambda=5e5: train -> analyze -> aggregate
# Usage:
#   sbatch ewc_5e5_full.sh 9 0,1,2,3,4    # first batch
#   sbatch ewc_5e5_full.sh 9 5,6,7,8,9    # second batch
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

IDX="${1:-9}"
SEED_STR="${2:-0,1,2,3,4}"

IFS=',' read -ra SEEDS <<< "${SEED_STR}"
LAMBDA=500000

WORK_DIR="/data/run01/scxk458/drift/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_cnn_"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}"

COMMON_TRAIN=(
    --dataset imagenet1k
    --model resnet18_tiny_gn
    --num_classes 100
    --img_size 224
    --increment 5
    --epochs 60
    --batch_size 256
    --optimizer sgd
    --lr 0.1
    --patience 10
    --scheduler cosine
    --channels_last
    --method ewc
    --ewc_protect all
    --ewc_lambda ${LAMBDA}
)

LAYERS="layer1,layer2,layer3,layer4"

# Analysis: skip slow steps (sample_sim, UMAP, distributions, health, subspace_overlap)
# Keep: reference drift (metrics.json), model sim (.npy), gap drift, performance plots
COMMON_ANALYSIS=(
    --layers "${LAYERS}"
    --skip_distributions
    --skip_sample_sim
    --skip_umap
    --skip_health
    --skip_subspace_overlap
)

cd "${WORK_DIR}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))

# --------------- parallel dispatcher ---------------
declare -a GPU_PIDS=()
for ((g=0; g<N_GPUS; g++)); do GPU_PIDS+=(0); done

wait_for_slot() {
    while true; do
        for ((g=0; g<N_GPUS; g++)); do
            if [ "${GPU_PIDS[$g]}" -eq 0 ]; then
                AVAIL_GPU=$g; return
            fi
            if ! kill -0 "${GPU_PIDS[$g]}" 2>/dev/null; then
                wait "${GPU_PIDS[$g]}" 2>/dev/null || true
                GPU_PIDS[$g]=0
                AVAIL_GPU=$g; return
            fi
        done
        sleep 2
    done
}

wait_all() {
    for ((g=0; g<N_GPUS; g++)); do
        if [ "${GPU_PIDS[$g]}" -ne 0 ]; then
            wait "${GPU_PIDS[$g]}" 2>/dev/null || true
            GPU_PIDS[$g]=0
        fi
    done
}

# ==================== PHASE 1: TRAINING ====================
echo "=========================================="
echo "PHASE 1: Training EWC lambda=${LAMBDA}, ${#SEEDS[@]} seeds"
echo "=========================================="

T0=$(date +%s)
n_launched=0
n_skipped=0

for s in "${SEEDS[@]}"; do
    save_dir="${EXP_ROOT}/${PREFIX}ewc_l${LAMBDA}_seed${s}"

    if [ -f "${save_dir}/comprehensive_evaluation.json" ]; then
        echo "[skip train] ewc_l${LAMBDA} seed${s}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    wait_for_slot
    gpu_id=${AVAIL_GPU}
    log_file="${LOG_DIR}/${PREFIX}ewc_l${LAMBDA}_seed${s}_train.log"

    echo "[gpu${gpu_id}] train ewc_l${LAMBDA} seed${s}"
    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per_job=$(( n_cores / N_GPUS ))
        export OMP_NUM_THREADS=$(( per_job > 1 ? per_job : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        python run_experiment.py "${COMMON_TRAIN[@]}" --seed "${s}" --save_dir "${save_dir}" \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    GPU_PIDS[${gpu_id}]=$!
    n_launched=$((n_launched + 1))
done

wait_all
echo "Training done. launched=${n_launched}, skipped=${n_skipped}, elapsed=$(($(date +%s) - T0))s"

# ==================== PHASE 2: ANALYSIS ====================
echo ""
echo "=========================================="
echo "PHASE 2: Drift analysis"
echo "=========================================="

T1=$(date +%s)
n_launched=0
n_skipped=0

for s in "${SEEDS[@]}"; do
    ckpt_dir="${EXP_ROOT}/${PREFIX}ewc_l${LAMBDA}_seed${s}"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "[missing] ${ckpt_dir}"
        continue
    fi

    if [ -f "${ckpt_dir}/drift_analysis/metrics.json" ]; then
        echo "[skip analysis] ewc_l${LAMBDA} seed${s}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    wait_for_slot
    gpu_id=${AVAIL_GPU}
    log_file="${LOG_DIR}/${PREFIX}ewc_l${LAMBDA}_seed${s}_analysis.log"

    echo "[gpu${gpu_id}] analyze ewc_l${LAMBDA} seed${s}"
    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per_job=$(( n_cores / N_GPUS ))
        export OMP_NUM_THREADS=$(( per_job > 1 ? per_job : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        python analyze_drift.py --ckpt_dir "${ckpt_dir}" "${COMMON_ANALYSIS[@]}" \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    GPU_PIDS[${gpu_id}]=$!
    n_launched=$((n_launched + 1))
done

wait_all
echo "Analysis done. launched=${n_launched}, skipped=${n_skipped}, elapsed=$(($(date +%s) - T1))s"

# ==================== PHASE 3: AGGREGATE ====================
echo ""
echo "=========================================="
echo "PHASE 3: Aggregate across seeds"
echo "=========================================="

python aggregate_seeds.py \
    --exp_root "${EXP_ROOT}" \
    --prefix "${PREFIX}" \
    --methods ewc_l${LAMBDA} \
    --layers "${LAYERS}" \
    --output_dir "${EXP_ROOT}/aggregate_ewc_l${LAMBDA}"

echo ""
echo "=========================================="
echo "ALL DONE.  Total elapsed: $(($(date +%s) - T0))s"
echo "=========================================="
