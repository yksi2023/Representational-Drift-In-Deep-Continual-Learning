#!/usr/bin/env bash
#SBATCH --job-name=ewc_sweep_b
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# EWC lambda sweep B (4090): interleaved values from 1e4–1e8
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

IDX="${1:-1}"
SEED_STR="${2:-0}"

IFS=',' read -ra SEEDS <<< "${SEED_STR}"

# Interleaved lambdas (not a contiguous subrange)
LAMBDAS=(15000 75000 350000 1000000 3500000 10000000 35000000 100000000)

WORK_DIR="/data/run01/scxk458/drift/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_cnn_"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}"

COMMON=(
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
)

cd "${WORK_DIR}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))

# --------------- build job list ---------------
declare -a JOBS=()
for s in "${SEEDS[@]}"; do
    for L in "${LAMBDAS[@]}"; do
        JOBS+=("ewc_l${L}|${s}|--ewc_lambda ${L}")
    done
done

N_PARALLEL=$(( ${#JOBS[@]} < N_GPUS ? ${#JOBS[@]} : N_GPUS ))
N_PARALLEL=$(( N_PARALLEL > 0 ? N_PARALLEL : 1 ))

echo "=== EWC Sweep B (4090) ==="
echo "  IDX=${IDX}, GPUs=${N_GPUS}, Seeds=(${SEED_STR})"
echo "  Lambdas: ${LAMBDAS[*]}"
echo "  Total jobs: ${#JOBS[@]}, max parallel: ${N_PARALLEL}"
echo ""

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

T0=$(date +%s)
n_launched=0
n_skipped=0

for job_spec in "${JOBS[@]}"; do
    IFS='|' read -r name seed extra <<< "${job_spec}"
    save_dir="${EXP_ROOT}/${PREFIX}${name}_seed${seed}"

    if [ -f "${save_dir}/comprehensive_evaluation.json" ]; then
        echo "[skip] ${name} seed${seed}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    wait_for_slot
    gpu_id=${AVAIL_GPU}
    log_file="${LOG_DIR}/${PREFIX}${name}_seed${seed}.log"

    echo "[gpu${gpu_id}] ${name} seed${seed}  -> ${log_file}"
    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per_job=$(( n_cores / N_PARALLEL ))
        export OMP_NUM_THREADS=$(( per_job > 1 ? per_job : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        export _DATALOADER_NUM_WORKERS=$(( per_job / 2 > 2 ? per_job / 2 : 2 ))
        python run_experiment.py "${COMMON[@]}" --seed "${seed}" --save_dir "${save_dir}" ${extra} \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
    GPU_PIDS[${gpu_id}]=$!
    n_launched=$((n_launched + 1))
done

wait_all
printf 'TOTAL: %ds  (launched=%d, skipped=%d)\n' $(( $(date +%s) - T0 )) "${n_launched}" "${n_skipped}"
echo "Done."
