#!/usr/bin/env bash
#SBATCH --job-name=cnn_train
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Train all CL methods for CNN on ImageNet-1K subset (ResNet18 GN, from scratch).
# 5 classes per task × 20 tasks = 100 classes (first 100 of 200).
# Dataset prep: python cnn/tools/process_imagenet.py
# Results: cnn/experiments/exp<i>_cnn_<method>_seed<s>/
#
# Submit:
#   sbatch cnn.sh 1 --seeds 0,1,2,3,4
#   sbatch --partition gpu_5090 --gpus 8 cnn.sh 1 --seeds 0,1,2,3,4
#
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

# --------------- argument parsing ---------------
if [ $# -lt 1 ] || [[ "$1" == --* ]]; then
    echo "Usage: sbatch [slurm opts] cnn.sh <i> [--seeds s1,s2,...]"
    echo "  <i> is the experiment index, e.g.: sbatch cnn.sh 1 --seeds 0,1,2,3,4"
    exit 1
fi
IDX="$1"; shift

SEED_STR="42"

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)  SEED_STR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect GPUs from SLURM allocation (fallback: nvidia-smi)
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))

IFS=',' read -ra SEEDS <<< "${SEED_STR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_cnn_"
TIMING="${EXP_ROOT}/exp${IDX}_cnn_timing.txt"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}"
: > "${TIMING}"

COMMON=(
    --dataset imagenet1k
    --model resnet18_tiny_gn
    --num_classes 100
    --img_size 224
    --increment 5
    --epochs 60
    --batch_size 256
    --optimizer sgd
    --lr 0.2
    --patience 10
    --scheduler cosine
    --channels_last
)

cd "${WORK_DIR}"

# --------------- job queue ---------------
declare -a JOBS=()

for s in "${SEEDS[@]}"; do
    JOBS+=("normal|${s}|--method normal")
    JOBS+=("replay|${s}|--method replay --memory_per_class 300")
    JOBS+=("ewc|${s}|--method ewc --ewc_lambda 1e8")
    JOBS+=("lwf|${s}|--method lwf --lwf_lambda 30.0 --lwf_temperature 2.0")
done

# Actual concurrency: never more parallel jobs than there are jobs.
N_PARALLEL=$(( ${#JOBS[@]} < N_GPUS ? ${#JOBS[@]} : N_GPUS ))
N_PARALLEL=$(( N_PARALLEL > 0 ? N_PARALLEL : 1 ))

echo "=== CNN Training ==="
echo "  IDX=${IDX}, GPUs=${N_GPUS}, Seeds=(${SEED_STR})"
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

    # Skip completed runs (comprehensive_evaluation.json is the last file written)
    if [ -f "${save_dir}/comprehensive_evaluation.json" ]; then
        echo "[skip] ${name} seed${seed}"
        printf '%s seed%s: skipped (exists)\n' "${name}" "${seed}" >> "${TIMING}"
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
        t0=$(date +%s)
        python run_experiment.py "${COMMON[@]}" --seed "${seed}" --save_dir "${save_dir}" ${extra} \
            2>&1 | tail -n 200 > "${log_file}"
        printf '%s seed%s: %ds (gpu%s)\n' "${name}" "${seed}" $(( $(date +%s) - t0 )) "${gpu_id}" >> "${TIMING}"
    ) &
    GPU_PIDS[${gpu_id}]=$!
    n_launched=$((n_launched + 1))
done

wait_all
printf 'TOTAL: %ds  (launched=%d, skipped=%d)\n' $(( $(date +%s) - T0 )) "${n_launched}" "${n_skipped}" | tee -a "${TIMING}"
echo "Done. Run: bash analysis_cnn_agg.sh ${IDX} --gpus ${N_GPUS}"
