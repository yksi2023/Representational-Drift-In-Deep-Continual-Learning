#!/usr/bin/env bash
#SBATCH --job-name=expB_train
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Experiment B (CNN) -- multi-GPU parallel training.
#
# Trains both arms over a lambda grid x seeds:
#   * replay,          anchor_lambda = 0   (weights free, code drifts)
#   * anchored_replay, anchor_lambda > 0   (weights free, code pinned)
#
# Results: cnn/experiments/exp<i>b_cnn_<arm>_seed<s>/
#
# Submit:
#   sbatch b_train.sh 1
#   sbatch --partition gpu_5090 --gpus 8 b_train.sh 1 --seeds 0,1,2 --lambdas 0.01,0.1,1
#
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

# --------------- argument parsing ---------------
if [ $# -lt 1 ]; then
    echo "Usage: sbatch [slurm opts] b_train.sh <i> [--seeds s1,s2,...] [--lambdas l1,l2,...]"
    exit 1
fi
IDX="$1"; shift

SEED_STR="0,1,2,3,4"
LAMBDA_STR="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30"

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)  SEED_STR="$2";  shift 2 ;;
        --lambdas) LAMBDA_STR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect GPUs from SLURM allocation (fallback: nvidia-smi)
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))

IFS=',' read -ra SEEDS <<< "${SEED_STR}"
IFS=',' read -ra LAMBDAS <<< "${LAMBDA_STR}"

ANCHOR_LOSS="mse"
ANCHOR_LAYERS="layer3,layer4"
MEMORY_PER_CLASS=300

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/cnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}b_cnn_"
TIMING="${EXP_ROOT}/exp${IDX}b_cnn_timing.txt"
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
# Build list of all (name, seed, extra_args...) tuples
declare -a JOBS=()

for s in "${SEEDS[@]}"; do
    JOBS+=("replay_l0|${s}|--method replay --memory_per_class ${MEMORY_PER_CLASS}")
    for L in "${LAMBDAS[@]}"; do
        tag="anchored_${ANCHOR_LOSS}_l${L//./p}"
        JOBS+=("${tag}|${s}|--method anchored_replay --memory_per_class ${MEMORY_PER_CLASS} --anchor_lambda ${L} --anchor_loss ${ANCHOR_LOSS} --anchor_layers ${ANCHOR_LAYERS}")
    done
done

# Actual concurrency: never more parallel jobs than there are jobs.
N_PARALLEL=$(( ${#JOBS[@]} < N_GPUS ? ${#JOBS[@]} : N_GPUS ))
N_PARALLEL=$(( N_PARALLEL > 0 ? N_PARALLEL : 1 ))

echo "=== Experiment B Training ==="
echo "  IDX=${IDX}, GPUs=${N_GPUS}, Seeds=(${SEED_STR}), Lambdas=(${LAMBDA_STR})"
echo "  Total jobs: ${#JOBS[@]}, max parallel: ${N_PARALLEL}"
echo ""

# --------------- parallel dispatcher ---------------
# Maintains up to N_GPUS concurrent processes, pinning each to a GPU.
declare -a GPU_PIDS=()       # PID running on each GPU slot (0 = free)
for ((g=0; g<N_GPUS; g++)); do GPU_PIDS+=(0); done

wait_for_slot() {
    # Block until at least one GPU slot is free.
    while true; do
        for ((g=0; g<N_GPUS; g++)); do
            if [ "${GPU_PIDS[$g]}" -eq 0 ]; then
                AVAIL_GPU=$g; return
            fi
            # Check if the process finished
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
        # Limit CPU threads per job to avoid oversubscription across concurrent processes.
        n_cores=$(nproc)
        per_job=$(( n_cores / N_PARALLEL ))
        export OMP_NUM_THREADS=$(( per_job > 1 ? per_job : 1 ))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        # DataLoader workers: code uses min(8, cpu_count//2). Override cpu_count visibility.
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
echo "Done. Run: bash b_analysis.sh ${IDX} --gpus ${N_GPUS}"
