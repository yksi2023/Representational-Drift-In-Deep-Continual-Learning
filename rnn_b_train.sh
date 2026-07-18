#!/usr/bin/env bash
#SBATCH --job-name=expB_rnn_train
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Experiment B (RNN): replay baseline plus state-trajectory anchoring.
# Results: rnn/experiments/exp<i>b_rnn_<arm>_seed<s>/
#
# Submit:
#   sbatch rnn_b_train.sh 1
#   sbatch rnn_b_train.sh 1 --seeds 0,1,2 --lambdas 0.01,0.1,1
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: sbatch [slurm opts] rnn_b_train.sh <i> [--seeds s1,s2,...] [--lambdas l1,l2,...]"
    exit 1
fi
IDX="$1"; shift
SEED_STR="0,1,2,3,4"
LAMBDA_STR="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30"
while [ $# -gt 0 ]; do
    case "$1" in
        --seeds) SEED_STR="$2"; shift 2 ;;
        --lambdas) LAMBDA_STR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))
IFS=',' read -ra SEEDS <<< "${SEED_STR}"
IFS=',' read -ra LAMBDAS <<< "${LAMBDA_STR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}b_rnn_"
LOG_DIR="${EXP_ROOT}/logs"
TIMING="${EXP_ROOT}/${PREFIX}timing.txt"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}"
: > "${TIMING}"

COMMON=(
    --hidden_size 256
    --sigma_rec 0.05
    --activation softplus
    --w_rec_init diag
    --num_iterations 5000
    --batch_size 1024
    --lr 0.001
    --train_pool_size 30
    --memory_per_task 300
    --replay_num_tasks 5
    --anchor_loss mse
    --anchor_probe_size 200
)

declare -a JOBS=()
for seed in "${SEEDS[@]}"; do
    JOBS+=("replay_l0|${seed}|--method replay")
    for lambda in "${LAMBDAS[@]}"; do
        tag="anchored_replay_mse_l${lambda//./p}"
        JOBS+=("${tag}|${seed}|--method anchored_replay --anchor_lambda ${lambda}")
    done
done

N_PARALLEL=$(( ${#JOBS[@]} < N_GPUS ? ${#JOBS[@]} : N_GPUS ))
N_PARALLEL=$(( N_PARALLEL > 0 ? N_PARALLEL : 1 ))
cd "${WORK_DIR}"
declare -a GPU_PIDS=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do GPU_PIDS+=(0); done

wait_for_slot() {
    while true; do
        for ((gpu=0; gpu<N_GPUS; gpu++)); do
            if [ "${GPU_PIDS[$gpu]}" -eq 0 ]; then AVAIL_GPU=$gpu; return; fi
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                wait "${GPU_PIDS[$gpu]}" 2>/dev/null || true
                GPU_PIDS[$gpu]=0; AVAIL_GPU=$gpu; return
            fi
        done
        sleep 2
    done
}

wait_all() {
    for ((gpu=0; gpu<N_GPUS; gpu++)); do
        if [ "${GPU_PIDS[$gpu]}" -ne 0 ]; then
            wait "${GPU_PIDS[$gpu]}" 2>/dev/null || true
            GPU_PIDS[$gpu]=0
        fi
    done
}

echo "=== Experiment B RNN Training ==="
echo "  IDX=${IDX}, GPUs=${N_GPUS}, Seeds=(${SEED_STR}), Lambdas=(${LAMBDA_STR})"
echo "  Total jobs: ${#JOBS[@]}, max parallel: ${N_PARALLEL}"

T0=$(date +%s)
n_launched=0
n_skipped=0
for job_spec in "${JOBS[@]}"; do
    IFS='|' read -r name seed extra <<< "${job_spec}"
    save_dir="${EXP_ROOT}/${PREFIX}${name}_seed${seed}"
    if [ -f "${save_dir}/performance_history.json" ]; then
        echo "[skip] ${name} seed${seed}"
        printf '%s seed%s: skipped (exists)\n' "${name}" "${seed}" >> "${TIMING}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    wait_for_slot
    gpu_id=${AVAIL_GPU}
    log_file="${LOG_DIR}/${PREFIX}${name}_seed${seed}.log"
    echo "[gpu${gpu_id}] ${name} seed${seed} -> ${log_file}"
    (
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        n_cores=$(nproc)
        per_job=$((n_cores / N_PARALLEL))
        export OMP_NUM_THREADS=$((per_job > 1 ? per_job : 1))
        export MKL_NUM_THREADS=${OMP_NUM_THREADS}
        export TORCH_NUM_THREADS=${OMP_NUM_THREADS}
        export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
        t0=$(date +%s)
        python run_experiment.py "${COMMON[@]}" --seed "${seed}" --save_dir "${save_dir}" ${extra} \
            2>&1 | tail -n 200 > "${log_file}"
        printf '%s seed%s: %ds (gpu%s)\n' "${name}" "${seed}" "$(( $(date +%s) - t0 ))" "${gpu_id}" >> "${TIMING}"
    ) &
    GPU_PIDS[${gpu_id}]=$!
    n_launched=$((n_launched + 1))
done

wait_all
printf 'TOTAL: %ds (launched=%d, skipped=%d)\n' "$(( $(date +%s) - T0 ))" "${n_launched}" "${n_skipped}" | tee -a "${TIMING}"
echo "Done. Analyze with: sbatch rnn_b_analysis.sh ${IDX}"
