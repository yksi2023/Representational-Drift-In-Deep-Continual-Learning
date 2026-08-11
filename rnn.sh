#!/usr/bin/env bash
#SBATCH --job-name=rnn_train
#SBATCH --partition=gpu_4090
#SBATCH --gpus=4
#SBATCH --output=logs/%x_%j.out
#
# Train RNN CL methods -- multi-GPU parallel, multi-seed.
# Default methods: normal, replay.
# Results: rnn/experiments/exp<i>_rnn_<method>_seed<s>/
#
# Submit:
#   sbatch rnn.sh 1
#   sbatch --partition gpu_5090 --gpus 8 rnn.sh 1 --seeds 0,1,2
#   sbatch rnn.sh 1 --methods normal,replay,duncker
#
# -----------------------------------------------------------------------------
set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

if [ $# -lt 1 ]; then
    echo "Usage: sbatch [slurm opts] rnn.sh <i> [--seeds s1,s2,...] [--methods m1,m2,...]"
    exit 1
fi
IDX="$1"; shift

SEED_STR="0,1,2,3,4"
METHOD_STR="normal,replay"

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)   SEED_STR="$2";   shift 2 ;;
        --methods) METHOD_STR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_GPUS=$((N_GPUS > 0 ? N_GPUS : 1))

IFS=',' read -ra SEEDS <<< "${SEED_STR}"
IFS=',' read -ra METHODS <<< "${METHOD_STR}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORK_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
PREFIX="exp${IDX}_rnn_"
TIMING="${EXP_ROOT}/${PREFIX}timing.txt"
LOG_DIR="${EXP_ROOT}/logs"
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
)

cd "${WORK_DIR}"

# --------------- build job list ---------------
# Map method name -> extra CLI args
method_args() {
    case "$1" in
        normal)   echo "--method normal" ;;
        replay)   echo "--method replay --memory_per_task 300" ;;
        ewc)      echo "--method ewc --ewc_lambda 100.0 --fisher_samples 200" ;;
        lwf)      echo "--method lwf --lwf_lambda 1.0 --lwf_temperature 2.0" ;;
        duncker)  echo "--method duncker --duncker_alpha 0.01 --duncker_samples 512" ;;
        hypernet) echo "--method hypernet --hnet_beta 0.5 --hnet_chunks 10 --hnet_hidden 128" ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

declare -a JOBS=()
for s in "${SEEDS[@]}"; do
    for m in "${METHODS[@]}"; do
        extra=$(method_args "${m}") || exit 1
        JOBS+=("${m}|${s}|${extra}")
    done
done

N_PARALLEL=$(( ${#JOBS[@]} < N_GPUS ? ${#JOBS[@]} : N_GPUS ))
N_PARALLEL=$(( N_PARALLEL > 0 ? N_PARALLEL : 1 ))

echo "=== RNN Training ==="
echo "  IDX=${IDX}, GPUs=${N_GPUS}, Seeds=(${SEED_STR}), Methods=(${METHOD_STR})"
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

    # Skip completed runs
    if [ -f "${save_dir}/performance_history.json" ]; then
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
echo "Done. Run: sbatch analysis_rnn.sh ${IDX}"
