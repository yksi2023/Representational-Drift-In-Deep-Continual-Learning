#!/usr/bin/env bash
#SBATCH --job-name=ewc_5e6_s89
#SBATCH --partition=gpu_4090
#SBATCH --gpus=2
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
ulimit -n 65536
module load miniforge3/26.1
source activate drift

IDX="${1:-9}"
SEEDS=(8 9)
LAMBDA=5000000

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
    --ewc_lambda ${LAMBDA}
)

cd "${WORK_DIR}"

T0=$(date +%s)

for i in 0 1; do
    s=${SEEDS[$i]}
    save_dir="${EXP_ROOT}/${PREFIX}ewc_l${LAMBDA}_seed${s}"
    log_file="${LOG_DIR}/${PREFIX}ewc_l${LAMBDA}_seed${s}_train.log"

    if [ -f "${save_dir}/comprehensive_evaluation.json" ]; then
        echo "[skip] seed${s}"
        continue
    fi

    echo "[gpu${i}] seed${s}"
    (
        export CUDA_VISIBLE_DEVICES=${i}
        python run_experiment.py "${COMMON[@]}" --seed "${s}" --save_dir "${save_dir}" \
            2>&1 | tail -n 200 > "${log_file}"
    ) &
done

wait
echo "Done. ${#SEEDS[@]} seeds, $(($(date +%s) - T0))s"
