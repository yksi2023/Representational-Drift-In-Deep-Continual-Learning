#!/usr/bin/env bash
#SBATCH --job-name=rnn_ewc_sweep
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --array=0-5%4

set -euo pipefail
ulimit -n 65536
source /data/apps/miniforge3/26.1/etc/profile.d/conda.sh
conda activate drift

if [ $# -lt 1 ]; then
    echo "Usage: sbatch tmp_ewc_sweep/rnn_ewc_sweep.sh <idx> [seed]"
    exit 1
fi

IDX="$1"
SEED="${2:-0}"
LAMBDAS=(0.01 0.1 1 10 100 1000)
LAMBDA="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

WORK_DIR="${SLURM_SUBMIT_DIR}/rnn"
EXP_ROOT="${WORK_DIR}/experiments"
NAME="ewc_l${LAMBDA}"
SAVE_DIR="${EXP_ROOT}/exp${IDX}_rnn_${NAME}_seed${SEED}"

cd "${WORK_DIR}"

if [ -f "${SAVE_DIR}/performance_history.json" ]; then
    echo "[skip] ${NAME} seed${SEED} already complete"
    exit 0
fi

echo "Training ${NAME}, seed=${SEED}, output=${SAVE_DIR}"

python -u run_experiment.py \
    --hidden_size 256 \
    --sigma_rec 0.05 \
    --activation softplus \
    --w_rec_init diag \
    --num_iterations 5000 \
    --batch_size 1024 \
    --lr 0.001 \
    --train_pool_size 30 \
    --method ewc \
    --ewc_lambda "${LAMBDA}" \
    --fisher_samples 200 \
    --seed "${SEED}" \
    --save_dir "${SAVE_DIR}"
