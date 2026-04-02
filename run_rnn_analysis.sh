#!/usr/bin/env bash
# Analyze all exp1_* experiment directories under rnn/experiments/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RNN_DIR="${SCRIPT_DIR}/rnn"
EXP_ROOT="${RNN_DIR}/experiments"

cd "${RNN_DIR}"

for exp_dir in "${EXP_ROOT}"/exp1_*; do
    [ -d "${exp_dir}" ] || continue
    name="$(basename "${exp_dir}")"
    echo ""
    echo "======== Analyzing: ${name} ========"
    python analyze_drift.py --exp_dir "${exp_dir}" --skip_sample_sim
done

echo ""
echo "All analyses complete."
