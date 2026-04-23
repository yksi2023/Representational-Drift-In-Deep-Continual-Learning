#!/usr/bin/env bash
# Drift analysis on every rnn/experiments/exp<i>_rnn_*/ directory.
# Usage:  bash analysis_rnn.sh <i>
set -euo pipefail
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash analysis_rnn.sh <i> [<i2> ...]   # analyzes all exp<i>_rnn_*"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/rnn"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Collect all matching directories for every requested index.
# ------------------------------------------------------------------
all_dirs=()
for IDX in "$@"; do
    PREFIX="exp${IDX}_rnn_"
    shopt -s nullglob
    dirs=("${WORK_DIR}/experiments/${PREFIX}"*/)
    shopt -u nullglob
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No directories matched: ${WORK_DIR}/experiments/${PREFIX}*"
    else
        all_dirs+=("${dirs[@]}")
    fi
done

if [ ${#all_dirs[@]} -eq 0 ]; then
    echo "No experiment directories matched any provided indices."
    exit 1
fi

echo "Analyzing ${#all_dirs[@]} experiment(s):"
printf '  - %s\n' "${all_dirs[@]##*/experiments/}"

for d in "${all_dirs[@]}"; do
    d="${d%/}"
    echo ""
    echo "==> Analyzing: $(basename "${d}")"
    python analyze_drift.py --exp_dir "${d}" --skip_sample_sim
done

echo ""
echo "All RNN drift analyses complete."
