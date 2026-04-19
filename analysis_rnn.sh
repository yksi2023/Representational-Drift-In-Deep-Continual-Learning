#!/usr/bin/env bash
# Drift analysis on every rnn/experiments/<prefix>* directory.
# Usage:  bash analysis_rnn.sh [PREFIX]     # default prefix: exp2_rnn_
set -euo pipefail
source activate drift

PREFIX="${1:-exp2_rnn_}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/rnn"

cd "${WORK_DIR}"

shopt -s nullglob
dirs=("${WORK_DIR}/experiments/${PREFIX}"*/)
shopt -u nullglob

if [ ${#dirs[@]} -eq 0 ]; then
    echo "No directories matched: ${WORK_DIR}/experiments/${PREFIX}*"
    exit 1
fi

echo "Analyzing ${#dirs[@]} experiment(s) with prefix '${PREFIX}':"
printf '  - %s\n' "${dirs[@]##*/experiments/}"

for d in "${dirs[@]}"; do
    d="${d%/}"
    echo ""
    echo "==> Analyzing: $(basename "${d}")"
    python analyze_drift.py --exp_dir "${d}" --skip_sample_sim
done

echo ""
echo "All RNN drift analyses complete."
