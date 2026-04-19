#!/usr/bin/env bash
# Drift analysis on every cnn/experiments/<prefix>* directory.
# Usage:  bash analysis_cnn.sh [PREFIX]     # default prefix: exp_cnn_
set -euo pipefail
source activate drift

PREFIX="${1:-exp_cnn_}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"
LAYERS="backbone.layer3.0.relu,backbone.layer3.1.relu,backbone.layer4.0.relu,backbone.layer4.1.relu"

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
    python analyze_drift.py \
        --ckpt_dir "${d}" \
        --layers "${LAYERS}" \
        --dataset tiny_imagenet \
        --amp
done

echo ""
echo "All CNN drift analyses complete."
