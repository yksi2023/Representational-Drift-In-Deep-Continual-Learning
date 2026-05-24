#!/usr/bin/env bash
# Drift analysis on every rnn/experiments/exp<i>_rnn_*/ directory.
# Usage:  bash analysis_rnn.sh <i> [options]
# Options:
#   --skip_umap            Skip direction UMAP analysis
#   --n_directions N       Number of stimulus directions for UMAP (default: 8)
#   --n_trials_per_dir N   Repetitions per direction for UMAP (default: 40)
set -euo pipefail
source activate drift

# Parse flags from arguments
EXTRA_ARGS=()
INDICES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip_umap)
            EXTRA_ARGS+=("$1"); shift
            ;;
        --n_directions)
            EXTRA_ARGS+=("$1" "$2"); shift 2
            ;;
        --n_directions=*)
            EXTRA_ARGS+=("$1"); shift
            ;;
        --n_trials_per_dir)
            EXTRA_ARGS+=("$1" "$2"); shift 2
            ;;
        --n_trials_per_dir=*)
            EXTRA_ARGS+=("$1"); shift
            ;;
        *)
            INDICES+=("$1"); shift
            ;;
    esac
done

if [ ${#INDICES[@]} -lt 1 ]; then
    echo "Usage: bash analysis_rnn.sh <i> [<i2> ...] [options]"
    echo "  --skip_umap            Skip direction UMAP analysis"
    echo "  --n_directions N       Number of stimulus directions for UMAP (default: 8)"
    echo "  --n_trials_per_dir N   Repetitions per direction for UMAP (default: 40)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/rnn"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Collect all matching directories for every requested index.
# ------------------------------------------------------------------
all_dirs=()
for IDX in "${INDICES[@]}"; do
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
    python analyze_drift.py --exp_dir "${d}" --skip_sample_sim "${EXTRA_ARGS[@]}"
done

echo ""
echo "All RNN drift analyses complete."
