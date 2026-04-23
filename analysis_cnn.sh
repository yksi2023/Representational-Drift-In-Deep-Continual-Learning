#!/usr/bin/env bash
# Drift analysis on every cnn/experiments/exp<i>_cnn_*/ directory.
# Auto-detects probe layer names from the first matching config.
# Usage:  bash analysis_cnn.sh <i> [LAYERS]
set -euo pipefail
source activate drift

INDICES=()
LAYERS_OVERRIDE=""
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        INDICES+=("$arg")
    else
        LAYERS_OVERRIDE="$arg"
    fi
done

if [ ${#INDICES[@]} -eq 0 ]; then
    echo "Usage: bash analysis_cnn.sh <i> [<i2> ...] [LAYERS]"
    echo "  <i>      experiment index -> matches exp<i>_cnn_*"
    echo "  LAYERS   optional comma-separated layer names (override auto-default)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"

cd "${WORK_DIR}"

# ------------------------------------------------------------------
# Collect all matching directories for every requested index.
# ------------------------------------------------------------------
all_dirs=()
for IDX in "${INDICES[@]}"; do
    PREFIX="exp${IDX}_cnn_"
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

# Pick default probe layers based on the model recorded in the first matching
# experiment_config.json. Override with the optional 2nd positional arg.
default_layers_for() {
    case "$1" in
        resnet18_pretrained)
            echo "backbone.layer3.0.relu,backbone.layer3.1.relu,backbone.layer4.0.relu,backbone.layer4.1.relu" ;;
        resnet18_tiny)
            echo "conv_layer.4,conv_layer.5,conv_layer.6,conv_layer.7" ;;
        resnet18_cifar_gn)
            echo "layer2.1.relu2,layer3.0.relu2,layer3.1.relu2,layer4.0.relu2,layer4.1.relu2" ;;
        bit_s_r50x1_in1k)
            echo "backbone.stages.0,backbone.stages.1,backbone.stages.2,backbone.stages.3,backbone.norm" ;;
        *)
            echo "" ;;
    esac
}

echo "Analyzing ${#all_dirs[@]} experiment(s):"
printf '  - %s\n' "${all_dirs[@]##*/experiments/}"

for d in "${all_dirs[@]}"; do
    d="${d%/}"
    cfg="${d}/experiment_config.json"
    if [ -n "${LAYERS_OVERRIDE}" ]; then
        layers="${LAYERS_OVERRIDE}"
    elif [ -f "${cfg}" ]; then
        model=$(python -c "import json; print(json.load(open('${cfg}')).get('model', ''))")
        layers=$(default_layers_for "${model}")
        if [ -z "${layers}" ]; then
            echo "!! No default layers for model '${model}' in $(basename ${d}); skipping. Pass LAYERS as 2nd arg to override."
            continue
        fi
    else
        echo "!! Missing ${cfg}; skipping $(basename ${d})."
        continue
    fi

    echo ""
    echo "==> Analyzing: $(basename "${d}")   (layers=${layers})"
    python analyze_drift.py \
        --ckpt_dir "${d}" \
        --layers "${layers}" 
done

echo ""
echo "All CNN drift analyses complete."
