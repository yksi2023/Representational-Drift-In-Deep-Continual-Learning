#!/usr/bin/env bash
# Drift analysis on every cnn/experiments/exp<i>_cnn_*/ directory.
# Auto-detects probe layer names from the first matching config.
# Usage:  bash analysis_cnn.sh <i> [LAYERS]
set -euo pipefail
source activate drift

if [ $# -lt 1 ]; then
    echo "Usage: bash analysis_cnn.sh <i> [LAYERS]"
    echo "  <i>      experiment index -> matches exp<i>_cnn_*"
    echo "  LAYERS   optional comma-separated layer names (override auto-default)"
    exit 1
fi
IDX="$1"
LAYERS_OVERRIDE="${2:-}"
PREFIX="exp${IDX}_cnn_"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/cnn"

cd "${WORK_DIR}"

shopt -s nullglob
dirs=("${WORK_DIR}/experiments/${PREFIX}"*/)
shopt -u nullglob

if [ ${#dirs[@]} -eq 0 ]; then
    echo "No directories matched: ${WORK_DIR}/experiments/${PREFIX}*"
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
            echo "backbone.stages.2.blocks.5.norm3,backbone.stages.3.blocks.2.norm3" ;;
        *)
            echo "" ;;
    esac
}

echo "Analyzing ${#dirs[@]} experiment(s) with prefix '${PREFIX}':"
printf '  - %s\n' "${dirs[@]##*/experiments/}"

for d in "${dirs[@]}"; do
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
        --layers "${layers}" \
        --amp
done

echo ""
echo "All CNN drift analyses complete."
