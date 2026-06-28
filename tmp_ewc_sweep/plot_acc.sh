#!/usr/bin/env bash
# Quick accuracy matrix plot for exp<IDX>_cnn_* directories.
# No GPU needed, just reads performance_history.json.
#
# Usage:
#   bash tmp_ewc_sweep/plot_acc.sh 8
#   bash tmp_ewc_sweep/plot_acc.sh 1 2 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/plot_acc.py" "$@"
