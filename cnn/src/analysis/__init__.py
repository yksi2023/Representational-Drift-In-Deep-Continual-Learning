import matplotlib
import matplotlib.pyplot as plt

# Global plot style configuration — paper-ready
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12
plt.rcParams.update({
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

from .cache import build_reps_cache
from .baseline_drift import run_baseline_drift
from .model_similarity import run_model_similarity
from .sample_similarity import run_sample_similarity
from .subspace_drift import run_subspace_drift
from .gap_drift import run_gap_drift
from .performance import plot_cnn_performance
from .drift_metrics import (
    compute_metrics,
    compute_pairwise_similarity_matrix,
)

__all__ = [
    "build_reps_cache",
    "run_baseline_drift",
    "run_model_similarity",
    "run_sample_similarity",
    "run_subspace_drift",
    "run_gap_drift",
    "plot_cnn_performance",
    "compute_metrics",
    "compute_pairwise_similarity_matrix",
]
