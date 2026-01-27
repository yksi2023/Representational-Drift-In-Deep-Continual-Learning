import matplotlib.pyplot as plt

# Global plot style configuration
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

from .baseline_drift import run_baseline_drift
from .model_similarity import run_model_similarity
from .sample_similarity import run_sample_similarity
from .drift_metrics import compute_metrics, compute_pairwise_similarity_matrix

__all__ = [
    "run_baseline_drift",
    "run_model_similarity", 
    "run_sample_similarity",
    "compute_metrics",
    "compute_pairwise_similarity_matrix",
]
