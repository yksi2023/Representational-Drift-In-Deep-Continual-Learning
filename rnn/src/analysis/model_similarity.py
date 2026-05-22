"""Model STPV similarity matrix module for RNN.

Computes pairwise similarity between Spatiotemporal Population Vectors (STPVs)
from different checkpoint models using pre-saved .npz files.
STPV = concatenation of Population Vectors across all time steps.
"""
import os
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.analysis._plot_utils import t_labels, sparse_ticks

from src.drift_metrics import (
    compute_pairwise_similarity_matrix,
    compute_pairwise_pearson_matrix,
)
from src.analysis.baseline_drift import _load_reps_from_npz


def plot_similarity_matrix(
    sim_matrix: torch.Tensor,
    task_names: List[str],
    probe_task: str,
    output_path: str,
    metric_label: str = "Cosine Similarity",
):
    """Plot similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap='viridis', vmin=0, vmax=1)

    sp, sl = sparse_ticks(len(task_names))
    ax.set_xticks(sp)
    ax.set_yticks(sp)
    ax.set_xticklabels(sl)
    ax.set_yticklabels(sl)
    plt.setp(ax.get_xticklabels(), ha="center")

    ax.set_xlabel('Model after Task')
    ax.set_ylabel('Model after Task')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Matrix saved to {output_path}")
    plt.close()


def run_model_similarity(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
) -> None:
    """
    Generate model pairwise STPV similarity matrices (cosine + Pearson).

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' STPVs to analyze.
        task_names: Ordered list of all task names.
        output_dir: Directory to save results.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    matrix_dir = os.path.join(output_dir, "model_similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    for probe_task in probe_tasks:
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())
        reps_dict = {k: torch.from_numpy(v).float() for k, v in raw_reps.items()}
        reps_list = [reps_dict[t] for t in sorted_indices]

        for metric, label, matrix_fn in [
            ("cosine", "Cosine Similarity", compute_pairwise_similarity_matrix),
            ("pearson", "Pearson Correlation", compute_pairwise_pearson_matrix),
        ]:
            print(f"  {label} for probe: {probe_task}")
            sim_matrix = matrix_fn(reps_list)
            matrix_path = os.path.join(matrix_dir, f"{metric}_matrix_{probe_task}.pdf")
            plot_similarity_matrix(sim_matrix, task_names, probe_task, matrix_path,
                                   metric_label=label)
