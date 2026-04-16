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
    fig, ax = plt.subplots(figsize=(12, 10))

    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap='viridis', vmin=0, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(metric_label, rotation=-90, va="bottom")

    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(task_names, fontsize=7)
    ax.set_yticklabels(task_names, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells (only if matrix is not too large)
    if len(task_names) <= 20:
        for i in range(len(task_names)):
            for j in range(len(task_names)):
                ax.text(j, i, f'{matrix_np[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9)

    ax.set_title(f'STPV {metric_label} Matrix\nProbe: {probe_task}')
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
            matrix_path = os.path.join(matrix_dir, f"{metric}_matrix_{probe_task}.png")
            plot_similarity_matrix(sim_matrix, task_names, probe_task, matrix_path,
                                   metric_label=label)
