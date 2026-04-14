"""Sample-wise similarity matrix module for RNN.

Computes cosine similarity between all sample pairs within each checkpoint model,
using pre-saved .npz representations.
"""
import os
from typing import List

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.analysis.baseline_drift import _load_reps_from_npz


def compute_sample_similarity_matrix(reps: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """
    Compute pairwise similarity matrix between all samples.

    Args:
        reps: Tensor of shape [N, D]
        metric: "cosine" or "pearson"

    Returns:
        Similarity matrix of shape [N, N].
    """
    r = reps
    if metric == "pearson":
        r = r - r.mean(dim=1, keepdim=True)
    r_norm = F.normalize(r, p=2, dim=1)
    return torch.mm(r_norm, r_norm.t())


def plot_sample_similarity_matrix(
    sim_matrix: torch.Tensor,
    task_idx: int,
    probe_task: str,
    output_path: str,
    metric_label: str = "Cosine Similarity",
):
    """Plot sample-wise similarity matrix as a colormap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap='viridis', vmin=-1, vmax=1, aspect='auto')

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(metric_label, rotation=-90, va="bottom")

    ax.set_title(f'Sample {metric_label} — Model after Task {task_idx}\nProbe: {probe_task}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_sample_similarity(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
) -> None:
    """
    Analyze sample-wise similarity matrices (cosine + Pearson) for each checkpoint.

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' representations to analyze.
        task_names: Ordered list of all task names.
        output_dir: Directory to save results.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    sample_sim_dir = os.path.join(output_dir, "sample_similarity_matrices")
    os.makedirs(sample_sim_dir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  Sample similarity for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())

        probe_dir = os.path.join(sample_sim_dir, probe_task)
        os.makedirs(probe_dir, exist_ok=True)

        for task_idx in sorted_indices:
            reps = torch.from_numpy(raw_reps[task_idx]).float()
            task_label = task_names[task_idx] if task_idx < len(task_names) else f"task_{task_idx}"

            for metric, label in [("cosine", "Cosine Similarity"), ("pearson", "Pearson Correlation")]:
                sim_matrix = compute_sample_similarity_matrix(reps, metric=metric)
                output_path = os.path.join(probe_dir, f"sample_{metric}_after_{task_label}.png")
                plot_sample_similarity_matrix(sim_matrix, task_idx, probe_task, output_path,
                                              metric_label=label)

        print(f"    Saved matrices to {probe_dir}")
