"""Model similarity matrix module for RNN.

Computes pairwise similarity between representations from different checkpoint
models using pre-saved .npz files.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.drift_metrics import (
    compute_pairwise_similarity_matrix,
    compute_pairwise_cka_matrix,
    compute_linear_cka,
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
                        ha="center", va="center", color="black", fontsize=6)

    ax.set_title(f'Representation {metric_label} Matrix\nProbe: {probe_task}')
    ax.set_xlabel('Model after Task')
    ax.set_ylabel('Model after Task')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Matrix saved to {output_path}")
    plt.close()


def compute_similarity_by_gap(
    reps_dict: Dict[int, torch.Tensor],
    task_indices: List[int],
    exclude_first: bool = True,
) -> Tuple[List[int], List[float], List[float]]:
    """Compute average cosine similarity grouped by task gap."""
    start_idx = 1 if exclude_first else 0
    gap_to_sims: Dict[int, List[float]] = {}

    for i in range(start_idx, len(task_indices)):
        for j in range(i + 1, len(task_indices)):
            gap = task_indices[j] - task_indices[i]
            rep_i = reps_dict[task_indices[i]]
            rep_j = reps_dict[task_indices[j]]
            rep_i_norm = F.normalize(rep_i, p=2, dim=1)
            rep_j_norm = F.normalize(rep_j, p=2, dim=1)
            cos_sims = (rep_i_norm * rep_j_norm).sum(dim=1)
            if gap not in gap_to_sims:
                gap_to_sims[gap] = []
            gap_to_sims[gap].extend(cos_sims.tolist())

    gaps = sorted(gap_to_sims.keys())
    means = [np.mean(gap_to_sims[g]) for g in gaps]
    stds = [np.std(gap_to_sims[g]) for g in gaps]
    return gaps, means, stds


def compute_cka_by_gap(
    reps_dict: Dict[int, torch.Tensor],
    task_indices: List[int],
    exclude_first: bool = True,
) -> Tuple[List[int], List[float], List[float]]:
    """Compute average CKA grouped by task gap."""
    start_idx = 1 if exclude_first else 0
    gap_to_cka: Dict[int, List[float]] = {}

    for i in range(start_idx, len(task_indices)):
        for j in range(i + 1, len(task_indices)):
            gap = task_indices[j] - task_indices[i]
            cka_val = compute_linear_cka(reps_dict[task_indices[i]], reps_dict[task_indices[j]])
            if gap not in gap_to_cka:
                gap_to_cka[gap] = []
            gap_to_cka[gap].append(cka_val)

    gaps = sorted(gap_to_cka.keys())
    means = [np.mean(gap_to_cka[g]) for g in gaps]
    stds = [np.std(gap_to_cka[g]) for g in gaps]
    return gaps, means, stds


def _plot_decay_profile(
    reps_dict: Dict[int, torch.Tensor],
    task_indices: List[int],
    probe_task: str,
    output_path: str,
    metric: str = "cosine",
):
    """Plot decay profile (similarity vs task gap)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric == "cosine":
        gaps, means, stds = compute_similarity_by_gap(reps_dict, task_indices)
        ylabel = "Cosine Similarity"
        title = f"Similarity Decay Profile — probe: {probe_task}"
    else:
        gaps, means, stds = compute_cka_by_gap(reps_dict, task_indices)
        ylabel = "CKA"
        title = f"CKA Decay Profile — probe: {probe_task}"

    ax.errorbar(gaps, means, yerr=stds, marker='o', capsize=5)
    ax.set_title(title)
    ax.set_xlabel("Task Gap")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    if gaps:
        ax.set_xticks(gaps)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Decay profile saved to {output_path}")
    plt.close()


def run_model_similarity(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
) -> None:
    """
    Generate model pairwise cosine similarity matrices and decay profiles.

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' representations to analyze.
        task_names: Ordered list of all task names.
        output_dir: Directory to save results.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    matrix_dir = os.path.join(output_dir, "model_similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  Cosine similarity for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())
        reps_dict = {k: torch.from_numpy(v).float() for k, v in raw_reps.items()}
        reps_list = [reps_dict[t] for t in sorted_indices]

        sim_matrix = compute_pairwise_similarity_matrix(reps_list)
        matrix_path = os.path.join(matrix_dir, f"cosine_matrix_{probe_task}.png")
        plot_similarity_matrix(sim_matrix, task_names, probe_task, matrix_path)

        profile_path = os.path.join(output_dir, f"cosine_decay_{probe_task}.png")
        _plot_decay_profile(reps_dict, sorted_indices, probe_task, profile_path, metric="cosine")


def run_model_cka_similarity(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
) -> None:
    """
    Generate model pairwise CKA matrices and decay profiles.

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' representations to analyze.
        task_names: Ordered list of all task names.
        output_dir: Directory to save results.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    matrix_dir = os.path.join(output_dir, "model_cka_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  CKA for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())
        reps_dict = {k: torch.from_numpy(v).float() for k, v in raw_reps.items()}
        reps_list = [reps_dict[t] for t in sorted_indices]

        cka_matrix = compute_pairwise_cka_matrix(reps_list)
        matrix_path = os.path.join(matrix_dir, f"cka_matrix_{probe_task}.png")
        plot_similarity_matrix(cka_matrix, task_names, probe_task, matrix_path, metric_label="CKA")

        profile_path = os.path.join(output_dir, f"cka_decay_{probe_task}.png")
        _plot_decay_profile(reps_dict, sorted_indices, probe_task, profile_path, metric="cka")
