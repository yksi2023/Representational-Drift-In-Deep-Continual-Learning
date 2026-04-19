"""Model pairwise cosine similarity matrices + similarity decay vs task gap."""
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.drift_metrics import compute_pairwise_similarity_matrix


def plot_similarity_matrix(
    sim_matrix: torch.Tensor,
    task_indices: List[int],
    layer_name: str,
    output_path: str,
    metric_label: str = "Cosine Similarity",
):
    """Plot similarity matrix as a heatmap with colormap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap="viridis", vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(metric_label, rotation=-90, va="bottom")

    ax.set_xticks(range(len(task_indices)))
    ax.set_yticks(range(len(task_indices)))
    ax.set_xticklabels([f"T{t}" for t in task_indices])
    ax.set_yticklabels([f"T{t}" for t in task_indices])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(task_indices)):
        for j in range(len(task_indices)):
            ax.text(j, i, f"{matrix_np[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f"Representation {metric_label} Matrix\nLayer: {layer_name}")
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Model after Task")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Similarity matrix saved to {output_path}")
    plt.close()


def compute_similarity_by_gap(
    reps_dict: Dict[int, torch.Tensor],
    task_indices: List[int],
    exclude_first: bool = True,
) -> Tuple[List[int], List[float], List[float]]:
    """Mean per-sample cosine similarity grouped by task gap."""
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
            gap_to_sims.setdefault(gap, []).extend(cos_sims.tolist())

    gaps = sorted(gap_to_sims.keys())
    means = [float(np.mean(gap_to_sims[g])) for g in gaps]
    stds = [float(np.std(gap_to_sims[g])) for g in gaps]
    return gaps, means, stds


def plot_similarity_decay_profile(
    all_reps: Dict[str, Dict[int, torch.Tensor]],
    task_indices: List[int],
    layer_names: List[str],
    output_path: str,
    exclude_first: bool = True,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_gaps: List[int] = []
    for layer_name in layer_names:
        gaps, means, stds = compute_similarity_by_gap(all_reps[layer_name], task_indices, exclude_first)
        all_gaps.extend(gaps)
        ax.errorbar(gaps, means, yerr=stds, marker="o", capsize=5, label=layer_name)

    ax.set_title("Similarity Decay Profile")
    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1)
    if all_gaps:
        ax.set_xticks(sorted(set(all_gaps)))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Similarity decay profile saved to {output_path}")
    plt.close()


def _reshape_by_layer(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Convert {task: {layer: T}} -> {layer: {task: T}}."""
    sorted_tasks = sorted(reps_cache.keys())
    return {ln: {t: reps_cache[t][ln] for t in sorted_tasks} for ln in layer_names}


def run_model_similarity(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
):
    """Cosine pairwise similarity matrix per layer + decay profile across layers."""
    print("\nGenerating model similarity matrix heatmaps...")
    matrix_dir = os.path.join(output_dir, "model_similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())
    all_reps = _reshape_by_layer(reps_cache, layer_names)

    for layer in layer_names:
        reps_list = [all_reps[layer][t] for t in sorted_task_indices]
        sim_matrix = compute_pairwise_similarity_matrix(reps_list)
        safe_layer_name = layer.replace(".", "_").replace("/", "_")
        matrix_path = os.path.join(matrix_dir, f"similarity_matrix_{safe_layer_name}.png")
        plot_similarity_matrix(sim_matrix, sorted_task_indices, layer, matrix_path)

    profile_path = os.path.join(output_dir, "similarity_decay_profile.png")
    plot_similarity_decay_profile(all_reps, sorted_task_indices, layer_names, profile_path)
