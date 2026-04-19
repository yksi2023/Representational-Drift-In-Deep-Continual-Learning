"""Sample-wise cosine similarity matrices, sorted by class label."""
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def compute_sample_similarity_matrix(reps: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity matrix between samples, shape [N, N]."""
    reps_norm = F.normalize(reps, p=2, dim=1)
    return torch.mm(reps_norm, reps_norm.t())


def plot_sample_similarity_matrix(
    sim_matrix: torch.Tensor,
    task_idx: int,
    layer_name: str,
    output_path: str,
    class_boundaries: List[int] = None,
):
    """Plot sample-wise similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")

    if class_boundaries is not None:
        for boundary in class_boundaries:
            ax.axhline(y=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(x=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)

    ax.set_title(f"Sample Similarity Matrix - Model after Task {task_idx}\nLayer: {layer_name}")
    ax.set_xlabel("Sample Index (sorted by class)")
    ax.set_ylabel("Sample Index (sorted by class)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_sample_similarity(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
):
    """Plot sample-wise similarity matrix for every (checkpoint, layer)."""
    print("\n" + "=" * 60)
    print("GENERATING SAMPLE-WISE SIMILARITY MATRICES")
    print("=" * 60)

    sample_sim_dir = os.path.join(output_dir, "sample_similarity_matrices")
    os.makedirs(sample_sim_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())

    sort_indices = torch.argsort(labels)
    sorted_labels = labels[sort_indices]
    unique_labels = sorted_labels.unique()
    class_boundaries = []
    for lbl in unique_labels[1:]:
        boundary_idx = (sorted_labels == lbl).nonzero(as_tuple=True)[0][0].item()
        class_boundaries.append(boundary_idx)

    print(f"Total samples: {len(labels)}, Classes: {len(unique_labels)}")
    print(f"Class boundaries at indices: {class_boundaries}")

    for task_idx in sorted_task_indices:
        print(f"  Processing model after Task {task_idx}...")
        for layer in layer_names:
            layer_reps = reps_cache[task_idx][layer][sort_indices]
            sim_matrix = compute_sample_similarity_matrix(layer_reps)
            safe_layer_name = layer.replace(".", "_").replace("/", "_")
            layer_dir = os.path.join(sample_sim_dir, safe_layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            output_path = os.path.join(
                layer_dir, f"sample_sim_task{task_idx}_{safe_layer_name}.png"
            )
            plot_sample_similarity_matrix(
                sim_matrix, task_idx, layer, output_path, class_boundaries
            )

    print(f"\nAll sample similarity matrices saved to {sample_sim_dir}")
