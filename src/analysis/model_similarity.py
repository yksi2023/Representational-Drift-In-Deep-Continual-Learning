"""Model similarity matrix module.

Computes pairwise similarity between representations from different checkpoint models.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.checkpoints import list_checkpoints, load_model
from src.representations import extract_representations
from src.analysis.drift_metrics import compute_pairwise_similarity_matrix


def plot_similarity_matrix(
    sim_matrix: torch.Tensor, 
    task_indices: List[int], 
    layer_name: str, 
    output_path: str
):
    """Plot similarity matrix as a heatmap with colormap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(range(len(task_indices)))
    ax.set_yticks(range(len(task_indices)))
    ax.set_xticklabels([f'T{t}' for t in task_indices])
    ax.set_yticklabels([f'T{t}' for t in task_indices])
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(task_indices)):
        for j in range(len(task_indices)):
            text = ax.text(j, i, f'{matrix_np[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Representation Similarity Matrix\nLayer: {layer_name}')
    ax.set_xlabel('Model after Task')
    ax.set_ylabel('Model after Task')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Similarity matrix saved to {output_path}")
    plt.close()


def compute_similarity_by_gap(
    reps_dict: Dict[int, torch.Tensor],
    task_indices: List[int],
    exclude_first: bool = True
) -> Tuple[List[int], List[float], List[float]]:
    """
    Compute average similarity grouped by task gap using raw representations.
    
    For each pair of models with gap g, compute per-sample cosine similarity,
    then aggregate all sample similarities for that gap to compute mean and std.
    
    Args:
        reps_dict: Dict mapping task_idx -> [N, D] representation tensor
        task_indices: Sorted list of task indices
        exclude_first: If True, exclude the first task model from analysis
    
    Returns:
        gaps: List of task gap values
        means: List of mean similarities for each gap
        stds: List of std for each gap
    """
    import torch.nn.functional as F
    
    start_idx = 1 if exclude_first else 0  # Skip first task model if needed
    
    # Group all sample-wise similarities by gap
    gap_to_sims: Dict[int, List[float]] = {}
    
    for i in range(start_idx, len(task_indices)):
        for j in range(i + 1, len(task_indices)):
            task_i = task_indices[i]
            task_j = task_indices[j]
            gap = task_j - task_i
            
            # Get representations for both models
            rep_i = reps_dict[task_i]  # [N, D]
            rep_j = reps_dict[task_j]  # [N, D]
            
            # Compute per-sample cosine similarity
            rep_i_norm = F.normalize(rep_i, p=2, dim=1)
            rep_j_norm = F.normalize(rep_j, p=2, dim=1)
            cos_sims = (rep_i_norm * rep_j_norm).sum(dim=1)  # [N]
            
            if gap not in gap_to_sims:
                gap_to_sims[gap] = []
            gap_to_sims[gap].extend(cos_sims.tolist())
    
    # Compute mean and std for each gap
    gaps = sorted(gap_to_sims.keys())
    means = [np.mean(gap_to_sims[g]) for g in gaps]
    stds = [np.std(gap_to_sims[g]) for g in gaps]
    
    return gaps, means, stds


def plot_similarity_decay_profile(
    all_reps: Dict[str, Dict[int, torch.Tensor]],
    task_indices: List[int],
    layer_names: List[str],
    output_path: str,
    exclude_first: bool = True
):
    """
    Plot similarity decay profile for all layers on one figure.
    
    X-axis: Number of tasks between two models (gap)
    Y-axis: Average cosine similarity for model pairs with that gap (computed from raw representations)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for layer_name in layer_names:
        reps_dict = all_reps[layer_name]
        gaps, means, stds = compute_similarity_by_gap(reps_dict, task_indices, exclude_first)
        ax.errorbar(gaps, means, yerr=stds, marker='o', capsize=5, label=layer_name)
    
    ax.set_title('Similarity Decay Profile')
    ax.set_xlabel('Task Gap')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0, 1)
    
    # X-axis ticks: integers only (use gaps from last layer)
    ax.set_xticks(gaps)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Similarity decay profile saved to {output_path}")
    plt.close()


def run_model_similarity(
    model: torch.nn.Module,
    probe_loader: torch.utils.data.DataLoader,
    ckpt_dir: str,
    layer_names: List[str],
    output_dir: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    neuron_indices: Optional[Dict[str, Optional[torch.Tensor]]] = None,
):
    """
    Generate model pairwise similarity matrices.
    
    For each layer, computes cosine similarity between mean representations
    from different checkpoint models.
    
    Args:
        model: Model to analyze
        probe_loader: DataLoader for probe data
        ckpt_dir: Directory containing checkpoints
        layer_names: List of layer names to analyze
        output_dir: Directory to save results
        device: Torch device
        max_batches: Maximum batches to process
        use_amp: Use automatic mixed precision
        neuron_indices: Dict mapping layer name to sampled neuron indices
    """
    print("\nGenerating model similarity matrix heatmaps...")
    
    matrix_dir = os.path.join(output_dir, "model_similarity_matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    
    ckpts = list_checkpoints(ckpt_dir)
    sorted_task_indices = sorted(ckpts.keys())
    
    # Collect representations from all checkpoints
    all_reps: Dict[str, Dict[int, torch.Tensor]] = {ln: {} for ln in layer_names}
    
    for task_idx in sorted_task_indices:
        print(f"Extracting representations from Task {task_idx} model...")
        load_model(model, ckpt_dir, task_idx, map_location=device)
        reps = extract_representations(
            model, probe_loader, layer_names,
            device=device, max_batches=max_batches, use_amp=use_amp
        )
        for ln in layer_names:
            if neuron_indices is not None and neuron_indices.get(ln) is not None:
                all_reps[ln][task_idx] = reps[ln][:, neuron_indices[ln]]
            else:
                all_reps[ln][task_idx] = reps[ln]
    
    # Compute and plot similarity matrix for each layer
    for layer in layer_names:
        reps_list = [all_reps[layer][t] for t in sorted_task_indices]
        sim_matrix = compute_pairwise_similarity_matrix(reps_list)
        
        safe_layer_name = layer.replace(".", "_").replace("/", "_")
        
        # Plot similarity matrix heatmap
        matrix_path = os.path.join(matrix_dir, f"similarity_matrix_{safe_layer_name}.png")
        plot_similarity_matrix(sim_matrix, sorted_task_indices, layer, matrix_path)
    
    # Plot similarity decay profile for all layers on one figure
    profile_path = os.path.join(output_dir, "similarity_decay_profile.png")
    plot_similarity_decay_profile(all_reps, sorted_task_indices, layer_names, profile_path)
    
    print(f"All model similarity matrices saved to {matrix_dir}")
    print(f"Similarity decay profile saved to {profile_path}")
