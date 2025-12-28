"""Model similarity matrix module.

Computes pairwise similarity between representations from different checkpoint models.
"""
import os
from typing import Dict, List, Optional

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
    plt.savefig(output_path, dpi=150)
    print(f"Similarity matrix saved to {output_path}")
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
        output_path = os.path.join(matrix_dir, f"similarity_matrix_{safe_layer_name}.png")
        
        plot_similarity_matrix(sim_matrix, sorted_task_indices, layer, output_path)
    
    print(f"All model similarity matrices saved to {matrix_dir}")
