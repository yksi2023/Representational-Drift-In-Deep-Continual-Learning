"""Sample-wise similarity matrix module.

Computes cosine similarity between all sample pairs within each checkpoint model.
"""
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.checkpoints import list_checkpoints, load_model
from src.representations import register_activation_hooks


@torch.no_grad()
def extract_representations_with_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
) -> tuple:
    """
    Extract representations along with labels for sorting by class.
    
    Returns:
        reps: Dict[str, torch.Tensor] - layer_name -> [N, D] representations
        labels: torch.Tensor - [N] labels for each sample
    """
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    all_labels: List[torch.Tensor] = []
    activations, handles = register_activation_hooks(model, layer_names)

    try:
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            all_labels.append(labels)
            
            if use_amp and (device.type == 'cuda'):
                with torch.amp.autocast(device_type=device.type):
                    _ = model(inputs)
            else:
                _ = model(inputs)
            
            for ln in layer_names:
                if ln in activations:
                    collected[ln].append(activations[ln])
            
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    finally:
        for h in handles:
            h.remove()

    reps = {ln: torch.cat(tensors, dim=0) if tensors else torch.empty(0) 
            for ln, tensors in collected.items()}
    labels_tensor = torch.cat(all_labels, dim=0)
    
    return reps, labels_tensor


def compute_sample_similarity_matrix(reps: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between all samples.
    
    Args:
        reps: Tensor of shape [N, D] - representations for N samples
        
    Returns:
        Similarity matrix of shape [N, N]
    """
    reps_norm = F.normalize(reps, p=2, dim=1)
    sim_matrix = torch.mm(reps_norm, reps_norm.t())
    return sim_matrix


def plot_sample_similarity_matrix(
    sim_matrix: torch.Tensor,
    labels: torch.Tensor,
    task_idx: int,
    layer_name: str,
    output_path: str,
    class_boundaries: Optional[List[int]] = None
):
    """
    Plot sample-wise similarity matrix as a colormap.
    
    Args:
        sim_matrix: [N, N] similarity matrix
        labels: [N] labels for each sample (used for boundary lines)
        task_idx: Task index for title
        layer_name: Layer name for title
        output_path: Path to save the figure
        class_boundaries: Optional list of indices where class boundaries occur
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom")
    
    # Draw class boundary lines if provided
    if class_boundaries is not None:
        for boundary in class_boundaries:
            ax.axhline(y=boundary - 0.5, color='black', linewidth=0.5, alpha=0.5)
            ax.axvline(x=boundary - 0.5, color='black', linewidth=0.5, alpha=0.5)
    
    ax.set_title(f'Sample Similarity Matrix - Model after Task {task_idx}\nLayer: {layer_name}')
    ax.set_xlabel('Sample Index (sorted by class)')
    ax.set_ylabel('Sample Index (sorted by class)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_sample_similarity(
    model: torch.nn.Module,
    probe_loader: DataLoader,
    ckpt_dir: str,
    layer_names: List[str],
    output_dir: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    neuron_indices: Optional[Dict[str, Optional[torch.Tensor]]] = None,
):
    """
    Analyze sample-wise similarity matrices for each checkpoint model.
    
    For each model checkpoint, extracts representations for probe data,
    computes sample-wise cosine similarity, and plots the matrix.
    Samples are sorted by class label to ensure consistent ordering.
    
    Args:
        model: Model to analyze
        probe_loader: DataLoader for probe data (must have shuffle=False)
        ckpt_dir: Directory containing checkpoints
        layer_names: List of layer names to analyze
        output_dir: Directory to save results
        device: Torch device
        max_batches: Maximum batches to process
        use_amp: Use automatic mixed precision
        neuron_indices: Dict mapping layer name to sampled neuron indices
    """
    print("\n" + "="*60)
    print("GENERATING SAMPLE-WISE SIMILARITY MATRICES")
    print("="*60)
    
    sample_sim_dir = os.path.join(output_dir, "sample_similarity_matrices")
    os.makedirs(sample_sim_dir, exist_ok=True)
    
    ckpts = list_checkpoints(ckpt_dir)
    sorted_task_indices = sorted(ckpts.keys())
    
    # First pass: extract labels and determine sort order (only once)
    print("Extracting labels to determine sample order...")
    load_model(model, ckpt_dir, sorted_task_indices[0], map_location=device)
    _, labels = extract_representations_with_labels(
        model, probe_loader, layer_names[:1],
        device=device, max_batches=max_batches, use_amp=use_amp
    )
    
    # Sort indices by class label
    sort_indices = torch.argsort(labels)
    sorted_labels = labels[sort_indices]
    
    # Find class boundaries for visualization
    unique_labels = sorted_labels.unique()
    class_boundaries = []
    for lbl in unique_labels[1:]:
        boundary_idx = (sorted_labels == lbl).nonzero(as_tuple=True)[0][0].item()
        class_boundaries.append(boundary_idx)
    
    print(f"Total samples: {len(labels)}, Classes: {len(unique_labels)}")
    print(f"Class boundaries at indices: {class_boundaries}")
    
    # Process each checkpoint
    for task_idx in sorted_task_indices:
        print(f"\nProcessing model after Task {task_idx}...")
        load_model(model, ckpt_dir, task_idx, map_location=device)
        
        reps, _ = extract_representations_with_labels(
            model, probe_loader, layer_names,
            device=device, max_batches=max_batches, use_amp=use_amp
        )
        
        for layer in layer_names:
            layer_reps = reps[layer]
            
            # Apply neuron sampling if needed
            if neuron_indices is not None and neuron_indices.get(layer) is not None:
                layer_reps = layer_reps[:, neuron_indices[layer]]
            
            # Sort representations by class
            sorted_reps = layer_reps[sort_indices]
            
            # Compute sample similarity matrix
            sim_matrix = compute_sample_similarity_matrix(sorted_reps)
            
            # Plot and save
            safe_layer_name = layer.replace(".", "_").replace("/", "_")
            
            # Create layer-specific directory
            layer_dir = os.path.join(sample_sim_dir, safe_layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            output_path = os.path.join(
                layer_dir, 
                f"sample_sim_task{task_idx}_{safe_layer_name}.png"
            )
            plot_sample_similarity_matrix(
                sim_matrix, sorted_labels, task_idx, layer, output_path, class_boundaries
            )
            print(f"  Saved: {output_path}")
    
    print(f"\nAll sample similarity matrices saved to {sample_sim_dir}")
