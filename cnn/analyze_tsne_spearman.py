"""
Analyze representational drift using t-SNE and Spearman correlation of sample similarity matrices.

This script implements:
1. t-SNE analysis of representation vectors for different samples (colored by class).
2. Spearman rank correlation heatmap between sample similarity matrices of different checkpoints.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

from src.models import FashionMNISTModel, ResNet18_Tiny, PretrainedResNet18
from src.checkpoints import list_checkpoints, load_model
from src.analysis.sample_similarity import extract_representations_with_labels, compute_sample_similarity_matrix
from datasets import IncrementalFashionMNIST, IncrementalTinyImageNet

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze t-SNE and Spearman correlation of sample similarity")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="network.1") 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save all analysis results")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--probe_type", type=str, default="test", choices=["test","train"])
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE")
    return parser.parse_args()

def setup_environment(args):
    """Initialize model, data manager, and common settings."""
    # Initialize data manager and model
    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST()
        model = FashionMNISTModel(output_size=10)
    elif args.dataset == "tiny_imagenet":
        data_manager = IncrementalTinyImageNet()
        model = PretrainedResNet18(num_classes=200)
    else:
        raise ValueError("Invalid dataset")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "tsne_spearman_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Parse layer names
    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]
    
    # Read metadata to determine increment
    meta_path = os.path.join(args.ckpt_dir, "model_after_task_1.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find baseline metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        increment = json.load(f)["training_params"]["increment"]
    
    # Create probe loader (shuffle=False for consistent sample order)
    probe_loader = data_manager.get_loader(
        mode=args.probe_type, 
        label=range(increment),
        batch_size=args.batch_size,
        shuffle=False 
    )
    
    # Verify checkpoints exist
    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")
    
    return model, probe_loader, layer_names, device, ckpts

def plot_tsne(reps: torch.Tensor, labels: torch.Tensor, task_idx: int, layer_name: str, output_dir: str, perplexity: float = 30.0):
    """Computes and plots t-SNE for the given representations."""
    print(f"  Computing t-SNE for Task {task_idx}, Layer {layer_name}...")
    
    # Convert to numpy
    reps_np = reps.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reps_embedded = tsne.fit_transform(reps_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reps_embedded[:, 0], reps_embedded[:, 1], c=labels_np, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f't-SNE - Task {task_idx} - {layer_name}')
    
    safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
    output_path = os.path.join(output_dir, f"tsne_task{task_idx}_{safe_layer_name}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved t-SNE plot to {output_path}")

def get_upper_tri_flat(matrix: torch.Tensor) -> np.ndarray:
    """Extracts the flattened upper triangle of a matrix (excluding diagonal)."""
    # Get indices for upper triangle, offset=1 to exclude diagonal
    # Using torch.triu_indices might be better but matrix is square
    n = matrix.shape[0]
    rows, cols = torch.triu_indices(n, n, offset=1)
    return matrix[rows, cols].cpu().numpy()

def main():
    args = parse_args()
    model, probe_loader, layer_names, device, ckpts = setup_environment(args)
    
    sorted_task_indices = sorted(ckpts.keys())
    
    # Storage for similarity vectors: layer -> {task_idx -> flat_vector}
    sim_vectors: Dict[str, Dict[int, np.ndarray]] = {ln: {} for ln in layer_names}
    
    print(f"Starting analysis for {len(sorted_task_indices)} checkpoints...")
    print(f"Layers: {layer_names}")
    
    # Process each checkpoint
    for task_idx in sorted_task_indices:
        print(f"\nProcessing model after Task {task_idx}...")
        load_model(model, args.ckpt_dir, task_idx, map_location=device)
        
        reps_dict, labels = extract_representations_with_labels(
            model, probe_loader, layer_names,
            device=device, max_batches=args.max_batches, use_amp=args.amp
        )
        
        for layer in layer_names:
            reps = reps_dict[layer]
            
            # 1. t-SNE Analysis
            tsne_dir = os.path.join(args.output_dir, "tsne")
            os.makedirs(tsne_dir, exist_ok=True)
            plot_tsne(reps, labels, task_idx, layer, tsne_dir, args.perplexity)
            
            # 2. Sample Similarity & Spearman Prep
            # Compute sample similarity matrix
            sim_matrix = compute_sample_similarity_matrix(reps)
            
            # Flatten upper triangle
            flat_sim = get_upper_tri_flat(sim_matrix)
            sim_vectors[layer][task_idx] = flat_sim
            
            print(f"  Stored similarity vector for Layer {layer} (shape: {flat_sim.shape})")

    # 3. Compute and Plot Spearman Correlations
    print("\nComputing Spearman Rank Correlations...")
    spearman_dir = os.path.join(args.output_dir, "spearman_heatmap")
    os.makedirs(spearman_dir, exist_ok=True)
    
    for layer in layer_names:
        print(f"Generating heatmap for {layer}...")
        
        tasks = sorted_task_indices
        n_tasks = len(tasks)
        spearman_matrix = np.zeros((n_tasks, n_tasks))
        
        # Calculate pairwise Spearman correlations
        for i, task_i in enumerate(tasks):
            vec_i = sim_vectors[layer][task_i]
            for j, task_j in enumerate(tasks):
                if i == j:
                    spearman_matrix[i, j] = 1.0
                elif i < j: # Symmetric
                    vec_j = sim_vectors[layer][task_j]
                    corr, _ = spearmanr(vec_i, vec_j)
                    spearman_matrix[i, j] = corr
                    spearman_matrix[j, i] = corr
        
        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(spearman_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Spearman Correlation', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels([f'T{t}' for t in tasks])
        ax.set_yticklabels([f'T{t}' for t in tasks])
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                text = ax.text(j, i, f'{spearman_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'Spearman Rank Correlation of Sample Similarity\nLayer: {layer}')
        ax.set_xlabel('Model after Task')
        ax.set_ylabel('Model after Task')
        
        plt.tight_layout()
        
        safe_layer_name = layer.replace(".", "_").replace("/", "_")
        output_path = os.path.join(spearman_dir, f"spearman_heatmap_{safe_layer_name}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved heatmap to {output_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
