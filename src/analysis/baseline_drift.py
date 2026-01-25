"""Baseline drift analysis module.

Computes drift metrics by comparing representations from each checkpoint
against the baseline (first task) model.
"""
import json
import os
import random
from typing import Dict, List, Optional

import torch
import matplotlib.pyplot as plt

from src.checkpoints import list_checkpoints, load_model
from src.representations import extract_representations
from src.analysis.drift_metrics import compute_metrics


def plot_drift_results(results: List[Dict], output_dir: str):
    """Plot drift metrics and save to file."""
    tasks = [r['target_task'] for r in results]
    layers = sorted(list(set(r['layer'] for r in results)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for layer in layers:
        layer_data = [r for r in results if r['layer'] == layer]
        layer_data.sort(key=lambda x: x['target_task'])
        
        xs = [d['target_task'] for d in layer_data]
        
        # Cosine Sim
        cos_means = [d['cosine_sim_mean'] for d in layer_data]
        cos_stds = [d['cosine_sim_std'] for d in layer_data]
        line = ax1.errorbar(xs, cos_means, yerr=cos_stds, label=f"{layer}", capsize=5, marker='o')
        
        # Shuffled Baseline (Same color, dashed)
        shuffled_means = [d['shuffled_sim_mean'] for d in layer_data]
        ax1.plot(xs, shuffled_means, linestyle='--', color=line[0].get_color(), alpha=0.5, label=f"{layer} (Random)")
        
        # L2 Dist
        l2_means = [d['l2_dist_mean'] for d in layer_data]
        l2_stds = [d['l2_dist_std'] for d in layer_data]
        ax2.errorbar(xs, l2_means, yerr=l2_stds, label=layer, capsize=5, marker='o')

    ax1.set_title("Cosine Similarity (Solid) vs Shuffled Baseline (Dashed)")
    ax1.set_xlabel("Task Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title("L2 Distance Drift")
    ax2.set_xlabel("Task Index")
    ax2.set_ylabel("L2 Distance")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "drift_plot.png")
    plt.savefig(output_path)
    print(f"Drift plot saved to {output_path}")
    plt.close()


def plot_activation_distribution(
    vec_base: torch.Tensor, 
    vec_curr: torch.Tensor, 
    layer_name: str, 
    task_idx: int, 
    output_dir: str,
    num_samples: int = 3
):
    """
    Plot histogram of activation values for a few sample vectors.
    """
    # Detach and CPU
    vec_base = vec_base.detach().cpu()
    vec_curr = vec_curr.detach().cpu()
    
    # Randomly select a few samples to visualize
    N = vec_base.shape[0]
    indices = torch.randperm(N)[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1: axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        v_b = vec_base[idx].numpy().flatten()
        v_c = vec_curr[idx].numpy().flatten()
        
        # Plot histograms
        ax.hist(v_b, bins=50, alpha=0.5, label='Baseline', density=True, color='blue')
        ax.hist(v_c, bins=50, alpha=0.5, label=f'Task {task_idx}', density=True, color='orange')
        
        ax.set_title(f"Sample {idx.item()} - Activation Distribution")
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    # Save per layer/task
    save_dir = os.path.join(output_dir, "distributions", layer_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"task_{task_idx}.png"))
    plt.close()


def run_baseline_drift(
    model: torch.nn.Module,
    probe_loader: torch.utils.data.DataLoader,
    ckpt_dir: str,
    layer_names: List[str],
    output_dir: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    neuron_ratio: float = 1.0,
    seed: int = 42,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Run baseline drift analysis.
    
    Compares representations from each checkpoint against the baseline model.
    
    Args:
        model: Model to analyze
        probe_loader: DataLoader for probe data
        ckpt_dir: Directory containing checkpoints
        layer_names: List of layer names to analyze
        output_dir: Directory to save results
        device: Torch device
        max_batches: Maximum batches to process
        use_amp: Use automatic mixed precision
        neuron_ratio: Ratio of neurons to sample (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        neuron_indices: Dict mapping layer name to sampled neuron indices (or None if ratio=1.0)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    drift_plot_path = os.path.join(output_dir, "drift_plot.png")
    
    ckpts = list_checkpoints(ckpt_dir)
    sorted_task_indices = sorted(ckpts.keys())
    baseline_idx = sorted_task_indices[0]
    
    print(f"Using Task {baseline_idx} as baseline.")
    
    # Extract baseline features
    load_model(model, ckpt_dir, baseline_idx, map_location=device)
    print(f"Extracting baseline features from Task {baseline_idx}...")
    baseline_reps = extract_representations(
        model, probe_loader, layer_names, 
        device=device, max_batches=max_batches, use_amp=use_amp
    )
    
    # Sample neurons if ratio < 1.0
    neuron_indices: Dict[str, Optional[torch.Tensor]] = {}
    if neuron_ratio < 1.0:
        print(f"Randomly sampling {neuron_ratio*100:.1f}% of neurons for each layer...")
        for layer in layer_names:
            num_neurons = baseline_reps[layer].shape[1]
            num_sample = max(1, int(num_neurons * neuron_ratio))
            indices = torch.tensor(random.sample(range(num_neurons), num_sample))
            neuron_indices[layer] = indices
            print(f"  {layer}: {num_sample}/{num_neurons} neurons selected")
            baseline_reps[layer] = baseline_reps[layer][:, indices]
    else:
        for layer in layer_names:
            neuron_indices[layer] = None
    
    # Compare subsequent tasks
    results = []
    
    # Initial point: Baseline compares to itself
    for layer in layer_names:
        feat_base = baseline_reps[layer]
        # Note: baseline_reps is already subsampled if neuron_ratio < 1.0
        
        metrics = compute_metrics(feat_base, feat_base)
        
        plot_activation_distribution(
            feat_base, feat_base,
            layer_name=layer,
            task_idx=baseline_idx,
            output_dir=output_dir
        )
        
        results.append({
            "baseline_task": baseline_idx,
            "target_task": baseline_idx,
            "layer": layer,
            **metrics
        })

    for task_idx in sorted_task_indices:
        if task_idx == baseline_idx:
            continue
            
        print(f"Comparing Task {task_idx} against Baseline...")
        load_model(model, ckpt_dir, task_idx, map_location=device)
        
        current_reps = extract_representations(
            model, probe_loader, layer_names, 
            device=device, max_batches=max_batches, use_amp=use_amp
        )
        
        for layer in layer_names:
            feat_base = baseline_reps[layer]
            feat_curr = current_reps[layer]
            if neuron_indices[layer] is not None:
                feat_curr = feat_curr[:, neuron_indices[layer]]
            
            metrics = compute_metrics(feat_base, feat_curr)
            
            plot_activation_distribution(
                feat_base, feat_curr,
                layer_name=layer,
                task_idx=task_idx,
                output_dir=output_dir
            )
            
            results.append({
                "baseline_task": baseline_idx,
                "target_task": task_idx,
                "layer": layer,
                **metrics
            })

    # Save statistics and plot
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")

    plot_drift_results(results, output_dir)
    
    return neuron_indices
