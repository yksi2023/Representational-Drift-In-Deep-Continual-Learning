"""Baseline drift analysis.

Compares representations from each checkpoint against the baseline (first
checkpoint) for every layer, reporting cosine similarity, L2 distance, and a
shuffled-baseline control.
"""
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.drift_metrics import compute_metrics


def plot_drift_results(results: List[Dict], output_dir: str):
    """Plot drift metrics and save to file."""
    tasks = [r["target_task"] for r in results]
    layers = sorted(list(set(r["layer"] for r in results)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    max_task = max(tasks) if tasks else 0
    xticks = np.arange(0, max_task + 4, 4)

    for layer in layers:
        layer_data = [r for r in results if r["layer"] == layer]
        layer_data.sort(key=lambda x: x["target_task"])

        xs = [d["target_task"] for d in layer_data]

        cos_means = [d["cosine_sim_mean"] for d in layer_data]
        cos_stds = [d["cosine_sim_std"] for d in layer_data]
        line = ax1.errorbar(xs, cos_means, yerr=cos_stds, label=f"{layer}", capsize=5, marker="o")

        shuffled_means = [d["shuffled_sim_mean"] for d in layer_data]
        ax1.plot(xs, shuffled_means, linestyle="--", color=line[0].get_color(), alpha=0.5,
                 label=f"{layer} (Random)")

        l2_means = [d["l2_dist_mean"] for d in layer_data]
        l2_stds = [d["l2_dist_std"] for d in layer_data]
        ax2.errorbar(xs, l2_means, yerr=l2_stds, label=layer, capsize=5, marker="o")

    ax1.set_title("Cosine Similarity (Solid) vs Shuffled Baseline (Dashed)")
    ax1.set_xlabel("Task Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_xticks(xticks)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("L2 Distance Drift")
    ax2.set_xlabel("Task Index")
    ax2.set_ylabel("L2 Distance")
    ax2.set_xticks(xticks)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

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
    num_samples: int = 3,
):
    """Plot histogram of activation values for a few sample vectors."""
    vec_base = vec_base.detach().cpu()
    vec_curr = vec_curr.detach().cpu()

    N = vec_base.shape[0]
    indices = torch.randperm(N)[:num_samples]

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        v_b = vec_base[idx].numpy().flatten()
        v_c = vec_curr[idx].numpy().flatten()

        ax.hist(v_b, bins=50, alpha=0.5, label="Baseline", density=True, color="blue")
        ax.hist(v_c, bins=50, alpha=0.5, label=f"Task {task_idx}", density=True, color="orange")

        ax.set_title(f"Sample {idx.item()} - Activation Distribution")
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir = os.path.join(output_dir, "distributions", layer_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"task_{task_idx}.png"))
    plt.close()


def run_baseline_drift(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
    plot_distributions: bool = True,
) -> None:
    """Run baseline drift analysis from pre-extracted representation cache.

    Args:
        reps_cache: {task_idx: {layer: Tensor(N, D)}} from build_reps_cache.
        layer_names: Layers to include.
        output_dir: Directory to save results.
        plot_distributions: Whether to save per-layer/task activation histograms.
    """
    metrics_path = os.path.join(output_dir, "metrics.json")

    sorted_task_indices = sorted(reps_cache.keys())
    baseline_idx = sorted_task_indices[0]
    print(f"Using Task {baseline_idx} as baseline.")

    results = []
    for layer in layer_names:
        feat_base = reps_cache[baseline_idx][layer]
        metrics = compute_metrics(feat_base, feat_base)
        if plot_distributions:
            plot_activation_distribution(feat_base, feat_base, layer, baseline_idx, output_dir)
        results.append({
            "baseline_task": baseline_idx,
            "target_task": baseline_idx,
            "layer": layer,
            **metrics,
        })

    for task_idx in sorted_task_indices:
        if task_idx == baseline_idx:
            continue
        print(f"Comparing Task {task_idx} against baseline...")
        for layer in layer_names:
            feat_base = reps_cache[baseline_idx][layer]
            feat_curr = reps_cache[task_idx][layer]
            metrics = compute_metrics(feat_base, feat_curr)
            if plot_distributions:
                plot_activation_distribution(feat_base, feat_curr, layer, task_idx, output_dir)
            results.append({
                "baseline_task": baseline_idx,
                "target_task": task_idx,
                "layer": layer,
                **metrics,
            })

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_path}")

    plot_drift_results(results, output_dir)
