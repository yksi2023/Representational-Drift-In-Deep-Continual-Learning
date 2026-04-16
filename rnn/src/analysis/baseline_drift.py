"""Baseline STPV drift analysis module for RNN.

Computes drift metrics by comparing Spatiotemporal Population Vectors (STPVs)
from each checkpoint against the baseline (first task) model.
STPV = concatenation of Population Vectors across all time steps.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.drift_metrics import compute_metrics


def _load_reps_from_npz(reps_dir: str, probe_task: str) -> Dict[int, np.ndarray]:
    """Load STPVs for a probe task from saved .npz file.

    Returns:
        Dict mapping task_idx -> np.ndarray of shape [N, D].
    """
    npz_path = os.path.join(reps_dir, f"{probe_task}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Representations not found: {npz_path}")
    data = np.load(npz_path)
    reps = {}
    for key in sorted(data.files, key=lambda k: int(k.split("_")[-1])):
        idx = int(key.split("_")[-1])
        reps[idx] = data[key]
    return reps


def plot_drift_results(results: List[Dict], task_names: List[str], output_dir: str):
    """Plot drift metrics and save to file."""
    probe_tasks = sorted(set(r['probe_task'] for r in results))

    for probe_task in probe_tasks:
        task_results = [r for r in results if r['probe_task'] == probe_task]
        tasks_idx = sorted(set(r['target_task'] for r in task_results))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        cos_means = [next(r for r in task_results if r['target_task'] == t)['cosine_sim_mean'] for t in tasks_idx]
        cos_stds = [next(r for r in task_results if r['target_task'] == t)['cosine_sim_std'] for t in tasks_idx]
        shuffled_means = [next(r for r in task_results if r['target_task'] == t)['shuffled_sim_mean'] for t in tasks_idx]

        line = ax1.errorbar(tasks_idx, cos_means, yerr=cos_stds, label='Cosine Sim', capsize=5, marker='o')
        ax1.plot(tasks_idx, shuffled_means, linestyle='--', color=line[0].get_color(), alpha=0.5, label='Shuffled Baseline')
        ax1.set_title(f"STPV Cosine Similarity Drift — probe: {probe_task}")
        ax1.set_xlabel("Trained Task Index")
        ax1.set_ylabel("Cosine Similarity")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        l2_means = [next(r for r in task_results if r['target_task'] == t)['l2_dist_mean'] for t in tasks_idx]
        l2_stds = [next(r for r in task_results if r['target_task'] == t)['l2_dist_std'] for t in tasks_idx]
        ax2.errorbar(tasks_idx, l2_means, yerr=l2_stds, label='L2 Distance', capsize=5, marker='o')
        ax2.set_title(f"STPV L2 Distance Drift — probe: {probe_task}")
        ax2.set_xlabel("Trained Task Index")
        ax2.set_ylabel("L2 Distance")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Use task names as x-tick labels if available
        if task_names and len(task_names) == len(tasks_idx):
            ax1.set_xticks(tasks_idx)
            ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
            ax2.set_xticks(tasks_idx)
            ax2.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"drift_plot_{probe_task}.png")
        plt.savefig(output_path)
        print(f"  Drift plot saved to {output_path}")
        plt.close()


def run_baseline_drift(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
    neuron_ratio: float = 1.0,
    seed: int = 42,
) -> None:
    """
    Run baseline STPV drift analysis.

    Compares STPVs from each checkpoint against the baseline
    (after training on the first task).

    Args:
        exp_dir: Experiment directory containing representations/ subdirectory.
        probe_tasks: Which tasks' representations to analyze for drift.
        task_names: Ordered list of all task names in the sequence.
        output_dir: Directory to save results.
        neuron_ratio: Ratio of neurons to sample (0.0-1.0).
        seed: Random seed for neuron sampling.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    reps_dir = os.path.join(exp_dir, "representations")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for probe_task in probe_tasks:
        print(f"  Analyzing drift for probe task: {probe_task}")
        reps_dict = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(reps_dict.keys())
        baseline_idx = sorted_indices[0]

        baseline = torch.from_numpy(reps_dict[baseline_idx]).float()

        # Neuron subsampling
        neuron_idx = None
        if neuron_ratio < 1.0:
            D = baseline.shape[1]
            num_sample = max(1, int(D * neuron_ratio))
            neuron_idx = torch.tensor(random.sample(range(D), num_sample))
            baseline = baseline[:, neuron_idx]
            print(f"    Subsampled {num_sample}/{D} neurons")

        for task_idx in sorted_indices:
            current = torch.from_numpy(reps_dict[task_idx]).float()
            if neuron_idx is not None:
                current = current[:, neuron_idx]

            metrics = compute_metrics(baseline, current)
            all_results.append({
                "probe_task": probe_task,
                "baseline_task": baseline_idx,
                "target_task": task_idx,
                **metrics,
            })

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "baseline_drift_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Plot
    plot_drift_results(all_results, task_names, output_dir)
