"""Sample-wise cosine similarity matrices, sorted by class label."""
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.analysis._plot_utils import apply_paper_axis_style


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
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix_np = sim_matrix.numpy()
    im = ax.imshow(matrix_np, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    if class_boundaries is not None:
        for boundary in class_boundaries:
            ax.axhline(y=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(x=boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Sample Index (sorted by class)")
    ax.set_ylabel("Sample Index (sorted by class)")
    apply_paper_axis_style(ax)
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
                layer_dir, f"sample_sim_task{task_idx}_{safe_layer_name}.pdf"
            )
            plot_sample_similarity_matrix(
                sim_matrix, task_idx, layer, output_path, class_boundaries
            )

    print(f"\nAll sample similarity matrices saved to {sample_sim_dir}")


def _hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for centered kernels."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    return (HKH * HLH).sum().item() / ((n - 1) ** 2)


def linear_cka(S1: torch.Tensor, S2: torch.Tensor) -> float:
    """Linear CKA between two similarity matrices (already kernel matrices)."""
    hsic_12 = _hsic(S1, S2)
    hsic_11 = _hsic(S1, S1)
    hsic_22 = _hsic(S2, S2)
    denom = (hsic_11 * hsic_22) ** 0.5
    if denom < 1e-12:
        return 0.0
    return hsic_12 / denom


def frobenius_norm_diff(S1: torch.Tensor, S2: torch.Tensor) -> float:
    """Frobenius norm of the difference between two matrices, normalized by N^2."""
    diff = S1 - S2
    return (diff * diff).sum().sqrt().item() / S1.shape[0]


def run_sample_similarity_evolution(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
):
    """Track how sample similarity matrix changes relative to task 0.

    For each layer, computes CKA and Frobenius norm between S(task_0) and S(task_t),
    then produces a line plot and saves metrics JSON.
    """
    print("\n" + "=" * 60)
    print("SAMPLE SIMILARITY MATRIX EVOLUTION (CKA & Frobenius)")
    print("=" * 60)

    evo_dir = os.path.join(output_dir, "sample_similarity_evolution")
    os.makedirs(evo_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())
    if len(sorted_task_indices) < 2:
        print("  Not enough checkpoints for evolution analysis.")
        return

    sort_indices = torch.argsort(labels)
    ref_task = sorted_task_indices[0]

    metrics: Dict[str, List[dict]] = {ln: [] for ln in layer_names}

    for layer in layer_names:
        print(f"  Layer: {layer}")
        ref_reps = reps_cache[ref_task][layer][sort_indices]
        S_ref = compute_sample_similarity_matrix(ref_reps)

        for task_idx in sorted_task_indices:
            if task_idx == ref_task:
                cka_val = 1.0
                frob_val = 0.0
            else:
                cur_reps = reps_cache[task_idx][layer][sort_indices]
                S_cur = compute_sample_similarity_matrix(cur_reps)
                cka_val = linear_cka(S_ref, S_cur)
                frob_val = frobenius_norm_diff(S_ref, S_cur)

            metrics[layer].append({
                "task": int(task_idx),
                "cka": round(cka_val, 6),
                "frobenius_norm": round(frob_val, 6),
            })
            print(f"    task {task_idx}: CKA={cka_val:.4f}, Frob={frob_val:.4f}")

    # Save metrics JSON
    metrics_path = os.path.join(evo_dir, "similarity_evolution_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for layer in layer_names:
        tasks = [m["task"] for m in metrics[layer]]
        cka_vals = [m["cka"] for m in metrics[layer]]
        frob_vals = [m["frobenius_norm"] for m in metrics[layer]]
        axes[0].plot(tasks, cka_vals, marker="o", markersize=3, label=layer)
        axes[1].plot(tasks, frob_vals, marker="o", markersize=3, label=layer)

    axes[0].set_xlabel("Task")
    axes[0].set_ylabel("CKA (vs Task 0)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    apply_paper_axis_style(axes[0])

    axes[1].set_xlabel("Task")
    axes[1].set_ylabel("Frobenius Norm (vs Task 0)")
    axes[1].legend(fontsize=8)
    apply_paper_axis_style(axes[1])

    plt.tight_layout()
    plot_path = os.path.join(evo_dir, "similarity_evolution.pdf")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Plot saved to {plot_path}")
