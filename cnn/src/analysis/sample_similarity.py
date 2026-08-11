"""Sample-wise cosine similarity matrices, sorted by class label."""
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.analysis._plot_utils import (
    SINGLE_FIGSIZE,
    WIDE_FIGSIZE,
    apply_paper_axis_style,
    layer_color_map,
    layer_errorbar_kwargs,
    layer_display_name,
    layer_line_kwargs,
    layer_marker_map,
    savefig_compact,
    sparse_ticks,
    sparse_value_ticks,
)


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
    matrix_np = sim_matrix.detach().cpu().numpy() if torch.is_tensor(sim_matrix) else np.asarray(sim_matrix)
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


def plot_sample_sim_cka_matrix(
    cka_matrix: np.ndarray,
    output_path: str,
):
    """Plot T×T CKA matrix between checkpoint sample-similarity matrices."""
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(cka_matrix.shape[0])
    ax.set_xticks(sp); ax.set_xticklabels(sl)
    ax.set_yticks(sp); ax.set_yticklabels(sl)
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Model after Task")
    apply_paper_axis_style(ax)
    savefig_compact(fig, output_path)
    plt.close()


def compute_pairwise_sample_sim_cka(
    S_by_task: Dict[int, torch.Tensor],
    sorted_task_indices: List[int],
) -> np.ndarray:
    """Full T×T CKA matrix over sample-similarity matrices (symmetric, diag=1)."""
    n = len(sorted_task_indices)
    cka_mat = np.eye(n, dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            r = linear_cka(
                S_by_task[sorted_task_indices[a]],
                S_by_task[sorted_task_indices[b]],
            )
            cka_mat[a, b] = cka_mat[b, a] = r
    return cka_mat


def run_sample_similarity(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
):
    """Plot sample-wise similarity matrix for every (checkpoint, layer).

    Also saves ``.npy`` matrices for cross-seed aggregation.
    """
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

    with open(os.path.join(sample_sim_dir, "class_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "boundaries": class_boundaries,
                "n_samples": int(len(labels)),
                "n_classes": int(len(unique_labels)),
            },
            f,
            indent=2,
        )

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
            stem = f"sample_sim_task{task_idx}_{safe_layer_name}"
            output_path = os.path.join(layer_dir, f"{stem}.pdf")
            npy_path = os.path.join(layer_dir, f"{stem}.npy")
            plot_sample_similarity_matrix(
                sim_matrix, task_idx, layer, output_path, class_boundaries
            )
            np.save(npy_path, sim_matrix.detach().cpu().numpy().astype(np.float32))

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


def _compute_gap_metrics(
    S_by_task: Dict[int, torch.Tensor],
    sorted_task_indices: List[int],
):
    """Return (gaps, cka_means, cka_stds, frob_means, frob_stds) over all task pairs,
    grouped by gap = j - i.
    """
    gap_to_cka: Dict[int, List[float]] = {}
    gap_to_frob: Dict[int, List[float]] = {}
    for a_i, i in enumerate(sorted_task_indices):
        for b_i in range(a_i + 1, len(sorted_task_indices)):
            j = sorted_task_indices[b_i]
            gap = j - i
            gap_to_cka.setdefault(gap, []).append(linear_cka(S_by_task[i], S_by_task[j]))
            gap_to_frob.setdefault(gap, []).append(frobenius_norm_diff(S_by_task[i], S_by_task[j]))

    gaps = sorted(gap_to_cka.keys())
    cka_means = [round(float(np.mean(gap_to_cka[g])), 6) for g in gaps]
    cka_stds = [round(float(np.std(gap_to_cka[g])), 6) for g in gaps]
    frob_means = [round(float(np.mean(gap_to_frob[g])), 6) for g in gaps]
    frob_stds = [round(float(np.std(gap_to_frob[g])), 6) for g in gaps]
    return gaps, cka_means, cka_stds, frob_means, frob_stds


def _plot_gap_metric(
    all_results: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    ylabel: str,
    output_path: str,
):
    """Plot a gap-indexed metric (CKA or Frobenius norm) for all layers on one figure."""
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    all_gaps: List[int] = []
    colors = layer_color_map(list(all_results))
    markers = layer_marker_map(list(all_results))

    for layer, data in all_results.items():
        gaps = data["gaps"]
        means = data[f"{metric_key}_means"]
        stds = data[f"{metric_key}_stds"]
        all_gaps.extend(gaps)
        ax.errorbar(
            gaps, means, yerr=stds, label=layer_display_name(layer),
            **layer_errorbar_kwargs(colors[layer], markers[layer]),
        )

    ax.set_xlabel("Task Gap")
    ax.set_ylabel(ylabel)
    apply_paper_axis_style(ax, legend=True, legend_kwargs={"fontsize": 8})
    ax.grid(True, linestyle="--", alpha=0.3)
    if all_gaps:
        ticks, labels = sparse_value_ticks(all_gaps)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_sample_similarity_evolution(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
):
    """Track how the sample similarity matrix changes over training.

    For each layer, computes CKA and Frobenius norm:
      1. Relative to Task 1 (the first checkpoint), vs. task index.
      2. Between every pair of checkpoints, grouped by task gap.
    Saves both metrics JSON files and line plots.
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
    gap_metrics: Dict[str, Dict[str, List[float]]] = {}

    for layer in layer_names:
        print(f"  Layer: {layer}")
        S_by_task = {
            t: compute_sample_similarity_matrix(reps_cache[t][layer][sort_indices])
            for t in sorted_task_indices
        }
        S_ref = S_by_task[ref_task]

        for task_idx in sorted_task_indices:
            if task_idx == ref_task:
                cka_val = 1.0
                frob_val = 0.0
            else:
                cka_val = linear_cka(S_ref, S_by_task[task_idx])
                frob_val = frobenius_norm_diff(S_ref, S_by_task[task_idx])

            metrics[layer].append({
                "task": int(task_idx),
                "cka": round(cka_val, 6),
                "frobenius_norm": round(frob_val, 6),
            })
            print(f"    task {task_idx}: CKA={cka_val:.4f}, Frob={frob_val:.4f}")

        gaps, cka_means, cka_stds, frob_means, frob_stds = _compute_gap_metrics(
            S_by_task, sorted_task_indices
        )
        gap_metrics[layer] = {
            "gaps": gaps,
            "cka_means": cka_means,
            "cka_stds": cka_stds,
            "frobenius_norm_means": frob_means,
            "frobenius_norm_stds": frob_stds,
        }
        if gaps:
            print(f"    gap=1 CKA={cka_means[0]:.4f}, gap={gaps[-1]} CKA={cka_means[-1]:.4f}")

        # Full pairwise CKA matrix over sample-similarity matrices
        safe_layer = layer.replace(".", "_").replace("/", "_")
        cka_mat = compute_pairwise_sample_sim_cka(S_by_task, sorted_task_indices)
        cka_npy = os.path.join(evo_dir, f"sample_sim_cka_matrix_{safe_layer}.npy")
        cka_pdf = os.path.join(evo_dir, f"sample_sim_cka_matrix_{safe_layer}.pdf")
        np.save(cka_npy, cka_mat.astype(np.float32))
        plot_sample_sim_cka_matrix(cka_mat, cka_pdf)
        print(f"    CKA matrix saved to {cka_pdf}")

    # Save metrics JSON (vs Task 1)
    metrics_path = os.path.join(evo_dir, "similarity_evolution_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # Save gap-indexed metrics JSON
    gap_metrics_path = os.path.join(evo_dir, "similarity_evolution_gap_metrics.json")
    with open(gap_metrics_path, "w", encoding="utf-8") as f:
        json.dump(gap_metrics, f, ensure_ascii=False, indent=2)
    print(f"  Gap-indexed metrics saved to {gap_metrics_path}")

    # Plot: vs Task 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = layer_color_map(layer_names)
    markers = layer_marker_map(layer_names)

    for layer in layer_names:
        tasks = [m["task"] for m in metrics[layer]]
        cka_vals = [m["cka"] for m in metrics[layer]]
        frob_vals = [m["frobenius_norm"] for m in metrics[layer]]
        axes[0].plot(
            tasks, cka_vals, label=layer_display_name(layer),
            **layer_line_kwargs(colors[layer], markers[layer]),
        )
        axes[1].plot(
            tasks, frob_vals, label=layer_display_name(layer),
            **layer_line_kwargs(colors[layer], markers[layer]),
        )

    axes[0].set_xlabel("Task")
    axes[0].set_ylabel("CKA relative to Task 1")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    apply_paper_axis_style(axes[0])
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Task")
    axes[1].set_ylabel("Frobenius Norm relative to Task 1")
    axes[1].legend(fontsize=8)
    apply_paper_axis_style(axes[1])
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(evo_dir, "similarity_evolution.pdf")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Plot saved to {plot_path}")

    # Plot: vs Task Gap
    _plot_gap_metric(
        gap_metrics, "cka", "CKA",
        os.path.join(evo_dir, "similarity_evolution_gap_cka.pdf"),
    )
    _plot_gap_metric(
        gap_metrics, "frobenius_norm", "Frobenius Norm",
        os.path.join(evo_dir, "similarity_evolution_gap_frobenius_norm.pdf"),
    )
    print(f"  Gap-indexed plots saved to {evo_dir}")
