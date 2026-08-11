"""Network-health and interference metrics for Experiment B (Sec. 5.3).

Two downstream outcome measures used to decide whether representational drift is
functional or incidental:

  * ``run_network_health``  -- dead-unit fraction + participation ratio per
    checkpoint/layer, computed from the in-memory probe representation cache.
  * ``run_subspace_overlap`` -- cross-task coding-subspace overlap at a fixed
    checkpoint: how much the PCA subspaces recruited by successive tasks share.

Both write a JSON summary (and a simple PDF plot) into the analysis output dir.
"""
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from src.analysis._plot_utils import (
    layer_color_map,
    layer_display_name,
    layer_line_kwargs,
    layer_marker_map,
)


def _participation_ratio(reps: torch.Tensor) -> float:
    """PR = (sum lambda_i)^2 / sum lambda_i^2 of the representation covariance.

    Computed from the N x N Gram matrix (nonzero covariance eigenvalues match),
    which is cheap even when the feature dimension D is very large.
    """
    x = reps.float()
    n = x.shape[0]
    if n < 2:
        return float("nan")
    x = x - x.mean(dim=0, keepdim=True)
    gram = (x @ x.t()) / (n - 1)
    eig = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    s1 = eig.sum()
    s2 = eig.pow(2).sum()
    if s2 <= 0:
        return float("nan")
    return float((s1 * s1) / s2)


def _dead_unit_fraction(reps: torch.Tensor, rel_threshold: float = 1e-2) -> float:
    """Fraction of feature dimensions that are effectively silent over the probe set.

    A unit is "dead" if its mean absolute activation across probe samples is
    below ``rel_threshold`` times the layer's mean activation scale. This is the
    activation-only proxy for the dead-unit marker of plasticity loss.
    """
    x = reps.float().abs()
    per_unit = x.mean(dim=0)
    scale = per_unit.mean().clamp_min(1e-12)
    dead = (per_unit < rel_threshold * scale).float().mean()
    return float(dead)


def run_network_health(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
    dead_rel_threshold: float = 1e-2,
) -> Dict:
    """Per-checkpoint dead-unit fraction + participation ratio (probe set)."""
    task_indices = sorted(reps_cache.keys())
    metrics: Dict[str, Dict[str, list]] = {}

    for ln in layer_names:
        prs, deads = [], []
        for t in task_indices:
            reps = reps_cache[t][ln]
            prs.append(_participation_ratio(reps))
            deads.append(_dead_unit_fraction(reps, rel_threshold=dead_rel_threshold))
        metrics[ln] = {
            "task_indices": task_indices,
            "participation_ratio": prs,
            "dead_unit_fraction": deads,
        }

    out = {
        "dead_rel_threshold": dead_rel_threshold,
        "layers": metrics,
    }
    json_path = os.path.join(output_dir, "health_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Network-health metrics saved to {json_path}")

    # Plots: PR and dead-unit fraction vs checkpoint.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = layer_color_map(layer_names)
    markers = layer_marker_map(layer_names)
    for ln in layer_names:
        ti = metrics[ln]["task_indices"]
        style = layer_line_kwargs(colors[ln], markers[ln])
        axes[0].plot(ti, metrics[ln]["participation_ratio"], label=layer_display_name(ln), **style)
        axes[1].plot(ti, metrics[ln]["dead_unit_fraction"], label=layer_display_name(ln), **style)
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("Participation ratio")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("Dead-unit fraction")
    for ax in axes:
        ax.legend(fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "network_health.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Network-health plot saved to {fig_path}")
    return out


def _subspace_basis(reps: torch.Tensor, var_threshold: float = 0.90) -> torch.Tensor:
    """Orthonormal PCA basis (D x q) capturing ``var_threshold`` of variance."""
    x = reps.float()
    x = x - x.mean(dim=0, keepdim=True)
    # Economy SVD: rows of Vh are principal directions in feature space.
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    var = s.pow(2)
    if var.sum() <= 0:
        return vh[:1].t()
    cum = torch.cumsum(var, dim=0) / var.sum()
    q = int(torch.searchsorted(cum, torch.tensor(var_threshold)).item()) + 1
    q = max(1, min(q, vh.shape[0]))
    return vh[:q].t()  # D x q


def _subspace_overlap(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    """Normalized subspace overlap ||U_a^T U_b||_F^2 / min(q_a, q_b) in [0, 1]."""
    m = basis_a.t() @ basis_b
    qa, qb = basis_a.shape[1], basis_b.shape[1]
    return float(m.pow(2).sum() / max(1, min(qa, qb)))


def run_subspace_overlap(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    increment: int,
    output_dir: str,
    var_threshold: float = 0.90,
    checkpoint: str = "last",
    # Legacy kwargs accepted but ignored (keeps callers compatible during transition)
    **_kwargs,
) -> Dict:
    """Cross-task coding-subspace overlap at a fixed checkpoint.

    Uses the pre-built reps_cache (from step 0) and splits probe samples by
    label to derive per-task PCA subspaces. Reports overlap between successive
    tasks (k, k+1). No additional forward passes or data loading required.
    """
    task_keys = sorted(reps_cache.keys())
    ckpt_idx = task_keys[-1] if checkpoint == "last" else int(checkpoint)

    # Determine task boundaries from labels
    all_labels = labels.numpy()
    num_classes = int(all_labels.max()) + 1
    num_tasks = num_classes // increment

    # Use the chosen checkpoint's representations
    reps_at_ckpt = reps_cache[ckpt_idx]

    # Per-task PCA bases keyed by layer.
    bases: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    for k in range(num_tasks):
        task_classes = set(range(k * increment, (k + 1) * increment))
        mask = torch.tensor([int(lbl) in task_classes for lbl in all_labels], dtype=torch.bool)
        if mask.sum() == 0:
            for ln in layer_names:
                # Degenerate: single-vector basis
                bases[ln].append(torch.zeros(reps_at_ckpt[ln].shape[1], 1))
            continue
        for ln in layer_names:
            task_reps = reps_at_ckpt[ln][mask]
            bases[ln].append(_subspace_basis(task_reps, var_threshold=var_threshold))

    metrics: Dict[str, Dict[str, list]] = {}
    for ln in layer_names:
        overlaps = []
        for k in range(num_tasks - 1):
            overlaps.append(_subspace_overlap(bases[ln][k], bases[ln][k + 1]))
        metrics[ln] = {
            "successive_pairs": [f"{k}-{k + 1}" for k in range(num_tasks - 1)],
            "overlap": overlaps,
            "mean_overlap": float(sum(overlaps) / len(overlaps)) if overlaps else float("nan"),
        }

    out = {
        "checkpoint": ckpt_idx,
        "var_threshold": var_threshold,
        "layers": metrics,
    }
    json_path = os.path.join(output_dir, "subspace_overlap_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Subspace-overlap metrics saved to {json_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = layer_color_map(layer_names)
    markers = layer_marker_map(layer_names)
    for ln in layer_names:
        ax.plot(
            range(num_tasks - 1),
            metrics[ln]["overlap"],
            label=layer_display_name(ln),
            **layer_line_kwargs(colors[ln], markers[ln]),
        )
    ax.set_xlabel("Successive task pair (k, k+1)")
    ax.set_ylabel("Coding-subspace overlap")
    ax.legend(fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "subspace_overlap.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Subspace-overlap plot saved to {fig_path}")
    return out
