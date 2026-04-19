"""Coding / null subspace drift decomposition.

Adapted from ``rnn/src/analysis/subspace_drift.py``. For each layer:

1. PCA the baseline (first checkpoint) representations.
2. Define the *coding subspace* as the smallest top-k principal components
   whose cumulative explained variance exceeds ``threshold``.
3. Decompose drift vectors ``delta = rep_t - rep_0`` from each subsequent
   checkpoint into coding-subspace and null-subspace components.
4. Per-checkpoint PCA: track coding dimensionality k and participation ratio.

Outputs per layer:
- ``coding_fraction_<layer>.png``: stacked bar showing coding vs null drift
  variance fraction across checkpoints.
- ``drift_norms_<layer>.png``: mean L2 norm of coding / null components.
- ``variance_explained_<layer>.png``: baseline PCA cumulative variance curve.
- ``dimensionality_<layer>.png``: per-checkpoint k and participation ratio.
- ``subspace_metrics_<layer>.json``: numeric summary.
"""
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _determine_k(evr: np.ndarray, threshold: float) -> int:
    cumvar = np.cumsum(evr)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, len(evr))


def _compute_subspace_drift(
    reps_by_task: Dict[int, torch.Tensor],
    sorted_indices: List[int],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Decompose drift into coding and null subspace components.

    Returns:
        coding_fractions: (T,) fraction of drift variance in coding subspace
            (NaN for the baseline index).
        coding_norms: (T,) mean L2 norm of coding drift component.
        null_norms: (T,) mean L2 norm of null drift component.
        k: coding subspace dimensionality.
        evr: baseline PCA explained variance ratio (full).
    """
    baseline_idx = sorted_indices[0]
    baseline_np = reps_by_task[baseline_idx].cpu().numpy().astype(np.float32)

    pca = PCA()
    pca.fit(baseline_np)
    evr = pca.explained_variance_ratio_
    k = _determine_k(evr, threshold)
    components_k = pca.components_[:k]  # (k, D)

    n = len(sorted_indices)
    coding_fractions = np.full(n, np.nan)
    coding_norms = np.zeros(n)
    null_norms = np.zeros(n)

    baseline_t = reps_by_task[baseline_idx].float()
    U_k = torch.from_numpy(components_k).float()  # (k, D)

    for ci, idx in enumerate(sorted_indices):
        if idx == baseline_idx:
            continue
        current = reps_by_task[idx].float()
        delta = current - baseline_t  # (N, D)
        delta_coding = delta @ U_k.T @ U_k  # (N, D)
        delta_null = delta - delta_coding

        cn = torch.norm(delta_coding, dim=1)
        nn = torch.norm(delta_null, dim=1)
        total_sq = cn ** 2 + nn ** 2
        frac = (cn ** 2 / (total_sq + 1e-12)).mean().item()

        coding_fractions[ci] = frac
        coding_norms[ci] = cn.mean().item()
        null_norms[ci] = nn.mean().item()

    return coding_fractions, coding_norms, null_norms, k, evr


def _compute_per_checkpoint_dimensionality(
    reps_by_task: Dict[int, torch.Tensor],
    sorted_indices: List[int],
    threshold: float,
) -> Tuple[List[int], List[float]]:
    """PCA on each checkpoint; return k (variance-threshold) and participation ratio."""
    ks: List[int] = []
    prs: List[float] = []
    for idx in sorted_indices:
        x = reps_by_task[idx].cpu().numpy().astype(np.float32)
        pca = PCA()
        pca.fit(x)
        evr = pca.explained_variance_ratio_
        ks.append(_determine_k(evr, threshold))
        lam = pca.explained_variance_
        pr = (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)
        prs.append(float(pr))
    return ks, prs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_coding_fraction(fracs, task_indices, layer, k, threshold, path):
    n = len(fracs)
    indices = np.arange(n)
    valid = ~np.isnan(fracs)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices[valid], fracs[valid], color="#1f77b4", alpha=0.8, label="Coding subspace")
    ax.bar(indices[valid], 1.0 - fracs[valid], bottom=fracs[valid],
           color="#ff7f0e", alpha=0.8, label="Null subspace")
    ax.set_title(f"Drift Subspace Decomposition — layer: {layer}\n"
                 f"(coding dim k={k}, variance threshold={threshold:.0%})")
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Fraction of Drift Variance")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"T{t}" for t in task_indices], rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_drift_norms(coding_norms, null_norms, task_indices, layer, k, path):
    n = len(coding_norms)
    indices = np.arange(n)
    valid = indices > 0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(indices[valid], coding_norms[valid], marker="o", label=f"Coding (k={k})")
    ax.plot(indices[valid], null_norms[valid], marker="s", label="Null")
    ax.set_title(f"Drift Magnitude by Subspace — layer: {layer}")
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Mean L2 Norm of Drift")
    ax.set_xticks(indices[valid])
    ax.set_xticklabels([f"T{task_indices[i]}" for i in indices[valid]],
                       rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_variance_explained(evr, k, layer, path):
    cumvar = np.cumsum(evr)
    # Show enough PCs to include the k marker (capped at 200)
    n_components = min(len(cumvar), max(100, k + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_components + 1), cumvar[:n_components],
            marker=".", markersize=4, label="Cumulative variance")
    if k <= len(cumvar):
        ax.axvline(k, color="red", linestyle="--", label=f"k={k} ({cumvar[k - 1]:.1%})")
        ax.axhline(cumvar[k - 1], color="red", linestyle=":", alpha=0.5)
    ax.set_title(f"Baseline PCA Cumulative Variance — layer: {layer}")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_dimensionality(ks, prs, task_indices, layer, threshold, path):
    n = len(ks)
    indices = np.arange(n)
    labels = [f"T{t}" for t in task_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(indices, ks, marker="o", color="#1f77b4")
    ax1.set_title(f"Coding Dimensionality (k) — layer: {layer}\n"
                  f"(variance threshold={threshold:.0%})")
    ax1.set_xlabel("Model after Task")
    ax1.set_ylabel("k (number of PCs)")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(indices, prs, marker="s", color="#ff7f0e")
    ax2.set_title(f"Participation Ratio — layer: {layer}")
    ax2.set_xlabel("Model after Task")
    ax2.set_ylabel("Participation Ratio")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_subspace_drift(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
    threshold: float = 0.99,
) -> None:
    """Run coding/null subspace drift decomposition for each layer.

    Args:
        reps_cache: ``{task_idx: {layer: Tensor(N, D)}}`` from ``build_reps_cache``.
        layer_names: Layers to analyze.
        output_dir: Output directory. Plots are written to ``<output_dir>/subspace_drift/``.
        threshold: Cumulative variance ratio defining the coding subspace.
    """
    out_subdir = os.path.join(output_dir, "subspace_drift")
    os.makedirs(out_subdir, exist_ok=True)

    sorted_indices = sorted(reps_cache.keys())
    if len(sorted_indices) < 2:
        print("  Skipping subspace drift: need at least 2 checkpoints")
        return

    summary: Dict[str, Dict] = {}

    for layer in layer_names:
        print(f"  Subspace drift for layer: {layer}")
        reps_by_task = {t: reps_cache[t][layer] for t in sorted_indices}
        safe = layer.replace(".", "_").replace("/", "_")

        coding_fracs, coding_norms, null_norms, k, evr = _compute_subspace_drift(
            reps_by_task, sorted_indices, threshold=threshold,
        )
        valid = coding_fracs[~np.isnan(coding_fracs)]
        mean_coding = float(valid.mean()) if len(valid) > 0 else float("nan")
        print(f"    k={k} ({np.cumsum(evr)[k - 1]:.1%} var), "
              f"mean coding fraction={mean_coding:.3f}")

        _plot_coding_fraction(
            coding_fracs, sorted_indices, layer, k, threshold,
            os.path.join(out_subdir, f"coding_fraction_{safe}.png"),
        )
        _plot_drift_norms(
            coding_norms, null_norms, sorted_indices, layer, k,
            os.path.join(out_subdir, f"drift_norms_{safe}.png"),
        )
        _plot_variance_explained(
            evr, k, layer,
            os.path.join(out_subdir, f"variance_explained_{safe}.png"),
        )

        ks, prs = _compute_per_checkpoint_dimensionality(
            reps_by_task, sorted_indices, threshold=threshold,
        )
        _plot_dimensionality(
            ks, prs, sorted_indices, layer, threshold,
            os.path.join(out_subdir, f"dimensionality_{safe}.png"),
        )

        summary[layer] = {
            "threshold": threshold,
            "baseline_k": k,
            "baseline_variance_at_k": float(np.cumsum(evr)[k - 1]),
            "mean_coding_fraction": mean_coding,
            "coding_fractions": coding_fracs.tolist(),
            "coding_norms": coding_norms.tolist(),
            "null_norms": null_norms.tolist(),
            "per_checkpoint_k": ks,
            "per_checkpoint_participation_ratio": prs,
            "task_indices": sorted_indices,
        }

    with open(os.path.join(out_subdir, "subspace_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Subspace drift results saved to {out_subdir}")
