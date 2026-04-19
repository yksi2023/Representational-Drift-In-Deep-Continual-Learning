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


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _get_svd_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_pca(
    X: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PCA via ``torch.linalg.svd`` (GPU when available).

    Centers X then runs a thin SVD. For [N, D] with D >> N (typical for conv
    features), thin SVD complexity is O(N^2 D) which is drastically faster
    than sklearn's default full SVD, and runs on GPU.

    Returns:
        evr: (k,) explained variance ratio (descending), on CPU float32.
        eigvals: (k,) variance (= S^2 / (N-1)), on CPU float32.
        components: (k, D) principal directions, on CPU float32.
        where k = min(N, D).
    """
    X = X.to(device=device, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    # full_matrices=False => thin SVD: U (N, k), S (k,), Vh (k, D), k=min(N,D)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    n = X.shape[0]
    eigvals = (S * S) / max(n - 1, 1)
    total = eigvals.sum().clamp_min(1e-12)
    evr = eigvals / total
    return evr.cpu(), eigvals.cpu(), Vh.cpu()


def _determine_k(evr: torch.Tensor, threshold: float) -> int:
    cumvar = torch.cumsum(evr, dim=0)
    # searchsorted-style: first index where cumvar >= threshold.
    hits = (cumvar >= threshold).nonzero(as_tuple=False)
    if len(hits) == 0:
        return int(evr.numel())
    return int(hits[0].item()) + 1


def _compute_all_pca(
    reps_by_task: Dict[int, torch.Tensor],
    sorted_indices: List[int],
    threshold: float,
    device: torch.device,
) -> Tuple[
    Dict[int, torch.Tensor],  # evr per ckpt
    Dict[int, torch.Tensor],  # eigvals per ckpt
    Dict[int, torch.Tensor],  # components per ckpt
    List[int],                # k per ckpt
    List[float],              # participation ratio per ckpt
]:
    """Single pass: PCA every checkpoint once. Feeds both the drift
    decomposition (baseline components) and the per-ckpt dimensionality plots,
    avoiding the previous two-loops-both-doing-SVD duplication.
    """
    evr_by: Dict[int, torch.Tensor] = {}
    eig_by: Dict[int, torch.Tensor] = {}
    comp_by: Dict[int, torch.Tensor] = {}
    ks: List[int] = []
    prs: List[float] = []
    for idx in sorted_indices:
        evr, eigvals, components = _torch_pca(reps_by_task[idx], device)
        evr_by[idx] = evr
        eig_by[idx] = eigvals
        comp_by[idx] = components
        ks.append(_determine_k(evr, threshold))
        lam = eigvals
        pr = float((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12))
        prs.append(pr)
    return evr_by, eig_by, comp_by, ks, prs


def _compute_subspace_drift(
    reps_by_task: Dict[int, torch.Tensor],
    sorted_indices: List[int],
    baseline_components: torch.Tensor,
    k: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project drift onto baseline coding / null subspaces. GPU-accelerated.

    Returns:
        coding_fractions: (T,) mean per-sample coding-variance fraction (NaN at baseline).
        coding_norms: (T,) mean L2 norm of coding drift component.
        null_norms: (T,) mean L2 norm of null drift component.
    """
    baseline_idx = sorted_indices[0]
    baseline_t = reps_by_task[baseline_idx].to(device=device, dtype=torch.float32)
    U_k = baseline_components[:k].to(device=device, dtype=torch.float32)  # (k, D)

    n = len(sorted_indices)
    coding_fractions = np.full(n, np.nan)
    coding_norms = np.zeros(n)
    null_norms = np.zeros(n)

    for ci, idx in enumerate(sorted_indices):
        if idx == baseline_idx:
            continue
        current = reps_by_task[idx].to(device=device, dtype=torch.float32)
        delta = current - baseline_t                         # (N, D)
        coeffs = delta @ U_k.T                               # (N, k)
        delta_coding = coeffs @ U_k                          # (N, D)
        delta_null = delta - delta_coding

        cn = torch.linalg.vector_norm(delta_coding, dim=1)
        nn = torch.linalg.vector_norm(delta_null, dim=1)
        total_sq = cn * cn + nn * nn
        frac = ((cn * cn) / (total_sq + 1e-12)).mean().item()

        coding_fractions[ci] = frac
        coding_norms[ci] = cn.mean().item()
        null_norms[ci] = nn.mean().item()

    return coding_fractions, coding_norms, null_norms


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
    device = _get_svd_device()
    baseline_idx = sorted_indices[0]

    for layer in layer_names:
        print(f"  Subspace drift for layer: {layer}")
        reps_by_task = {t: reps_cache[t][layer] for t in sorted_indices}
        safe = layer.replace(".", "_").replace("/", "_")

        # Single PCA pass per checkpoint (GPU SVD) -> feeds both the drift
        # decomposition (baseline components) and the per-ckpt dimensionality
        # plots. Used to be two independent CPU full-SVD loops.
        evr_by, _eig_by, comp_by, ks, prs = _compute_all_pca(
            reps_by_task, sorted_indices, threshold=threshold, device=device,
        )
        evr = evr_by[baseline_idx].numpy()
        k = ks[sorted_indices.index(baseline_idx)]
        baseline_components = comp_by[baseline_idx]

        coding_fracs, coding_norms, null_norms = _compute_subspace_drift(
            reps_by_task, sorted_indices, baseline_components, k, device=device,
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
