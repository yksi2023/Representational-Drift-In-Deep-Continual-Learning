"""Coding / Null subspace drift decomposition.

Defines the coding subspace via PCA on the baseline (first checkpoint) STPVs,
using a cumulative variance-explained threshold to determine dimensionality.
Then decomposes drift vectors from subsequent checkpoints into coding and
null components to quantify where representational change occurs.

Key outputs per probe task:
  1. Coding drift fraction vs checkpoint (bar/line plot)
  2. Coding subspace dimensionality (k) for each probe task
  3. Cumulative variance explained curve of the baseline PCA
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.analysis.baseline_drift import _load_reps_from_npz


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _determine_k(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    """Return smallest k such that cumulative variance >= threshold."""
    cumvar = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, len(explained_variance_ratio))


def _compute_subspace_drift(
    reps_dict: Dict[int, np.ndarray],
    sorted_indices: List[int],
    threshold: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Decompose drift into coding and null subspace components.

    Uses PCA of the first checkpoint to define the coding subspace.

    Args:
        reps_dict: task_idx -> STPV array of shape [N, D].
        sorted_indices: Ordered checkpoint indices.
        threshold: Cumulative variance ratio threshold for coding subspace.

    Returns:
        coding_fractions: (n_checkpoints,) fraction of drift in coding subspace.
            First element is NaN (no drift for baseline).
        coding_norms: (n_checkpoints,) L2 norm of coding drift component (mean over samples).
        null_norms: (n_checkpoints,) L2 norm of null drift component (mean over samples).
        k: Number of coding dimensions.
        explained_variance_ratio: Full explained variance ratio from baseline PCA.
    """
    baseline_idx = sorted_indices[0]
    baseline = reps_dict[baseline_idx]  # (N, D)

    # Fit PCA on baseline
    pca = PCA()
    pca.fit(baseline)
    evr = pca.explained_variance_ratio_
    k = _determine_k(evr, threshold)

    # Coding subspace projection matrix: P = U_k @ U_k^T  (D, D) — but we
    # only need to project vectors, so store U_k (D, k) instead.
    components_k = pca.components_[:k]  # (k, D)

    n_checkpoints = len(sorted_indices)
    coding_fractions = np.full(n_checkpoints, np.nan)
    coding_norms = np.zeros(n_checkpoints)
    null_norms = np.zeros(n_checkpoints)

    baseline_t = torch.from_numpy(baseline).float()

    for ci, idx in enumerate(sorted_indices):
        if idx == baseline_idx:
            continue
        current = torch.from_numpy(reps_dict[idx]).float()
        delta = current - baseline_t  # (N, D)

        # Project onto coding subspace
        U_k = torch.from_numpy(components_k).float()  # (k, D)
        delta_coding = delta @ U_k.T @ U_k  # (N, D)
        delta_null = delta - delta_coding  # (N, D)

        coding_norm = torch.norm(delta_coding, dim=1)  # (N,)
        null_norm = torch.norm(delta_null, dim=1)  # (N,)
        total_norm_sq = coding_norm ** 2 + null_norm ** 2

        # Fraction of drift variance in coding subspace (mean over samples)
        fraction = (coding_norm ** 2 / (total_norm_sq + 1e-12)).mean().item()
        coding_fractions[ci] = fraction
        coding_norms[ci] = coding_norm.mean().item()
        null_norms[ci] = null_norm.mean().item()

    return coding_fractions, coding_norms, null_norms, k, evr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_coding_fraction(
    coding_fractions: np.ndarray,
    task_names: List[str],
    probe_task: str,
    k: int,
    threshold: float,
    output_path: str,
):
    """Plot coding drift fraction vs checkpoint."""
    n = len(coding_fractions)
    indices = list(range(n))
    fracs = coding_fractions.copy()

    # Skip baseline (NaN)
    valid = ~np.isnan(fracs)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        np.array(indices)[valid],
        fracs[valid],
        color="#1f77b4",
        alpha=0.8,
        label="Coding subspace",
    )
    ax.bar(
        np.array(indices)[valid],
        1.0 - fracs[valid],
        bottom=fracs[valid],
        color="#ff7f0e",
        alpha=0.8,
        label="Null subspace",
    )

    ax.set_title(
        f"Drift Subspace Decomposition — probe: {probe_task}\n"
        f"(coding dim k={k}, variance threshold={threshold:.0%})"
    )
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Fraction of Drift Variance")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(indices)
    labels = task_names[:n] if len(task_names) >= n else [str(i) for i in indices]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_drift_norms(
    coding_norms: np.ndarray,
    null_norms: np.ndarray,
    task_names: List[str],
    probe_task: str,
    k: int,
    output_path: str,
):
    """Plot coding and null drift L2 norms vs checkpoint."""
    n = len(coding_norms)
    indices = np.arange(n)
    # Skip baseline (index 0)
    valid = indices > 0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(indices[valid], coding_norms[valid], marker="o", label=f"Coding (k={k})")
    ax.plot(indices[valid], null_norms[valid], marker="s", label="Null")
    ax.set_title(f"Drift Magnitude by Subspace — probe: {probe_task}")
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Mean L2 Norm of Drift")
    labels = task_names[:n] if len(task_names) >= n else [str(i) for i in range(n)]
    ax.set_xticks(indices[valid])
    ax.set_xticklabels([labels[i] for i in indices[valid]], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_variance_explained(
    evr: np.ndarray,
    k: int,
    probe_task: str,
    output_path: str,
):
    """Plot cumulative variance explained with the chosen k marked."""
    cumvar = np.cumsum(evr)
    n_components = min(len(cumvar), 100)  # show at most first 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_components + 1), cumvar[:n_components], marker=".", markersize=4)
    if k <= n_components:
        ax.axvline(k, color="red", linestyle="--", label=f"k={k} ({cumvar[k-1]:.1%})")
        ax.axhline(cumvar[k - 1], color="red", linestyle=":", alpha=0.5)
    ax.set_title(f"Baseline PCA Cumulative Variance — probe: {probe_task}")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-checkpoint dimensionality analysis
# ---------------------------------------------------------------------------

def _compute_per_checkpoint_dimensionality(
    reps_dict: Dict[int, np.ndarray],
    sorted_indices: List[int],
    threshold: float = 0.95,
) -> Tuple[List[int], List[float]]:
    """PCA on each checkpoint independently; return coding dim k and participation ratio.

    Participation Ratio = (sum lambda_i)^2 / sum(lambda_i^2), a continuous
    measure of effective dimensionality.

    Returns:
        ks: coding dimensionality per checkpoint.
        prs: participation ratio per checkpoint.
    """
    ks = []
    prs = []
    for idx in sorted_indices:
        pca = PCA()
        pca.fit(reps_dict[idx])
        evr = pca.explained_variance_ratio_
        k = _determine_k(evr, threshold)
        ks.append(k)
        # Participation ratio from eigenvalues
        lam = pca.explained_variance_
        pr = (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)
        prs.append(float(pr))
    return ks, prs


def _plot_dimensionality_over_checkpoints(
    ks: List[int],
    prs: List[float],
    task_names: List[str],
    probe_task: str,
    threshold: float,
    output_path: str,
):
    """Plot coding dimensionality k and participation ratio across checkpoints."""
    n = len(ks)
    indices = np.arange(n)
    labels = task_names[:n] if len(task_names) >= n else [str(i) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: coding dimensionality k
    ax1.plot(indices, ks, marker='o', color='#1f77b4')
    ax1.set_title(f"Coding Dimensionality (k) — probe: {probe_task}\n"
                  f"(variance threshold={threshold:.0%})")
    ax1.set_xlabel("Model after Task")
    ax1.set_ylabel("k (number of PCs)")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right: participation ratio
    ax2.plot(indices, prs, marker='s', color='#ff7f0e')
    ax2.set_title(f"Participation Ratio — probe: {probe_task}")
    ax2.set_xlabel("Model after Task")
    ax2.set_ylabel("Participation Ratio")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_subspace_drift(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
    threshold: float = 0.99,
) -> None:
    """Run coding/null subspace drift decomposition.

    For each probe task:
      1. PCA on baseline (first checkpoint) STPVs.
      2. Determine coding dimensionality k via cumulative variance threshold.
      3. Decompose drift from each subsequent checkpoint into coding and null
         subspace components.
      4. Per-checkpoint PCA: coding dimensionality k and participation ratio.
      5. Plot: drift fraction bar chart, drift norm curves, variance explained,
         dimensionality evolution.

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' STPVs to analyze.
        task_names: Ordered list of all task names in the sequence.
        output_dir: Directory to save results.
        threshold: Cumulative variance ratio threshold for coding subspace.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    out_subdir = os.path.join(output_dir, "subspace_drift")
    os.makedirs(out_subdir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  Subspace drift for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())

        if len(sorted_indices) < 2:
            print(f"    Skipping {probe_task}: need at least 2 checkpoints")
            continue

        reps_dict = {idx: raw_reps[idx] for idx in sorted_indices}

        coding_fracs, coding_norms, null_norms, k, evr = _compute_subspace_drift(
            reps_dict, sorted_indices, threshold=threshold,
        )

        print(f"    Coding subspace: k={k} dimensions "
              f"({np.cumsum(evr)[k-1]:.1%} variance explained)")
        # Report mean coding fraction (excluding baseline)
        valid_fracs = coding_fracs[~np.isnan(coding_fracs)]
        if len(valid_fracs) > 0:
            print(f"    Mean coding drift fraction: {valid_fracs.mean():.3f}")
            print(f"    Mean null drift fraction:   {1 - valid_fracs.mean():.3f}")

        _plot_coding_fraction(
            coding_fracs, task_names, probe_task, k, threshold,
            os.path.join(out_subdir, f"coding_fraction_{probe_task}.png"),
        )
        _plot_drift_norms(
            coding_norms, null_norms, task_names, probe_task, k,
            os.path.join(out_subdir, f"drift_norms_{probe_task}.png"),
        )
        _plot_variance_explained(
            evr, k, probe_task,
            os.path.join(out_subdir, f"variance_explained_{probe_task}.png"),
        )

        # Per-checkpoint dimensionality
        ks, prs = _compute_per_checkpoint_dimensionality(
            reps_dict, sorted_indices, threshold=threshold,
        )
        print(f"    Per-checkpoint coding dim k: {ks}")
        print(f"    Per-checkpoint participation ratio: "
              f"[{', '.join(f'{p:.1f}' for p in prs)}]")
        _plot_dimensionality_over_checkpoints(
            ks, prs, task_names, probe_task, threshold,
            os.path.join(out_subdir, f"dimensionality_{probe_task}.png"),
        )
        print(f"    Plots saved to {out_subdir}")
