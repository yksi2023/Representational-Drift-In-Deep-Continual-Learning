"""Vector-level representational drift analysis.

Given a 3D tensor (Batch, Time, Neuron) at each checkpoint, defines three
vector types and tracks their Pearson correlation as a function of task gap:

1. **STPV (Spatiotemporal Population Vector)**: PVs concatenated across all
   time steps, shape (T*N,). Pearson correlation averaged over batch.
2. **Population Vector (PV)**: for a given (checkpoint, timestep), shape (N,).
   Pearson correlation averaged over batch and timesteps.
3. **Ensemble Rate Vector (ERV)**: for a given checkpoint, averaged across time,
   shape (N,). Pearson correlation averaged over batch.
4. **Tuning Curve Vector (TCV)**: for a given (checkpoint, neuron), shape (T,).
   Pearson correlation averaged over batch and neurons.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.analysis.baseline_drift import _load_reps_from_npz


def _reshape_to_3d(flat: np.ndarray, hidden_size: int) -> torch.Tensor:
    """Reshape (Batch, T*N) -> (Batch, T, N)."""
    B, D = flat.shape
    T = D // hidden_size
    assert D == T * hidden_size
    return torch.from_numpy(flat).float().reshape(B, T, hidden_size)


def _pearson_batch(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Element-wise Pearson correlation along `dim`, returned per-element.

    Args:
        a, b: same shape tensors
        dim: dimension along which to compute correlation

    Returns:
        Tensor with `dim` removed, containing per-element Pearson r.
    """
    a_c = a - a.mean(dim=dim, keepdim=True)
    b_c = b - b.mean(dim=dim, keepdim=True)
    num = (a_c * b_c).sum(dim=dim)
    den = torch.sqrt((a_c ** 2).sum(dim=dim) * (b_c ** 2).sum(dim=dim)) + 1e-12
    return num / den


def _stpv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Pearson correlation of Spatiotemporal Population Vectors, averaged over batch.

    STPV at (b) is rep[b, :, :].reshape(-1) in R^{T*N}.
    """
    B, T, N = rep_i.shape
    flat_i = rep_i.reshape(B, T * N)  # (B, T*N)
    flat_j = rep_j.reshape(B, T * N)
    r = _pearson_batch(flat_i, flat_j, dim=1)  # (B,)
    return r.mean().item()


def _pv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Pearson correlation of Population Vectors, averaged over batch & timesteps.

    PV at (b, t) is rep[b, t, :] in R^N.
    """
    # rep shape: (B, T, N), correlate along N (dim=2)
    r = _pearson_batch(rep_i, rep_j, dim=2)  # (B, T)
    return r.mean().item()


def _erv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Pearson correlation of Ensemble Rate Vectors, averaged over batch.

    ERV at (b) is rep[b, :, :].mean(dim=0) in R^N — time-averaged activity per neuron.
    """
    erv_i = rep_i.mean(dim=1)  # (B, N)
    erv_j = rep_j.mean(dim=1)  # (B, N)
    r = _pearson_batch(erv_i, erv_j, dim=1)  # (B,)
    return r.mean().item()


def _tcv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Pearson correlation of Tuning Curve Vectors, averaged over batch & neurons.

    TCV at (b, n) is rep[b, :, n] in R^T.
    """
    # rep shape: (B, T, N), correlate along T (dim=1)
    r = _pearson_batch(rep_i, rep_j, dim=1)  # (B, N)
    return r.mean().item()


_VECTOR_FNS = {
    "STPV": _stpv_pearson,
    "PV": _pv_pearson,
    "ERV": _erv_pearson,
    "TCV": _tcv_pearson,
}


def _compute_correlation_vs_gap(
    reps_3d_list: List[torch.Tensor],
    corr_fn,
) -> Tuple[List[int], List[float], List[float]]:
    """Compute mean Pearson correlation grouped by task gap.

    Returns:
        gaps, means, stds
    """
    n = len(reps_3d_list)
    gap_to_vals: Dict[int, List[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            gap = j - i
            val = corr_fn(reps_3d_list[i], reps_3d_list[j])
            gap_to_vals.setdefault(gap, []).append(val)

    gaps = sorted(gap_to_vals.keys())
    means = [np.mean(gap_to_vals[g]) for g in gaps]
    stds = [np.std(gap_to_vals[g]) for g in gaps]
    return gaps, means, stds


def _plot_correlation_vs_gap(
    results: Dict[str, Tuple[List[int], List[float], List[float]]],
    probe_task: str,
    output_path: str,
):
    """Plot Pearson correlation vs task gap for STPV, PV, ERV, TCV on one figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"STPV": "#d62728", "PV": "#1f77b4", "ERV": "#ff7f0e", "TCV": "#2ca02c"}
    for vec_name in ["STPV", "PV", "ERV", "TCV"]:
        gaps, means, stds = results[vec_name]
        ax.errorbar(gaps, means, yerr=stds, marker='o', capsize=4,
                    label=vec_name, color=colors[vec_name])

    ax.set_title(f"Representational Drift — probe: {probe_task}")
    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    if results["STPV"][0]:
        ax.set_xticks(results["STPV"][0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_vector_drift(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
    hidden_size: int = 256,
) -> None:
    """Compute STPV / PV / ERV / TCV Pearson correlation vs task gap.

    For each probe task, produces one combined plot showing drift curves
    for all four vector types.

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' representations to analyze.
        task_names: Ordered list of all task names in the sequence.
        output_dir: Directory to save results.
        hidden_size: Hidden size of the RNN.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    out_subdir = os.path.join(output_dir, "vector_drift")
    os.makedirs(out_subdir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  Vector drift for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())

        reps_3d_list = [_reshape_to_3d(raw_reps[idx], hidden_size) for idx in sorted_indices]

        results = {}
        for vec_name, corr_fn in _VECTOR_FNS.items():
            gaps, means, stds = _compute_correlation_vs_gap(reps_3d_list, corr_fn)
            results[vec_name] = (gaps, means, stds)
            if gaps:
                print(f"    {vec_name}: gap=1 r={means[0]:.4f}, gap={gaps[-1]} r={means[-1]:.4f}")

        out_path = os.path.join(out_subdir, f"vector_drift_{probe_task}.png")
        _plot_correlation_vs_gap(results, probe_task, out_path)
        print(f"    Plot saved to {out_path}")
