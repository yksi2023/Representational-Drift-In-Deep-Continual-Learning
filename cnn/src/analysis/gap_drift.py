"""Pairwise representational drift as a function of task gap.

CNN analogue of the RNN ``vector_drift`` analysis, with time-dependent vectors
(PV / TCV) dropped (no time dimension) and renamed to CNN-appropriate terms:

- **Sample-PV** (sample population vector): per-sample flattened activation
  ``(D,)``. For each pair of checkpoints (i, j), compute Pearson correlation
  along the feature dimension per sample and average over samples. Group by
  gap = j - i.

- **ERV** (ensemble rate vector): per-neuron mean activation across probe
  samples ``(D,)`` for each checkpoint. Pearson correlation between ERVs of
  two checkpoints. Group by gap.

Both curves are plotted per layer on one figure, revealing drift saturation
as gap grows.
"""
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _pearson(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Element-wise Pearson correlation along ``dim``."""
    a_c = a - a.mean(dim=dim, keepdim=True)
    b_c = b - b.mean(dim=dim, keepdim=True)
    num = (a_c * b_c).sum(dim=dim)
    den = torch.sqrt((a_c ** 2).sum(dim=dim) * (b_c ** 2).sum(dim=dim)) + 1e-12
    return num / den


def _sample_pv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Mean over samples of per-sample Pearson correlation."""
    r = _pearson(rep_i.float(), rep_j.float(), dim=1)  # (N,)
    return r.mean().item()


def _erv_pearson(rep_i: torch.Tensor, rep_j: torch.Tensor) -> float:
    """Pearson correlation of ensemble rate vectors (per-neuron mean over samples)."""
    erv_i = rep_i.float().mean(dim=0)  # (D,)
    erv_j = rep_j.float().mean(dim=0)
    r = _pearson(erv_i, erv_j, dim=0)  # scalar tensor
    return r.item()


_VECTOR_FNS = {
    "Sample-PV": _sample_pv_pearson,
    "ERV": _erv_pearson,
}


def _compute_corr_vs_gap(
    reps_by_task: Dict[int, torch.Tensor],
    sorted_indices: List[int],
    corr_fn,
) -> Tuple[List[int], List[float], List[float]]:
    """Return (gaps, means, stds) grouped by task gap."""
    gap_to_vals: Dict[int, List[float]] = {}
    for a_i, i in enumerate(sorted_indices):
        for b_i in range(a_i + 1, len(sorted_indices)):
            j = sorted_indices[b_i]
            gap = j - i
            val = corr_fn(reps_by_task[i], reps_by_task[j])
            gap_to_vals.setdefault(gap, []).append(val)
    gaps = sorted(gap_to_vals.keys())
    means = [float(np.mean(gap_to_vals[g])) for g in gaps]
    stds = [float(np.std(gap_to_vals[g])) for g in gaps]
    return gaps, means, stds


def _plot_corr_vs_gap(
    results: Dict[str, Tuple[List[int], List[float], List[float]]],
    layer: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Sample-PV": "#1f77b4", "ERV": "#ff7f0e"}
    all_gaps: List[int] = []
    for vec_name in ["Sample-PV", "ERV"]:
        gaps, means, stds = results[vec_name]
        all_gaps.extend(gaps)
        ax.errorbar(gaps, means, yerr=stds, marker="o", capsize=4,
                    label=vec_name, color=colors[vec_name])

    ax.set_title(f"Representational Drift vs Task Gap — layer: {layer}")
    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    if all_gaps:
        ax.set_xticks(sorted(set(all_gaps)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_gap_drift(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
) -> None:
    """Compute Sample-PV and ERV Pearson correlation vs task gap, per layer."""
    out_subdir = os.path.join(output_dir, "gap_drift")
    os.makedirs(out_subdir, exist_ok=True)

    sorted_indices = sorted(reps_cache.keys())
    if len(sorted_indices) < 2:
        print("  Skipping gap drift: need at least 2 checkpoints")
        return

    summary: Dict[str, Dict] = {}
    for layer in layer_names:
        print(f"  Gap drift for layer: {layer}")
        reps_by_task = {t: reps_cache[t][layer] for t in sorted_indices}

        layer_res = {}
        for vec_name, fn in _VECTOR_FNS.items():
            gaps, means, stds = _compute_corr_vs_gap(reps_by_task, sorted_indices, fn)
            layer_res[vec_name] = (gaps, means, stds)
            if gaps:
                print(f"    {vec_name}: gap=1 r={means[0]:.4f}, "
                      f"gap={gaps[-1]} r={means[-1]:.4f}")

        safe = layer.replace(".", "_").replace("/", "_")
        _plot_corr_vs_gap(
            layer_res, layer, os.path.join(out_subdir, f"gap_drift_{safe}.png")
        )

        summary[layer] = {
            name: {"gaps": g, "means": m, "stds": s}
            for name, (g, m, s) in layer_res.items()
        }

    with open(os.path.join(out_subdir, "gap_drift_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Gap drift results saved to {out_subdir}")
