"""Pairwise representational drift as a function of task gap.

CNN analogue of the RNN ``vector_drift`` analysis, with time-dependent vectors
(PV / TCV) dropped (no time dimension) and renamed to CNN-appropriate terms:

- **Sample-PV** (sample population vector): per-sample flattened activation
  ``(D,)``. For each pair of checkpoints (i, j), compute Pearson correlation
  along the feature dimension per sample and average over samples. Group by
  gap = j - i.

All layers are plotted on a single figure, revealing per-layer drift
saturation as gap grows.
"""
import json
import os
import tempfile
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis._plot_utils import (
    SMALL_LEGEND_FONT_SIZE,
    SMALL_LEGEND_TITLE_SIZE,
    WIDE_FIGSIZE,
    apply_paper_axis_style,
    layer_color_map,
    layer_errorbar_kwargs,
    layer_marker_map,
    savefig_compact,
    sparse_value_ticks,
)


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


def _plot_all_layers(
    all_results: Dict[str, Tuple[List[int], List[float], List[float]]],
    output_path: str,
):
    """Plot Sample-PV correlation vs gap for all layers on one figure."""
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    all_gaps: List[int] = []
    colors = layer_color_map(list(all_results))
    markers = layer_marker_map(list(all_results))

    for layer, (gaps, means, stds) in all_results.items():
        all_gaps.extend(gaps)
        ax.errorbar(
            gaps, means, yerr=stds, label=layer,
            **layer_errorbar_kwargs(colors[layer], markers[layer]),
        )

    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-0.1, 1.05)
    apply_paper_axis_style(
        ax,
        legend=True,
        legend_kwargs={
            "loc": "lower left",
            "fontsize": SMALL_LEGEND_FONT_SIZE,
            "title_fontsize": SMALL_LEGEND_TITLE_SIZE,
        },
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    if all_gaps:
        ticks, labels = sparse_value_ticks(all_gaps)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    savefig_compact(fig, output_path)
    plt.close()


def run_gap_drift(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    layer_names: List[str],
    output_dir: str,
) -> None:
    """Compute Sample-PV Pearson correlation vs task gap, all layers on one plot."""
    out_subdir = os.path.join(output_dir, "gap_drift")
    os.makedirs(out_subdir, exist_ok=True)

    sorted_indices = sorted(reps_cache.keys())
    if len(sorted_indices) < 2:
        print("  Skipping gap drift: need at least 2 checkpoints")
        return

    summary: Dict[str, Dict] = {}
    all_layer_results: Dict[str, Tuple[List[int], List[float], List[float]]] = {}

    for layer in layer_names:
        print(f"  Gap drift for layer: {layer}")
        reps_by_task = {t: reps_cache[t][layer] for t in sorted_indices}

        gaps, means, stds = _compute_corr_vs_gap(reps_by_task, sorted_indices, _sample_pv_pearson)
        all_layer_results[layer] = (gaps, means, stds)
        if gaps:
            print(f"    Sample-PV: gap=1 r={means[0]:.4f}, "
                  f"gap={gaps[-1]} r={means[-1]:.4f}")

        summary[layer] = {"Sample-PV": {"gaps": gaps, "means": means, "stds": stds}}

    _plot_all_layers(all_layer_results, os.path.join(out_subdir, "gap_drift_sample_pv.pdf"))

    metrics_path = os.path.join(out_subdir, "gap_drift_metrics.json")
    fd, tmp_path = tempfile.mkstemp(
        dir=out_subdir, prefix=".gap_drift_metrics.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, metrics_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    print(f"  Gap drift results saved to {out_subdir}")
