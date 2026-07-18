"""Aggregate multi-seed CNN experiments into cross-seed-averaged plots per method.

Reads **pre-computed** results from each seed's ``drift_analysis/`` directory
(produced by ``analyze_drift.py`` / ``analysis_cnn.sh``).  No GPU forward passes
are performed; the script is pure CPU and finishes in seconds.

Outputs (per method):
  1. accuracy_matrix.pdf              -- task × stage accuracy heatmap (mean)
  2. similarity_matrix_<layer>.pdf    -- pairwise cosine similarity heatmap (mean)
  3. reference_drift.pdf              -- cosine sim & L2 vs task index (mean ± std)
  4. gap_drift_sample_pv.pdf          -- Sample-PV Pearson corr vs task gap (mean ± std)

Usage:
    python aggregate_seeds.py \\
        --exp_root experiments \\
        --prefix "exp1_cnn_" \\
        --methods normal,replay,ewc,lwf,gpm \\
        --layers layer3 \\
        --output_dir experiments/aggregate_report
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis._plot_utils import (
    SINGLE_FIGSIZE,
    WIDE_FIGSIZE,
    SMALL_LEGEND_FONT_SIZE,
    SMALL_LEGEND_TITLE_SIZE,
    apply_paper_axis_style,
    savefig_compact,
    sparse_ticks,
    sparse_value_ticks,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def discover_seed_dirs(exp_root: str, prefix: str, method: str) -> List[str]:
    """Find all seed directories for a given method under exp_root.

    Expects naming: <prefix><method>_seed<N>/
    """
    pattern = os.path.join(exp_root, f"{prefix}{method}_seed*")
    dirs = sorted(glob.glob(pattern))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs


def load_accuracy_matrix(exp_dir: str) -> Optional[np.ndarray]:
    """Load performance_history.json and return task × stage accuracy matrix."""
    path = os.path.join(exp_dir, "performance_history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        perf = json.load(f)
    raw_task_names = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
    num_tasks = len(raw_task_names)
    num_stages = max(len(perf[n]) for n in raw_task_names)
    matrix = np.full((num_tasks, num_stages), np.nan)
    for i, name in enumerate(raw_task_names):
        for j, entry in enumerate(perf[name]):
            if entry is not None:
                acc = entry.get("accuracy")
                if acc is not None:
                    matrix[i, j] = acc
    return matrix


def load_similarity_matrix(exp_dir: str, layer: str) -> Optional[np.ndarray]:
    """Load pre-computed similarity matrix .npy from drift_analysis/."""
    safe_layer = layer.replace(".", "_").replace("/", "_")
    npy_path = os.path.join(
        exp_dir, "drift_analysis", "model_similarity_matrices",
        f"similarity_matrix_{safe_layer}.npy",
    )
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def load_gap_drift(exp_dir: str, layers: List[str]) -> Optional[Dict[str, Tuple[List[int], List[float]]]]:
    """Load pre-computed gap drift metrics from drift_analysis/.

    JSON structure: {layer: {"Sample-PV": {"gaps": [...], "means": [...], "stds": [...]}}}
    """
    json_path = os.path.join(exp_dir, "drift_analysis", "gap_drift", "gap_drift_metrics.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, Tuple[List[int], List[float]]] = {}
    for layer in layers:
        if layer in data:
            entry = data[layer]
            # Navigate into the "Sample-PV" sub-key
            spv = entry.get("Sample-PV", entry)
            gaps = [int(g) for g in spv["gaps"]]
            means = [float(m) for m in spv["means"]]
            result[layer] = (gaps, means)
    return result if result else None


def load_reference_drift(exp_dir: str) -> Optional[List[dict]]:
    """Load reference drift metrics.json from drift_analysis/."""
    path = os.path.join(exp_dir, "drift_analysis", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── plot 1: accuracy matrix ──────────────────────────────────────────────────

def plot_avg_accuracy_matrix(
    matrices: List[np.ndarray],
    method: str,
    output_dir: str,
):
    """Average accuracy matrices across seeds and plot heatmap."""
    stacked = np.stack(matrices, axis=0)  # (S, T, T)
    mean_matrix = np.nanmean(stacked, axis=0)

    n_tasks = mean_matrix.shape[0]

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    im = ax.imshow(mean_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(n_tasks)
    ax.set_xticks(sp); ax.set_xticklabels(sl)
    ax.set_yticks(sp); ax.set_yticklabels(sl)
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")
    ax.set_title(f"{method} (n={len(matrices)} seeds)", fontsize=16)
    apply_paper_axis_style(ax)
    path = os.path.join(output_dir, "accuracy_matrix.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] accuracy_matrix.pdf  ({len(matrices)} seeds)")


# ── plot 2: similarity matrix ────────────────────────────────────────────────

def plot_avg_similarity_matrix(
    sim_matrices: List[np.ndarray],
    method: str,
    layer: str,
    output_dir: str,
):
    """Average similarity matrices across seeds and plot heatmap."""
    stacked = np.stack(sim_matrices, axis=0)  # (S, T, T)
    mean_sim = np.nanmean(stacked, axis=0)

    n = mean_sim.shape[0]
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    im = ax.imshow(mean_sim, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    sp, sl = sparse_ticks(n)
    ax.set_xticks(sp); ax.set_xticklabels(sl)
    ax.set_yticks(sp); ax.set_yticklabels(sl)
    ax.set_xlabel("Model after Task")
    ax.set_ylabel("Model after Task")
    ax.set_title(f"{method} – {layer} (n={len(sim_matrices)} seeds)", fontsize=16)
    apply_paper_axis_style(ax)
    safe_layer = layer.replace(".", "_").replace("/", "_")
    path = os.path.join(output_dir, f"similarity_matrix_{safe_layer}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] similarity_matrix_{safe_layer}.pdf  ({len(sim_matrices)} seeds)")


# ── plot 3: gap drift ────────────────────────────────────────────────────────

def plot_avg_reference_drift(
    all_seed_metrics: List[List[dict]],
    method: str,
    output_dir: str,
):
    """Average reference drift curves across seeds and plot with error bands.

    Each element is a list of dicts with keys:
      layer, target_task, cosine_sim_mean, l2_dist_mean, shuffled_sim_mean
    """
    # Group by (layer, target_task)
    grouped: Dict[str, Dict[int, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for seed_metrics in all_seed_metrics:
        for entry in seed_metrics:
            layer = entry["layer"]
            tt = entry["target_task"]
            grouped[layer][tt]["cosine"].append(entry["cosine_sim_mean"])
            grouped[layer][tt]["l2"].append(entry["l2_dist_mean"])
            grouped[layer][tt]["shuffled"].append(entry["shuffled_sim_mean"])

    layers = sorted(grouped.keys())
    cmap = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    all_tasks: List[int] = []

    for idx, layer in enumerate(layers):
        task_data = grouped[layer]
        sorted_tasks = sorted(task_data.keys())
        all_tasks.extend(sorted_tasks)
        color = cmap(idx % 10)

        cos_mean = [np.mean(task_data[t]["cosine"]) for t in sorted_tasks]
        cos_std = [np.std(task_data[t]["cosine"]) for t in sorted_tasks]
        shuf_mean = [np.mean(task_data[t]["shuffled"]) for t in sorted_tasks]

        ax1.errorbar(sorted_tasks, cos_mean, yerr=cos_std, marker="o", capsize=4,
                     label=layer, color=color, linewidth=1.5, markersize=5)
        ax1.plot(sorted_tasks, shuf_mean, linestyle="--", color=color, alpha=0.4)

        l2_mean = [np.mean(task_data[t]["l2"]) for t in sorted_tasks]
        l2_std = [np.std(task_data[t]["l2"]) for t in sorted_tasks]
        ax2.errorbar(sorted_tasks, l2_mean, yerr=l2_std, marker="o", capsize=4,
                     label=layer, color=color, linewidth=1.5, markersize=5)

    ax1.set_xlabel("Task Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"{method} (n={len(all_seed_metrics)} seeds)", fontsize=14)
    apply_paper_axis_style(
        ax1, legend=True,
        legend_kwargs={"fontsize": SMALL_LEGEND_FONT_SIZE, "title_fontsize": SMALL_LEGEND_TITLE_SIZE},
    )
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_xlabel("Task Index")
    ax2.set_ylabel("L2 Distance")
    apply_paper_axis_style(
        ax2, legend=True,
        legend_kwargs={"fontsize": SMALL_LEGEND_FONT_SIZE, "title_fontsize": SMALL_LEGEND_TITLE_SIZE},
    )
    ax2.grid(True, linestyle="--", alpha=0.6)

    if all_tasks:
        ticks, labels = sparse_value_ticks(all_tasks)
        ax1.set_xticks(ticks); ax1.set_xticklabels(labels)
        ax2.set_xticks(ticks); ax2.set_xticklabels(labels)

    plt.tight_layout()
    path = os.path.join(output_dir, "reference_drift.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] reference_drift.pdf  ({len(all_seed_metrics)} seeds)")


# ── plot 4: gap drift ────────────────────────────────────────────────────────

def plot_avg_gap_drift(
    all_seed_results: List[Dict[str, Tuple[List[int], List[float]]]],
    method: str,
    output_dir: str,
):
    """Average gap-drift curves across seeds and plot with error bands.

    all_seed_results: list of {layer: (gaps, means)} per seed.
    """
    layer_names = list(all_seed_results[0].keys())
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    cmap = plt.get_cmap("tab10")
    all_gaps_union: List[int] = []

    for idx, layer in enumerate(layer_names):
        gap_to_values: Dict[int, List[float]] = defaultdict(list)
        for seed_result in all_seed_results:
            if layer not in seed_result:
                continue
            gaps, means = seed_result[layer]
            for g, m in zip(gaps, means):
                gap_to_values[g].append(m)

        gaps_sorted = sorted(gap_to_values.keys())
        all_gaps_union.extend(gaps_sorted)
        avg = [np.mean(gap_to_values[g]) for g in gaps_sorted]
        std = [np.std(gap_to_values[g]) for g in gaps_sorted]

        color = cmap(idx % 10)
        ax.errorbar(gaps_sorted, avg, yerr=std, marker="o", capsize=3,
                    label=layer, color=color, linewidth=1.5, markersize=4)

    ax.set_xlabel("Task Gap")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(f"{method} (n={len(all_seed_results)} seeds)", fontsize=16)
    apply_paper_axis_style(
        ax, legend=True,
        legend_kwargs={
            "loc": "lower left",
            "fontsize": SMALL_LEGEND_FONT_SIZE,
            "title_fontsize": SMALL_LEGEND_TITLE_SIZE,
        },
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    if all_gaps_union:
        ticks, labels = sparse_value_ticks(all_gaps_union)
        ax.set_xticks(ticks); ax.set_xticklabels(labels)

    path = os.path.join(output_dir, "gap_drift_sample_pv.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] gap_drift_sample_pv.pdf  ({len(all_seed_results)} seeds)")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed CNN results")
    parser.add_argument("--exp_root", type=str, required=True,
                        help="Root directory containing experiment folders")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Directory prefix, e.g. 'exp1_cnn_'")
    parser.add_argument("--methods", type=str, default="normal,replay,ewc,lwf,gpm",
                        help="Comma-separated method names")
    parser.add_argument("--layers", type=str, default="layer1,layer2,layer3,layer4",
                        help="Comma-separated layers for similarity matrix + gap drift")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to exp_root/aggregate_report)")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    layers = [l.strip() for l in args.layers.split(",")]

    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_root, "aggregate_report")

    print(f"Methods: {methods}")
    print(f"Layers for sim/gap: {layers}")
    print(f"Output: {args.output_dir}")
    print()

    for method in methods:
        seed_dirs = discover_seed_dirs(args.exp_root, args.prefix, method)
        if not seed_dirs:
            print(f"[{method}] No seed directories found, skipping.")
            continue

        print(f"[{method}] Found {len(seed_dirs)} seed(s): "
              f"{[os.path.basename(d) for d in seed_dirs]}")

        method_out = os.path.join(args.output_dir, method)
        os.makedirs(method_out, exist_ok=True)

        # ── Plot 1: accuracy matrix ──
        acc_matrices = []
        for sd in seed_dirs:
            m = load_accuracy_matrix(sd)
            if m is not None:
                acc_matrices.append(m)
        if acc_matrices:
            plot_avg_accuracy_matrix(acc_matrices, method, method_out)
        else:
            print(f"  [{method}] No performance_history.json found, skipping accuracy plot.")

        # ── Plot 2: similarity matrices (from pre-computed .npy) ──
        for ln in layers:
            sim_mats = []
            for sd in seed_dirs:
                s = load_similarity_matrix(sd, ln)
                if s is not None:
                    sim_mats.append(s)
            if sim_mats:
                plot_avg_similarity_matrix(sim_mats, method, ln, method_out)
            else:
                print(f"  [{method}] No similarity .npy for {ln}, skipping. "
                      f"Run analysis_cnn.sh first.")

        # ── Plot 3: reference drift (from pre-computed metrics.json) ──
        ref_drift_results = []
        for sd in seed_dirs:
            r = load_reference_drift(sd)
            if r is not None:
                ref_drift_results.append(r)
        if ref_drift_results:
            plot_avg_reference_drift(ref_drift_results, method, method_out)
        else:
            print(f"  [{method}] No reference drift metrics.json found, skipping. "
                  f"Run analysis_cnn.sh first.")

        # ── Plot 4: gap drift (from pre-computed JSON) ──
        gap_results: List[Dict[str, Tuple[List[int], List[float]]]] = []
        for sd in seed_dirs:
            g = load_gap_drift(sd, layers)
            if g is not None:
                gap_results.append(g)
        if gap_results:
            plot_avg_gap_drift(gap_results, method, method_out)
        else:
            print(f"  [{method}] No gap drift metrics found, skipping. "
                  f"Run analysis_cnn.sh first.")

        print()

    print("Aggregation complete.")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
