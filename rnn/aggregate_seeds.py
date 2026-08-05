"""Aggregate multi-seed RNN experiments into cross-seed-averaged plots per method.

Directly reads each seed's representations/<probe>.npz to compute:
  1. accuracy_matrix.pdf          – from performance_history.json (mean across seeds)
  2. pearson_matrix_<probe>.pdf   – pairwise Pearson similarity (mean across seeds)
  3. vector_drift_<probe>.pdf     – STPV/PV/ERV/TCV Pearson vs task gap (mean ± std)
  4. temporal_similarity/cross_checkpoint_cosine_<probe>.pdf – temporal PV cosine matrix (mean across seeds)

No GPU needed; no prior analysis_rnn.sh required. Finishes in seconds.

Usage:
    python aggregate_seeds.py \\
        --exp_root experiments \\
        --prefix "exp1_rnn_" \\
        --methods normal,replay \\
        --probe fdgo \\
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
import torch

from src.analysis._plot_utils import (
    SINGLE_FIGSIZE,
    WIDE_FIGSIZE,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    apply_paper_axis_style,
    hide_axis,
    savefig_compact,
    sparse_value_ticks,
)
from src.drift_metrics import compute_pairwise_pearson_matrix
from src.analysis.reference_drift import _load_reps_from_npz
from src.analysis.temporal_similarity import _build_full_matrix, _plot_full_matrix


# ── helpers ──────────────────────────────────────────────────────────────────

def discover_seed_dirs(exp_root: str, prefix: str, method: str) -> List[str]:
    pattern = os.path.join(exp_root, f"{prefix}{method}_seed*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    return dirs


def load_accuracy_matrix(exp_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(exp_dir, "performance_history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        perf = json.load(f)
    task_names = list(perf.keys())
    num_tasks = len(task_names)
    num_stages = max(len(perf[n]) for n in task_names)
    matrix = np.full((num_tasks, num_stages), np.nan)
    for i, name in enumerate(task_names):
        for j, entry in enumerate(perf[name]):
            if entry is not None:
                acc = entry.get("accuracy")
                if acc is not None:
                    matrix[i, j] = acc
    return matrix


def load_task_names(exp_dir: str) -> List[str]:
    config_path = os.path.join(exp_dir, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "tasks" in cfg:
            return cfg["tasks"]
    perf_path = os.path.join(exp_dir, "performance_history.json")
    if os.path.exists(perf_path):
        with open(perf_path, "r", encoding="utf-8") as f:
            return list(json.load(f).keys())
    return []


TASK_DISPLAY_NAMES = {
    "fdgo": "Go",
    "reactgo": "RT Go",
    "delaygo": "Dly Go",
    "fdanti": "Anti",
    "reactanti": "RT Anti",
    "delayanti": "Dly Anti",
    "dm1": "DM 1",
    "dm2": "DM 2",
    "contextdm1": "Ctx DM 1",
    "contextdm2": "Ctx DM 2",
    "multidm": "MultSen DM",
    "delaydm1": "Dly DM 1",
    "delaydm2": "Dly DM 2",
    "contextdelaydm1": "Ctx Dly DM 1",
    "contextdelaydm2": "Ctx Dly DM 2",
    "multidelaydm": "MultSen Dly DM",
    "dmsgo": "DMS",
    "dmsnogo": "DNMS",
    "dmcgo": "DMC",
    "dmcnogo": "DNMC",
}


def task_display_labels(task_names: List[str], n: int) -> List[str]:
    """Map raw task keys to short display names, falling back to 1-indexed numbers."""
    if len(task_names) != n:
        return [str(i + 1) for i in range(n)]
    return [TASK_DISPLAY_NAMES.get(t, t) for t in task_names]


def _reshape_to_3d(flat: np.ndarray, hidden_size: int) -> torch.Tensor:
    B, D = flat.shape
    T = D // hidden_size
    assert D == T * hidden_size
    return torch.from_numpy(flat).float().reshape(B, T, hidden_size)


def _pearson_batch(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    a_c = a - a.mean(dim=dim, keepdim=True)
    b_c = b - b.mean(dim=dim, keepdim=True)
    num = (a_c * b_c).sum(dim=dim)
    den = torch.sqrt((a_c ** 2).sum(dim=dim) * (b_c ** 2).sum(dim=dim)) + 1e-12
    return num / den


def _stpv_pearson(ri, rj):
    B, T, N = ri.shape
    return _pearson_batch(ri.reshape(B, T * N), rj.reshape(B, T * N), dim=1).mean().item()

def _pv_pearson(ri, rj):
    return _pearson_batch(ri, rj, dim=2).mean().item()

def _erv_pearson(ri, rj):
    return _pearson_batch(ri.mean(dim=1), rj.mean(dim=1), dim=1).mean().item()

def _tcv_pearson(ri, rj):
    return _pearson_batch(ri, rj, dim=1).mean().item()

_VECTOR_FNS = {"STPV": _stpv_pearson, "PV": _pv_pearson, "ERV": _erv_pearson, "TCV": _tcv_pearson}


def _correlation_vs_gap(reps_3d: List[torch.Tensor], corr_fn) -> Tuple[List[int], List[float]]:
    n = len(reps_3d)
    gap_to_vals: Dict[int, List[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            gap_to_vals.setdefault(j - i, []).append(corr_fn(reps_3d[i], reps_3d[j]))
    gaps = sorted(gap_to_vals.keys())
    means = [np.mean(gap_to_vals[g]) for g in gaps]
    return gaps, means


# ── plot 1: accuracy matrix ──────────────────────────────────────────────────

def plot_avg_accuracy_matrix(
    matrices: List[np.ndarray], task_names: List[str], method: str, output_dir: str,
):
    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(stacked, axis=0)
    n_tasks = mean_matrix.shape[0]
    labels = task_display_labels(task_names, n_tasks)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    positions = list(range(n_tasks))
    ax.set_yticks(positions); ax.set_yticklabels(labels)
    ax.set_ylabel("Evaluated Task")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("After Training on Task")
    apply_paper_axis_style(ax)
    ax.tick_params(axis="both", labelsize=11)
    if method != "replay":
        hide_axis(ax, "x")
    path = os.path.join(output_dir, "accuracy_matrix.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] accuracy_matrix.pdf  ({len(matrices)} seeds)")


# ── plot 2: pearson similarity matrix ────────────────────────────────────────

def plot_avg_pearson_matrix(
    seed_dirs: List[str], probe_task: str, task_names: List[str],
    method: str, output_dir: str,
):
    matrices = []
    for sd in seed_dirs:
        reps_dir = os.path.join(sd, "representations")
        if not os.path.isdir(reps_dir):
            continue
        try:
            raw = _load_reps_from_npz(reps_dir, probe_task)
        except FileNotFoundError:
            continue
        sorted_idx = sorted(raw.keys())
        reps_list = [torch.from_numpy(raw[k]).float() for k in sorted_idx]
        mat = compute_pairwise_pearson_matrix(reps_list).numpy()
        matrices.append(mat)

    if not matrices:
        print(f"  [{method}] No representations/{probe_task}.npz found, skipping pearson matrix.")
        return

    mean_mat = np.mean(np.stack(matrices), axis=0)
    n = mean_mat.shape[0]

    labels = task_display_labels(task_names, n)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.imshow(mean_mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    positions = list(range(n))
    ax.set_yticks(positions); ax.set_yticklabels(labels)
    ax.set_ylabel("Model after Task")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Model after Task")
    apply_paper_axis_style(ax)
    ax.tick_params(axis="both", labelsize=11)
    if method != "replay":
        hide_axis(ax, "x")
    path = os.path.join(output_dir, f"pearson_matrix_{probe_task}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] pearson_matrix_{probe_task}.pdf  ({len(matrices)} seeds)")


# ── plot 3: vector drift ────────────────────────────────────────────────────

def plot_avg_vector_drift(
    seed_dirs: List[str], probe_task: str,
    method: str, output_dir: str, hidden_size: int = 256,
):
    # Collect per-seed per-vector-type gap→mean curves
    all_seed_results: List[Dict[str, Tuple[List[int], List[float]]]] = []
    for sd in seed_dirs:
        reps_dir = os.path.join(sd, "representations")
        if not os.path.isdir(reps_dir):
            continue
        try:
            raw = _load_reps_from_npz(reps_dir, probe_task)
        except FileNotFoundError:
            continue
        sorted_idx = sorted(raw.keys())
        reps_3d = [_reshape_to_3d(raw[k], hidden_size) for k in sorted_idx]
        seed_result = {}
        for vec_name, corr_fn in _VECTOR_FNS.items():
            gaps, means = _correlation_vs_gap(reps_3d, corr_fn)
            seed_result[vec_name] = (gaps, means)
        all_seed_results.append(seed_result)

    if not all_seed_results:
        print(f"  [{method}] No representations/{probe_task}.npz found, skipping vector drift.")
        return

    colors = {"STPV": "#d62728", "PV": "#1f77b4", "ERV": "#ff7f0e", "TCV": "#2ca02c"}
    vector_drift_figsize = (WIDE_FIGSIZE[0] / WIDE_FIGSIZE[1] * SINGLE_FIGSIZE[1], SINGLE_FIGSIZE[1])
    fig, ax = plt.subplots(figsize=vector_drift_figsize)
    all_gaps: List[int] = []

    for vec_name in ["STPV", "PV", "ERV", "TCV"]:
        gap_to_vals: Dict[int, List[float]] = defaultdict(list)
        for sr in all_seed_results:
            gaps, means = sr[vec_name]
            for g, m in zip(gaps, means):
                gap_to_vals[g].append(m)
        gaps_sorted = sorted(gap_to_vals.keys())
        all_gaps.extend(gaps_sorted)
        avg = [np.mean(gap_to_vals[g]) for g in gaps_sorted]
        std = [np.std(gap_to_vals[g]) for g in gaps_sorted]
        ax.errorbar(gaps_sorted, avg, yerr=std, marker="o", capsize=4,
                    label=vec_name, color=colors[vec_name])

    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Task Gap")
    apply_paper_axis_style(
        ax, legend=True,
        legend_kwargs={
            "loc": "upper right",
            "fontsize": LEGEND_FONT_SIZE,
            "title_fontsize": LEGEND_TITLE_SIZE,
        },
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    if all_gaps:
        ticks, labels = sparse_value_ticks(all_gaps)
        ax.set_xticks(ticks); ax.set_xticklabels(labels)

    if method != "replay":
        hide_axis(ax, "x")
    if method != "normal":
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    path = os.path.join(output_dir, f"vector_drift_{probe_task}.pdf")
    savefig_compact(fig, path)
    plt.close()
    print(f"  [{method}] vector_drift_{probe_task}.pdf  ({len(all_seed_results)} seeds)")


# ── plot 4: temporal similarity ─────────────────────────────────────────────

def plot_avg_temporal_similarity(
    seed_dirs: List[str], probe_task: str, task_names: List[str],
    method: str, output_dir: str, hidden_size: int = 256,
):
    """Average cross-checkpoint temporal similarity matrices across seeds."""
    matrices = []
    reps_3d_ref = None
    for sd in seed_dirs:
        reps_dir = os.path.join(sd, "representations")
        if not os.path.isdir(reps_dir):
            continue
        try:
            raw = _load_reps_from_npz(reps_dir, probe_task)
        except FileNotFoundError:
            continue
        sorted_idx = sorted(raw.keys())
        reps_3d = [_reshape_to_3d(raw[k], hidden_size) for k in sorted_idx]
        mat = _build_full_matrix(reps_3d, metric="cosine")
        matrices.append(mat)
        if reps_3d_ref is None:
            reps_3d_ref = reps_3d

    if not matrices:
        print(f"  [{method}] No representations/{probe_task}.npz found, skipping temporal similarity.")
        return

    mean_mat = np.mean(np.stack(matrices), axis=0)
    n_checkpoints = len(reps_3d_ref)
    seq_len = reps_3d_ref[0].shape[1]

    out_subdir = os.path.join(output_dir, "temporal_similarity")
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"cross_checkpoint_cosine_{probe_task}.pdf")
    _plot_full_matrix(
        mean_mat, seq_len, n_checkpoints, task_names,
        probe_task, out_path, metric_label="Cosine Similarity (mean)",
    )
    plt.close()
    print(f"  [{method}] temporal_similarity/cross_checkpoint_cosine_{probe_task}.pdf  ({len(matrices)} seeds)")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed RNN results")
    parser.add_argument("--exp_root", type=str, required=True,
                        help="Root directory containing experiment folders")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Directory prefix, e.g. 'exp1_rnn_'")
    parser.add_argument("--methods", type=str, default="normal,replay",
                        help="Comma-separated method names")
    parser.add_argument("--probe", type=str, default="fdgo",
                        help="Probe task name for similarity/drift analysis")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="RNN hidden size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to exp_root/aggregate_report)")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    probe_task = args.probe

    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_root, "aggregate_report")

    print(f"Methods: {methods}")
    print(f"Probe task: {probe_task}")
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

        # Resolve task names from first available seed
        task_names = load_task_names(seed_dirs[0])

        # ── 1. accuracy matrix ──
        acc_matrices = [load_accuracy_matrix(sd) for sd in seed_dirs]
        acc_matrices = [m for m in acc_matrices if m is not None]
        if acc_matrices:
            plot_avg_accuracy_matrix(acc_matrices, task_names, method, method_out)
        else:
            print(f"  [{method}] No performance_history.json found.")

        # ── 2. pearson similarity matrix ──
        plot_avg_pearson_matrix(seed_dirs, probe_task, task_names, method, method_out)

        # ── 3. vector drift ──
        plot_avg_vector_drift(seed_dirs, probe_task, method, method_out, args.hidden_size)

        # ── 4. temporal similarity ──
        plot_avg_temporal_similarity(seed_dirs, probe_task, task_names, method, method_out, args.hidden_size)

        print()

    print("Aggregation complete.")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
