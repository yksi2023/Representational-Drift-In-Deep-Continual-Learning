"""Performance plotting for CNN continual learning runs.

Reads ``performance_history.json`` (RNN-compatible format produced by
``src/methods/base.py::_evaluate_and_record_all``) and generates:

  1. Task x training-stage accuracy matrix heatmap.
  2. First-task accuracy / loss retention over task sequence.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_cnn_performance(exp_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Generate the full accuracy matrix and first-task retention plots.

    Args:
        exp_dir: Experiment directory containing ``performance_history.json``.
        output_dir: Directory to save plots (defaults to ``exp_dir``).
    """
    if output_dir is None:
        output_dir = exp_dir
    os.makedirs(output_dir, exist_ok=True)

    perf_path = os.path.join(exp_dir, "performance_history.json")
    if not os.path.exists(perf_path):
        raise FileNotFoundError(f"Performance history not found: {perf_path}")

    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)

    # Deterministic task ordering: task_0, task_1, ...
    task_names = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
    num_stages = max(len(perf[n]) for n in task_names)

    acc_matrix = np.full((len(task_names), num_stages), np.nan)
    for i, name in enumerate(task_names):
        for j, entry in enumerate(perf[name]):
            if entry is None:
                continue
            acc = entry.get("accuracy")
            if acc is not None:
                acc_matrix[i, j] = acc

    # --- 1. Accuracy heatmap ---
    _plot_matrix_heatmap(
        acc_matrix, task_names, task_names,
        title="Accuracy Matrix (eval_task x trained_after)",
        cbar_label="Accuracy",
        output_path=os.path.join(output_dir, "accuracy_matrix.png"),
        vmin=0, vmax=1, cmap="viridis",
    )

    # --- 2. First-task retention plot ---
    _plot_first_task_retention(perf, task_names, output_dir)

    print(f"  Performance plots saved to {output_dir}")


def _plot_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cbar_label: str,
    output_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=7)

    nrows, ncols = matrix.shape
    if nrows <= 20 and ncols <= 20:
        for i in range(nrows):
            for j in range(ncols):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f"{val:.2f}" if val < 10 else f"{val:.1f}"
                    ax.text(j, i, text, ha="center", va="center",
                            color="black", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()


def _plot_first_task_retention(
    perf: Dict,
    task_names: List[str],
    output_dir: str,
) -> None:
    first_task = task_names[0]
    entries = perf[first_task]

    accs, losses = [], []
    for entry in entries:
        if entry is None:
            accs.append(np.nan); losses.append(np.nan)
        else:
            a = entry.get("accuracy")
            accs.append(a if a is not None else np.nan)
            losses.append(entry.get("loss", np.nan))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(accs))

    ax1.plot(list(x), accs, marker="o", markersize=5)
    ax1.set_title(f"First Task ({first_task}) - Accuracy Retention")
    ax1.set_xlabel("After Training on Task")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(task_names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(list(x), losses, marker="o", markersize=5, color="red")
    ax2.set_title(f"First Task ({first_task}) - Loss Over Time")
    ax2.set_xlabel("After Training on Task")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(task_names, rotation=45, ha="right", fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "first_task_retention.png")
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()
