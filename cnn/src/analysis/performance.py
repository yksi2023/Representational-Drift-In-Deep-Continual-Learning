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

from src.analysis._plot_utils import apply_paper_axis_style, sparse_ticks


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
    raw_task_names = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
    task_names = [str(i + 1) for i in range(len(raw_task_names))]
    num_stages = max(len(perf[n]) for n in raw_task_names)

    acc_matrix = np.full((len(task_names), num_stages), np.nan)
    for i, name in enumerate(raw_task_names):
        for j, entry in enumerate(perf[name]):
            if entry is None:
                continue
            acc = entry.get("accuracy")
            if acc is not None:
                acc_matrix[i, j] = acc

    # --- 1. Accuracy heatmap ---
    _plot_matrix_heatmap(
        acc_matrix, task_names, task_names,
        output_path=os.path.join(output_dir, "accuracy_matrix.pdf"),
        vmin=0, vmax=1, cmap="viridis",
    )

    # --- 2. First-task retention plot ---
    _plot_first_task_retention(perf, task_names, raw_task_names, output_dir)

    print(f"  Performance plots saved to {output_dir}")


def _plot_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    output_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    sp, sl = sparse_ticks(len(col_labels))
    ax.set_xticks(sp)
    ax.set_xticklabels(sl)
    sp2, sl2 = sparse_ticks(len(row_labels))
    ax.set_yticks(sp2)
    ax.set_yticklabels(sl2)

    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")
    apply_paper_axis_style(ax)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()


def _plot_first_task_retention(
    perf: Dict,
    task_names: List[str],
    raw_task_names: List[str],
    output_dir: str,
) -> None:
    first_task = raw_task_names[0]
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
    ax1.set_xlabel("After Training on Task")
    ax1.set_ylabel("Accuracy")
    sp, sl = sparse_ticks(len(accs))
    ax1.set_xticks(sp)
    ax1.set_xticklabels(sl)
    ax1.set_ylim(-0.05, 1.05)
    apply_paper_axis_style(ax1)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(list(x), losses, marker="o", markersize=5, color="red")
    ax2.set_xlabel("After Training on Task")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(sp)
    ax2.set_xticklabels(sl)
    apply_paper_axis_style(ax2)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "first_task_retention.pdf")
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()
