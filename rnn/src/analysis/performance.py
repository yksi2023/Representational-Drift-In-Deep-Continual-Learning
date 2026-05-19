"""Performance plotting module for RNN.

Reads performance_history.json and generates performance plots showing
loss, accuracy, and fixation accuracy across the task sequence.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.analysis._plot_utils import t_labels


def plot_rnn_performance(exp_dir: str, output_dir: str) -> None:
    """
    Plot performance figures from saved performance_history.json.

    Generates:
    1. Accuracy matrix heatmap
    2. First-task retention plot (accuracy + loss)

    Args:
        exp_dir: Experiment directory containing performance_history.json.
        output_dir: Directory to save plots.
    """
    perf_path = os.path.join(exp_dir, "performance_history.json")
    if not os.path.exists(perf_path):
        raise FileNotFoundError(f"Performance history not found: {perf_path}")

    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    task_names = list(perf.keys())
    num_tasks = len(perf[task_names[0]])  # number of training stages

    # Build accuracy matrix: rows = eval task, cols = after training on task_i
    acc_matrix = np.full((len(task_names), num_tasks), np.nan)

    for i, task_name in enumerate(task_names):
        for j, entry in enumerate(perf[task_name]):
            if entry is None:
                continue
            acc_val = entry.get('accuracy')
            acc_matrix[i, j] = acc_val if acc_val is not None else np.nan

    # --- 1. Accuracy heatmap ---
    tick_labels = t_labels(task_names)
    _plot_matrix_heatmap(
        acc_matrix, tick_labels, tick_labels,
        output_path=os.path.join(output_dir, "accuracy_matrix.pdf"),
        vmin=0, vmax=1, cmap='viridis',
    )

    # --- 2. First-task retention plot ---
    _plot_first_task_retention(perf, task_names, output_dir)

    print(f"  Performance plots saved to {output_dir}")


def _plot_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    output_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
):
    """Plot a matrix as a heatmap with row/col labels."""
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

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
):
    """Plot how the first task's accuracy decays as more tasks are learned."""
    first_task = task_names[0]
    entries = perf[first_task]

    accs = []
    losses = []
    for entry in entries:
        if entry is None:
            accs.append(np.nan)
            losses.append(np.nan)
        else:
            acc = entry.get('accuracy')
            accs.append(acc if acc is not None else np.nan)
            losses.append(entry.get('loss', np.nan))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(accs))
    ax1.plot(list(x), accs, marker='o', markersize=5)
    ax1.set_xlabel("After Training on Task")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(t_labels(task_names), rotation=45, ha='right')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(list(x), losses, marker='o', markersize=5, color='red')
    ax2.set_xlabel("After Training on Task")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(t_labels(task_names), rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "first_task_retention.pdf")
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()
