"""Performance plotting module for RNN.

Reads performance_history.json and generates performance plots showing
loss, accuracy, and fixation accuracy across the task sequence.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_rnn_performance(exp_dir: str, output_dir: str) -> None:
    """
    Plot performance figures from saved performance_history.json.

    Generates:
    1. Loss matrix heatmap (eval_task x trained_task)
    2. Accuracy matrix heatmap
    3. Per-task accuracy over time (line plot)

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

    # Build matrices: rows = eval task, cols = after training on task_i
    loss_matrix = np.full((len(task_names), num_tasks), np.nan)
    acc_matrix = np.full((len(task_names), num_tasks), np.nan)
    fix_acc_matrix = np.full((len(task_names), num_tasks), np.nan)

    for i, task_name in enumerate(task_names):
        for j, entry in enumerate(perf[task_name]):
            if entry is None:
                continue
            loss_matrix[i, j] = entry.get('loss', np.nan)
            acc_val = entry.get('accuracy')
            acc_matrix[i, j] = acc_val if acc_val is not None else np.nan
            fix_val = entry.get('fix_accuracy')
            fix_acc_matrix[i, j] = fix_val if fix_val is not None else np.nan

    # --- 1. Loss heatmap ---
    _plot_matrix_heatmap(
        loss_matrix, task_names, task_names,
        title="Loss Matrix (eval_task x trained_after)",
        cbar_label="Loss",
        output_path=os.path.join(output_dir, "loss_matrix.png"),
        vmin=0, vmax=None, cmap='YlOrRd',
    )

    # --- 2. Accuracy heatmap ---
    _plot_matrix_heatmap(
        acc_matrix, task_names, task_names,
        title="Accuracy Matrix (eval_task x trained_after)",
        cbar_label="Accuracy",
        output_path=os.path.join(output_dir, "accuracy_matrix.png"),
        vmin=0, vmax=1, cmap='viridis',
    )

    # --- 3. Per-task accuracy line plot ---
    _plot_per_task_accuracy(acc_matrix, task_names, output_dir)

    # --- 4. First-task retention plot ---
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
    cmap: str = 'viridis',
):
    """Plot a matrix as a heatmap with row/col labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=7)

    # Annotate cells if matrix is small enough
    nrows, ncols = matrix.shape
    if nrows <= 20 and ncols <= 20:
        for i in range(nrows):
            for j in range(ncols):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f'{val:.2f}' if val < 10 else f'{val:.1f}'
                    ax.text(j, i, text, ha="center", va="center",
                            color="black", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()


def _plot_per_task_accuracy(
    acc_matrix: np.ndarray,
    task_names: List[str],
    output_dir: str,
):
    """Line plot of accuracy for each eval task over the training sequence."""
    fig, ax = plt.subplots(figsize=(14, 6))

    num_eval_tasks, num_train_stages = acc_matrix.shape
    x = range(num_train_stages)

    # Plot a subset to avoid clutter (first, middle, last trained tasks)
    indices_to_plot = [0]
    if num_eval_tasks > 2:
        indices_to_plot.append(num_eval_tasks // 2)
    if num_eval_tasks > 1:
        indices_to_plot.append(num_eval_tasks - 1)

    for i in indices_to_plot:
        vals = acc_matrix[i]
        mask = ~np.isnan(vals)
        if mask.any():
            ax.plot(np.array(x)[mask], vals[mask], marker='o', label=task_names[i], markersize=4)

    ax.set_title("Task Accuracy Over Training Sequence")
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "per_task_accuracy.png")
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
    ax1.set_title(f"First Task ({first_task}) — Accuracy Retention")
    ax1.set_xlabel("After Training on Task")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(list(x), losses, marker='o', markersize=5, color='red')
    ax2.set_title(f"First Task ({first_task}) — Loss Over Time")
    ax2.set_xlabel("After Training on Task")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "first_task_retention.png")
    plt.savefig(output_path)
    print(f"    Saved {output_path}")
    plt.close()
