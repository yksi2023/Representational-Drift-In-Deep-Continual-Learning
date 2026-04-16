"""Cross-checkpoint Population Vector (PV) similarity analysis.

For each probe task, builds a full cross-checkpoint similarity matrix where
axes are (checkpoint_0_t0, ..., checkpoint_0_tN, checkpoint_1_t0, ...).
Each element is the similarity between two PVs (hidden states at specific
timesteps from specific checkpoints), averaged over the batch.

The diagonal blocks show within-checkpoint temporal PV similarity; off-diagonal
blocks show how PVs at each timestep change across training.

Both cosine similarity and Pearson correlation variants are produced.
"""
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.analysis.baseline_drift import _load_reps_from_npz


def _reshape_to_3d(flat: np.ndarray, hidden_size: int) -> torch.Tensor:
    """Reshape (Batch, Seq_len*Hidden_size) -> (Batch, Seq_len, Hidden_size)."""
    batch_size, total_dim = flat.shape
    seq_len = total_dim // hidden_size
    assert total_dim == seq_len * hidden_size, \
        f"total_dim={total_dim} not divisible by hidden_size={hidden_size}"
    return torch.from_numpy(flat).float().reshape(batch_size, seq_len, hidden_size)


def _compute_cross_similarity(
    reps_a: torch.Tensor, reps_b: torch.Tensor, metric: str = "cosine"
) -> np.ndarray:
    """Similarity between every (timestep_i in a, timestep_j in b), averaged over batch.

    Args:
        reps_a: (Batch, T_a, Hidden)
        reps_b: (Batch, T_b, Hidden)
        metric: "cosine" or "pearson"

    Returns:
        (T_a, T_b) similarity matrix.
    """
    a, b = reps_a, reps_b
    if metric == "pearson":
        a = a - a.mean(dim=2, keepdim=True)
        b = b - b.mean(dim=2, keepdim=True)
    a_norm = F.normalize(a, p=2, dim=2)
    b_norm = F.normalize(b, p=2, dim=2)
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, T_a, T_b)
    return sim.mean(dim=0).numpy()


def _plot_full_matrix(
    full_matrix: np.ndarray,
    seq_len: int,
    n_checkpoints: int,
    task_names: List[str],
    probe_task: str,
    output_path: str,
    metric_label: str = "Cosine Similarity",
):
    """Plot the full (N_checkpoints*Seq_len) x (N_checkpoints*Seq_len) matrix."""
    total = full_matrix.shape[0]
    fig_size = min(max(10, total / 40), 30)
    fig, ax = plt.subplots(figsize=(fig_size + 2, fig_size))

    full_matrix = np.clip(full_matrix, 0, 1)
    im = ax.imshow(full_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal',
                   interpolation='none')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(metric_label, rotation=-90, va="bottom")

    # Draw grid lines at checkpoint boundaries
    for k in range(1, n_checkpoints):
        pos = k * seq_len - 0.5
        ax.axhline(y=pos, color='white', linewidth=0.5, alpha=0.7)
        ax.axvline(x=pos, color='white', linewidth=0.5, alpha=0.7)

    # Label each checkpoint block at its centre
    centres = [(k * seq_len + seq_len / 2) for k in range(n_checkpoints)]
    labels = [task_names[k] if k < len(task_names) else f"task_{k}"
              for k in range(n_checkpoints)]
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(f"Cross-Checkpoint PV {metric_label}\nprobe: {probe_task}")
    ax.set_xlabel("Checkpoint / Time step")
    ax.set_ylabel("Checkpoint / Time step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _build_full_matrix(reps_3d_list: List[torch.Tensor], metric: str) -> np.ndarray:
    """Build the full (N*T) x (N*T) cross-checkpoint similarity matrix."""
    n_checkpoints = len(reps_3d_list)
    seq_len = reps_3d_list[0].shape[1]
    total_len = n_checkpoints * seq_len
    full_matrix = np.zeros((total_len, total_len), dtype=np.float32)
    for i in range(n_checkpoints):
        for j in range(i, n_checkpoints):
            block = _compute_cross_similarity(reps_3d_list[i], reps_3d_list[j], metric=metric)
            ri, rj = i * seq_len, j * seq_len
            full_matrix[ri:ri+seq_len, rj:rj+seq_len] = block
            if i != j:
                full_matrix[rj:rj+seq_len, ri:ri+seq_len] = block.T
    return full_matrix


def run_temporal_similarity(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
    hidden_size: int = 256,
) -> None:
    """Generate temporal hidden state similarity analysis.

    For each probe task produces two full cross-checkpoint matrices
    (cosine similarity and Pearson correlation).

    Args:
        exp_dir: Experiment directory containing representations/.
        probe_tasks: Which tasks' representations to analyze.
        task_names: Ordered list of all task names in the sequence.
        output_dir: Directory to save results.
        hidden_size: Hidden size of the RNN (needed to reshape flat reps).
    """
    reps_dir = os.path.join(exp_dir, "representations")
    out_subdir = os.path.join(output_dir, "temporal_similarity")
    os.makedirs(out_subdir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  Temporal similarity for probe: {probe_task}")
        raw_reps = _load_reps_from_npz(reps_dir, probe_task)
        sorted_indices = sorted(raw_reps.keys())
        n_checkpoints = len(sorted_indices)

        # Reshape all checkpoints to 3D
        reps_3d_list = []
        for task_idx in sorted_indices:
            reps_3d = _reshape_to_3d(raw_reps[task_idx], hidden_size)
            reps_3d_list.append(reps_3d)

        seq_len = reps_3d_list[0].shape[1]
        total_len = n_checkpoints * seq_len

        for metric, label in [("cosine", "Cosine Similarity"), ("pearson", "Pearson Correlation")]:
            full_matrix = _build_full_matrix(reps_3d_list, metric)
            fname = f"cross_checkpoint_{metric}_{probe_task}.png"
            out_path = os.path.join(out_subdir, fname)
            _plot_full_matrix(full_matrix, seq_len, n_checkpoints, task_names,
                              probe_task, out_path, metric_label=label)
            print(f"    {label} matrix ({total_len}x{total_len}) saved to {out_path}")
