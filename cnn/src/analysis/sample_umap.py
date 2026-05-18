"""Sample UMAP visualization of representational drift across checkpoints.

Pipeline (as per prompt.md):
  1. Concatenate reps from all checkpoints: X_all = [X^(1); ...; X^(T)] shape (T*N, D)
  2. Fit PCA on X_all, reduce to ``pca_dims`` (default 20).
  3. Fit UMAP on PCA output, reduce to 2D.
  4. Split Z_umap back into T chunks of shape (N, 2).
  5. Visualize: either per-checkpoint subplots (color=class) or a single plot
     (color=checkpoint). Optional trajectory lines between same sample across
     checkpoints.

NOTE: PCA and UMAP are fit ONCE on the concatenated matrix so all checkpoints
share the same embedding space.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def run_sample_umap(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
    pca_dims: int = 20,
    color_by: str = "class",
    show_trajectory: bool = False,
    trajectory_alpha: float = 0.05,
    trajectory_subsample: int = 50,
) -> None:
    """Run PCA → UMAP on concatenated checkpoint reps and save visualizations.

    Args:
        reps_cache: {task_idx: {layer_name: Tensor(N, D)}} from build_reps_cache.
        labels: Tensor(N,) class labels for probe samples.
        layer_names: Which layers to visualize.
        output_dir: Directory to save output PDFs.
        pca_dims: Number of PCA dimensions before UMAP (0 to skip PCA).
        color_by: "class" → subplots per checkpoint, color=class;
                  "checkpoint" → single plot, color=checkpoint index.
        show_trajectory: Draw lines connecting the same sample across checkpoints.
        trajectory_alpha: Alpha for trajectory lines.
        trajectory_subsample: Max number of samples to draw trajectories for.
    """
    try:
        import umap as umap_lib
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for sample_umap. Install via 'pip install umap-learn'."
        ) from e

    umap_dir = os.path.join(output_dir, "sample_umap")
    os.makedirs(umap_dir, exist_ok=True)

    sorted_task_indices = sorted(reps_cache.keys())
    T = len(sorted_task_indices)
    labels_np = labels.numpy()

    for layer in layer_names:
        print(f"  [sample_umap] layer: {layer}")

        # --- 1. Concatenate all checkpoints ---
        reps_list = [reps_cache[t][layer].numpy().astype(np.float32)
                     for t in sorted_task_indices]
        N = reps_list[0].shape[0]
        X_all = np.concatenate(reps_list, axis=0)  # (T*N, D)

        # --- 2. PCA ---
        pca_var_explained: Optional[float] = None
        if pca_dims > 0 and pca_dims < X_all.shape[1]:
            from sklearn.decomposition import PCA
            print(f"    PCA: {X_all.shape[1]} → {pca_dims} dims...")
            pca = PCA(n_components=pca_dims, random_state=42)
            X_pca = pca.fit_transform(X_all)  # (T*N, pca_dims)
            pca_var_explained = float(pca.explained_variance_ratio_.sum())
            print(f"    PCA explained variance: {pca_var_explained * 100:.1f}%")
        else:
            X_pca = X_all

        # --- 3. UMAP ---
        print(f"    UMAP: {X_pca.shape[1]} → 2 dims (n={X_pca.shape[0]})...")
        reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
        Z_all = reducer.fit_transform(X_pca)  # (T*N, 2)

        # --- 4. Split back ---
        Z_list = [Z_all[i * N:(i + 1) * N] for i in range(T)]  # list of (N, 2)

        safe_layer = layer.replace(".", "_").replace("/", "_")

        # --- 5. Visualize ---
        if color_by == "checkpoint":
            _plot_by_checkpoint(
                Z_list, sorted_task_indices, safe_layer, umap_dir,
                show_trajectory, trajectory_alpha, trajectory_subsample,
                pca_dims=pca_dims, pca_var_explained=pca_var_explained,
            )
        else:
            _plot_by_class(
                Z_list, labels_np, sorted_task_indices, safe_layer, umap_dir,
                show_trajectory, trajectory_alpha, trajectory_subsample,
                pca_dims=pca_dims, pca_var_explained=pca_var_explained,
            )

    print(f"  [sample_umap] Results saved to {umap_dir}")


def _pca_subtitle(pca_dims: int, pca_var_explained: Optional[float]) -> str:
    """Build a subtitle string describing PCA compression."""
    if pca_var_explained is None:
        return ""
    return f"PCA {pca_dims}D: {pca_var_explained * 100:.1f}% var explained"


def _plot_by_class(
    Z_list: List[np.ndarray],
    labels_np: np.ndarray,
    task_indices: List[int],
    safe_layer: str,
    umap_dir: str,
    show_trajectory: bool,
    traj_alpha: float,
    traj_subsample: int,
    pca_dims: int = 20,
    pca_var_explained: Optional[float] = None,
) -> None:
    """One subplot per checkpoint, color = class label."""
    T = len(Z_list)
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    cmap = cm.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    class_to_color = {int(c): cmap(i) for i, c in enumerate(unique_classes)}

    # Unified axis limits
    all_z = np.concatenate(Z_list, axis=0)
    x_min, x_max = all_z[:, 0].min(), all_z[:, 0].max()
    y_min, y_max = all_z[:, 1].min(), all_z[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    ncols = min(T, 5)
    nrows = (T + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if T > 1 else [axes]

    for ax_idx, (ax, z, t) in enumerate(zip(axes_flat, Z_list, task_indices)):
        colors = [class_to_color[int(lb)] for lb in labels_np]
        ax.scatter(z[:, 0], z[:, 1], c=colors, s=4, alpha=0.6, linewidths=0)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        ax.text(0.02, 0.97, f"T{ax_idx + 1}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10)

    # Hide unused axes
    for ax in axes_flat[T:]:
        ax.set_visible(False)

    if show_trajectory:
        _draw_trajectories(axes_flat[:T], Z_list, traj_alpha, traj_subsample)

    subtitle = _pca_subtitle(pca_dims, pca_var_explained)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=10,
                 color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(umap_dir, f"umap_by_class_{safe_layer}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"    Saved {out_path}")


def _plot_by_checkpoint(
    Z_list: List[np.ndarray],
    task_indices: List[int],
    safe_layer: str,
    umap_dir: str,
    show_trajectory: bool,
    traj_alpha: float,
    traj_subsample: int,
    pca_dims: int = 20,
    pca_var_explained: Optional[float] = None,
) -> None:
    """Single plot, color = checkpoint index."""
    T = len(Z_list)
    cmap = cm.get_cmap("plasma", T)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (z, t) in enumerate(zip(Z_list, task_indices)):
        ax.scatter(z[:, 0], z[:, 1], c=[cmap(i)] * len(z),
                   s=4, alpha=0.5, linewidths=0, label=f"T{i + 1}")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    if show_trajectory:
        _draw_trajectories([ax] * T, Z_list, traj_alpha, traj_subsample)

    subtitle = _pca_subtitle(pca_dims, pca_var_explained)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, -0.08), xycoords="axes fraction",
                    ha="center", va="top", fontsize=10, color="gray")

    plt.tight_layout()
    out_path = os.path.join(umap_dir, f"umap_by_checkpoint_{safe_layer}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"    Saved {out_path}")


def _draw_trajectories(
    axes: List,
    Z_list: List[np.ndarray],
    alpha: float,
    subsample: int,
) -> None:
    """Draw lines connecting the same sample index across consecutive checkpoints."""
    N = Z_list[0].shape[0]
    sample_indices = np.random.choice(N, size=min(subsample, N), replace=False)
    for s in sample_indices:
        xs = [z[s, 0] for z in Z_list]
        ys = [z[s, 1] for z in Z_list]
        if len(set(axes)) == 1:
            axes[0].plot(xs, ys, color="gray", alpha=alpha, linewidth=0.5, zorder=0)
        else:
            for i in range(len(Z_list) - 1):
                axes[i].plot(
                    [Z_list[i][s, 0], Z_list[i + 1][s, 0]],
                    [Z_list[i][s, 1], Z_list[i + 1][s, 1]],
                    color="gray", alpha=alpha, linewidth=0.5, zorder=0,
                )
