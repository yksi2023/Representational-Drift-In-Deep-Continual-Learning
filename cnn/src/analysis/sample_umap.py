"""Sample UMAP visualization of representational drift across checkpoints.

Pipeline (as per prompt.md):
  1. Concatenate reps from all checkpoints: X_all = [X^(1); ...; X^(T)] shape (T*N, D)
  2. Fit PCA on X_all with a fixed cap of 512 components.
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

from src.analysis._plot_utils import (
    AXIS_LABEL_SIZE,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    OPEN_MARKER_LINEWIDTH,
    OPEN_MARKER_SIZE,
    SINGLE_FIGSIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    apply_paper_axis_style,
    categorical_colors,
    savefig_compact,
)


def run_sample_umap(
    reps_cache: Dict[int, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_names: List[str],
    output_dir: str,
    pca_var_threshold: float = 0.90,
    color_by: str = "class",
    show_trajectory: bool = False,
    trajectory_alpha: float = 0.05,
    trajectory_subsample: int = 50,
    max_display_per_class: int = 50,
    save_pca_diagnostics: bool = True,
) -> None:
    """Run PCA → UMAP on concatenated checkpoint reps and save visualizations.

    Args:
        reps_cache: {task_idx: {layer_name: Tensor(N, D)}} from build_reps_cache.
        labels: Tensor(N,) class labels for probe samples.
        layer_names: Which layers to visualize.
        output_dir: Directory to save output PDFs.
        pca_var_threshold: Positive values enable PCA with a fixed 512-component
            cap. Set to 0 to skip PCA entirely. The name is retained for CLI compatibility.
        color_by: "class" → subplots per checkpoint, color=class;
                  "checkpoint" → single plot, color=checkpoint index.
        show_trajectory: Draw lines connecting the same sample across checkpoints.
        trajectory_alpha: Alpha for trajectory lines.
        trajectory_subsample: Max number of samples to draw trajectories for.
        save_pca_diagnostics: Save PCA variance and scree plots alongside UMAP.
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

        # --- 2. PCA (variance-threshold) ---
        pca_var_explained: Optional[float] = None
        pca_n_components: Optional[int] = None
        pca_eigenvalues: Optional[np.ndarray] = None
        pca_explained_ratio: Optional[np.ndarray] = None
        if pca_var_threshold > 0:
            from sklearn.decomposition import PCA
            n_components = min(512, X_all.shape[0] - 1, X_all.shape[1] - 1)
            print(f"    PCA: {X_all.shape[1]}D → {n_components} components (fixed 512 cap)...")
            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
            X_pca = pca.fit_transform(X_all)
            pca_n_components = n_components
            pca_var_explained = float(np.sum(pca.explained_variance_ratio_))
            pca_eigenvalues = pca.explained_variance_
            pca_explained_ratio = pca.explained_variance_ratio_
            if save_pca_diagnostics:
                _plot_pca_diagnostics(
                    pca_eigenvalues,
                    pca_explained_ratio,
                    pca_n_components,
                    safe_name=layer.replace(".", "_").replace("/", "_"),
                    umap_dir=umap_dir,
                )
            print(f"    PCA kept {pca_n_components} components, explained {pca_var_explained * 100:.1f}%")
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
            )
        else:
            _plot_by_class(
                Z_list, labels_np, sorted_task_indices, safe_layer, umap_dir,
                show_trajectory, trajectory_alpha, trajectory_subsample,
                max_display_per_class=max_display_per_class,
            )
            _plot_by_class_paper_subset(
                Z_list, labels_np, sorted_task_indices, safe_layer, umap_dir,
                checkpoint_numbers=[1, 7, 13, 20],
                max_display_per_class=max_display_per_class,
            )

    print(f"  [sample_umap] Results saved to {umap_dir}")


def _plot_pca_diagnostics(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    selected_k: int,
    safe_name: str,
    umap_dir: str,
) -> None:
    """Plot cumulative explained variance and PCA scree diagnostics."""
    pcs = np.arange(1, len(eigenvalues) + 1)
    cumvar = np.cumsum(explained_ratio)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.plot(pcs, cumvar, linewidth=2.5)
    ax.axvline(selected_k, color="red", linestyle="--", linewidth=2, label=f"k={selected_k}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.02)
    apply_paper_axis_style(ax, legend=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    savefig_compact(fig, os.path.join(umap_dir, f"pca_explained_variance_{safe_name}.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.plot(pcs, eigenvalues, linewidth=2.5)
    ax.axvline(selected_k, color="red", linestyle="--", linewidth=2, label=f"k={selected_k}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue")
    apply_paper_axis_style(ax, legend=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    savefig_compact(fig, os.path.join(umap_dir, f"pca_scree_{safe_name}.pdf"))
    plt.close(fig)


def _save_standalone_legend(handles, title: Optional[str], output_path: str) -> None:
    """Save legend handles as a compact figure separate from the UMAP plot."""
    if not handles:
        return
    ncol = min(5, len(handles))
    nrows = (len(handles) + ncol - 1) // ncol
    fig = plt.figure(figsize=(max(5.0, 1.8 * ncol), 0.8 + 0.65 * nrows))
    fig.legend(
        handles=handles,
        loc="center",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        ncol=ncol,
        title=title,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    savefig_compact(fig, output_path)
    plt.close(fig)


def _subsample_display_mask(
    labels_np: np.ndarray, max_per_class: int, rng: np.random.Generator
) -> np.ndarray:
    """Return boolean mask keeping at most max_per_class samples per class."""
    mask = np.zeros(len(labels_np), dtype=bool)
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        chosen = rng.choice(idx, size=min(max_per_class, len(idx)), replace=False)
        mask[chosen] = True
    return mask


def _plot_by_class(
    Z_list: List[np.ndarray],
    labels_np: np.ndarray,
    task_indices: List[int],
    safe_layer: str,
    umap_dir: str,
    show_trajectory: bool,
    traj_alpha: float,
    traj_subsample: int,
    max_display_per_class: int = 50,
) -> None:
    """One subplot per checkpoint, color = class label."""
    T = len(Z_list)
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    palette = categorical_colors(n_classes)
    class_to_color = {int(c): palette[i] for i, c in enumerate(unique_classes)}

    # Per-class display subsample mask (same indices across all checkpoints)
    rng = np.random.default_rng(42)
    display_mask = _subsample_display_mask(labels_np, max_display_per_class, rng)

    # Unified axis limits (computed from full embedding, not just display subset)
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
        z_disp = z[display_mask]
        labels_disp = labels_np[display_mask]
        colors = [class_to_color[int(lb)] for lb in labels_disp]
        ax.scatter(
            z_disp[:, 0], z_disp[:, 1],
            facecolors="none", edgecolors=colors,
            s=OPEN_MARKER_SIZE, alpha=0.8, linewidths=OPEN_MARKER_LINEWIDTH,
        )
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        apply_paper_axis_style(ax)
        ax.text(0.02, 0.97, f"T{ax_idx + 1}", transform=ax.transAxes,
                va="top", ha="left", fontsize=TICK_LABEL_SIZE, fontweight="bold")

    # Hide unused axes
    for ax in axes_flat[T:]:
        ax.set_visible(False)

    if show_trajectory:
        _draw_trajectories(axes_flat[:T], Z_list, traj_alpha, traj_subsample)

    # Class legend (open-circle proxy markers)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="none", markeredgecolor=class_to_color[int(c)],
            markeredgewidth=OPEN_MARKER_LINEWIDTH * 1.4, markersize=10,
            label=f"Class {int(c)}",
        )
        for c in unique_classes
    ]
    _save_standalone_legend(
        legend_handles,
        None,
        os.path.join(umap_dir, "legend_by_class.pdf"),
    )

    plt.tight_layout()
    out_path = os.path.join(umap_dir, f"umap_by_class_{safe_layer}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"    Saved {out_path}")


def _plot_by_class_paper_subset(
    Z_list: List[np.ndarray],
    labels_np: np.ndarray,
    task_indices: List[int],
    safe_layer: str,
    umap_dir: str,
    checkpoint_numbers: List[int],
    max_display_per_class: int = 50,
) -> None:
    """One-row paper figure for selected CNN checkpoints, color = class."""
    selected = []
    for ckpt_num in checkpoint_numbers:
        if ckpt_num in task_indices:
            selected.append((task_indices.index(ckpt_num), ckpt_num))
        elif 0 <= ckpt_num - 1 < len(task_indices):
            selected.append((ckpt_num - 1, ckpt_num))

    if len(selected) != len(checkpoint_numbers):
        available = ", ".join(str(t) for t in task_indices)
        print(
            f"    Skipping paper UMAP subset for {safe_layer}: requested "
            f"{checkpoint_numbers}, available checkpoints are [{available}]"
        )
        return

    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    palette = categorical_colors(n_classes)
    class_to_color = {int(c): palette[i] for i, c in enumerate(unique_classes)}

    rng = np.random.default_rng(42)
    display_mask = _subsample_display_mask(labels_np, max_display_per_class, rng)

    all_z = np.concatenate(Z_list, axis=0)
    x_min, x_max = all_z[:, 0].min(), all_z[:, 0].max()
    y_min, y_max = all_z[:, 1].min(), all_z[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (pos, label_num) in zip(axes, selected):
        z_disp = Z_list[pos][display_mask]
        labels_disp = labels_np[display_mask]
        colors = [class_to_color[int(lb)] for lb in labels_disp]
        ax.scatter(
            z_disp[:, 0], z_disp[:, 1],
            facecolors="none", edgecolors=colors,
            s=OPEN_MARKER_SIZE, alpha=0.85, linewidths=OPEN_MARKER_LINEWIDTH,
        )
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        apply_paper_axis_style(ax)
        ax.text(0.03, 0.96, f"T{label_num}", transform=ax.transAxes,
                va="top", ha="left", fontsize=TICK_LABEL_SIZE, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(umap_dir, f"umap_by_class_paper_{safe_layer}.pdf")
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
) -> None:
    """Single plot, color = checkpoint index."""
    T = len(Z_list)
    palette = categorical_colors(T)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (z, t) in enumerate(zip(Z_list, task_indices)):
        ax.scatter(
            z[:, 0], z[:, 1],
            facecolors="none", edgecolors=[palette[i]] * len(z),
            s=OPEN_MARKER_SIZE, alpha=0.7, linewidths=OPEN_MARKER_LINEWIDTH,
            label=f"T{i + 1}",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    apply_paper_axis_style(ax)

    if show_trajectory:
        _draw_trajectories([ax] * T, Z_list, traj_alpha, traj_subsample)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="none", markeredgecolor=palette[i],
            markeredgewidth=OPEN_MARKER_LINEWIDTH * 1.4, markersize=10,
            label=f"T{i + 1}",
        )
        for i in range(T)
    ]
    _save_standalone_legend(
        legend_handles,
        "Checkpoint",
        os.path.join(umap_dir, "legend_by_checkpoint.pdf"),
    )

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
