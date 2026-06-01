"""UMAP visualization of RNN direction representations across checkpoints.

Hybrid approach:
  - First run: load checkpoints, generate controlled fixed-direction trials,
    extract epoch-specific hidden states, cache to representations/umap_*.npz.
  - Subsequent runs: read cached .npz (no GPU needed), consistent with other
    analyses.

Pipeline per probe task:
  1. Generate controlled trials with n_directions evenly-spaced stimulus
     directions, n_trials_per_dir repetitions each, fixed timing.
  2. For each checkpoint, run forward pass → extract hidden states.
  3. Compute epoch-specific reps (time-averaged hidden state per epoch)
     and full STPV (concatenated hidden states).
  4. Cache all to .npz.
  5. PCA → UMAP → per-checkpoint subplot plots colored by direction.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.analysis._plot_utils import (
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_SIZE,
    SINGLE_FIGSIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    apply_paper_axis_style,
    savefig_compact,
)

from datasets import Trial, get_default_config
from src.models import CognitiveRNN
from src.checkpoints import load_model, list_checkpoints

# Go-family tasks that have a single stimulus direction
_GO_FAMILY = {
    'fdgo', 'fdanti', 'reactgo', 'reactanti', 'delaygo', 'delayanti',
}

# Fixed timing parameters per task family (in timestep units, dt=20)
# Chosen as median of each task's random ranges.
_TIMING = {
    'fdgo':       {'stim_ons': 25, 'fix_offs': 75, 'stim_mod': 1},  # (500/20, stim_ons + 1000/20)
    'fdanti':     {'stim_ons': 25, 'fix_offs': 75, 'stim_mod': 1},
    'reactgo':    {'stim_ons': 75, 'stim_mod': 1},                   # (1500/20)
    'reactanti':  {'stim_ons': 75, 'stim_mod': 1},
    'delaygo':    {'stim_ons': 25, 'stim_offs': 45, 'fix_offs': 85, 'stim_mod': 1},
    'delayanti':  {'stim_ons': 25, 'stim_offs': 45, 'fix_offs': 85, 'stim_mod': 1},
}


# ======================================================================
# 1a. Controlled trial generator
# ======================================================================

def _build_controlled_trial(
    config: dict,
    task_name: str,
    directions: np.ndarray,
    n_per_dir: int,
) -> Tuple:
    """Build a Trial with fixed directions and fixed timing for Go-family tasks.

    Args:
        config: RNN task config (from get_default_config).
        task_name: e.g. 'fdgo', 'delayanti'.
        directions: array of angles in radians, shape (n_directions,).
        n_per_dir: repetitions per direction.

    Returns:
        trial: Trial object ready for forward pass.
        direction_labels: int array (batch_size,) with direction index 0..n_dir-1.
        epoch_dict: {epoch_name: (start_t, end_t)} in timestep indices.
    """
    n_dir = len(directions)
    batch_size = n_dir * n_per_dir

    # Repeat each direction n_per_dir times
    stim_locs = np.repeat(directions, n_per_dir)
    direction_labels = np.repeat(np.arange(n_dir), n_per_dir)

    anti = task_name in ('fdanti', 'reactanti', 'delayanti')
    response_locs = (stim_locs + np.pi) % (2 * np.pi) if anti else stim_locs.copy()

    timing = _TIMING[task_name]
    stim_mod = timing['stim_mod']

    if task_name in ('fdgo', 'fdanti'):
        stim_ons = timing['stim_ons']
        fix_offs = timing['fix_offs']
        tdim = fix_offs + int(500 / config['dt'])
        check_ons = fix_offs + int(100 / config['dt'])

        trial = Trial(config, tdim, batch_size)
        trial.add('fix_in', offs=fix_offs)
        trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
        trial.add('fix_out', offs=fix_offs)
        trial.add('out', response_locs, ons=fix_offs)
        trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

        epoch_dict = {
            'fix1': (0, stim_ons),
            'stim1': (stim_ons, fix_offs),
            'go1': (fix_offs, tdim),
        }

    elif task_name in ('reactgo', 'reactanti'):
        stim_ons = timing['stim_ons']
        tdim = stim_ons + int(500 / config['dt'])
        check_ons = stim_ons + int(100 / config['dt'])

        trial = Trial(config, tdim, batch_size)
        trial.add('fix_in')
        trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
        trial.add('fix_out', offs=stim_ons)
        trial.add('out', response_locs, ons=stim_ons)
        trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

        epoch_dict = {
            'fix1': (0, stim_ons),
            'go1': (stim_ons, tdim),
        }

    elif task_name in ('delaygo', 'delayanti'):
        stim_ons = timing['stim_ons']
        stim_offs = timing['stim_offs']
        fix_offs = timing['fix_offs']
        tdim = fix_offs + int(500 / config['dt'])
        check_ons = fix_offs + int(100 / config['dt'])

        trial = Trial(config, tdim, batch_size)
        trial.add('fix_in', offs=fix_offs)
        trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
        trial.add('fix_out', offs=fix_offs)
        trial.add('out', response_locs, ons=fix_offs)
        trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

        epoch_dict = {
            'fix1': (0, stim_ons),
            'stim1': (stim_ons, stim_offs),
            'delay1': (stim_offs, fix_offs),
            'go1': (fix_offs, tdim),
        }
    else:
        raise ValueError(f"Unsupported task for UMAP: {task_name}")

    # Add rule input and noise (same as _finalize_trial)
    trial.add_rule(task_name)
    rng_backup = config.get('rng')
    config['rng'] = np.random.RandomState(42)
    trial.add_x_noise()
    config['rng'] = rng_backup

    return trial, direction_labels, epoch_dict


# ======================================================================
# 1b. Extraction + caching
# ======================================================================

def _reconstruct_model(exp_config: dict, config: dict, device: torch.device):
    """Reconstruct CognitiveRNN from saved experiment_config.json."""
    model = CognitiveRNN(
        input_size=config['n_input'],
        hidden_size=exp_config.get('hidden_size', 256),
        output_size=config['n_output'],
        dt=config['dt'],
        tau=config['dt'] / config.get('alpha', 0.2),
        sigma_rec=exp_config.get('sigma_rec', 0.05),
        activation=exp_config.get('activation', 'softplus'),
        w_rec_init=exp_config.get('w_rec_init', 'diag'),
    )
    model.to(device)
    model.eval()
    return model


def _extract_epoch_reps(
    states: torch.Tensor,
    epoch_dict: Dict[str, Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Extract full STPV, epoch means, and selected last-timestep PVs.

    Args:
        states: (Seq_len, Batch, Hidden) tensor of hidden states.
        epoch_dict: {epoch_name: (start_t, end_t)}.

    Returns:
        dict with keys like 'stpv', 'fix1', 'stim1', 'go1', 'stim_last',
        'go_last', etc.
        Each value is np.ndarray of shape (Batch, D).
    """
    # (Seq_len, Batch, Hidden) → (Batch, Seq_len, Hidden)
    states_bth = states.transpose(0, 1)
    B, T, H = states_bth.shape

    reps = {}
    # Full STPV
    reps['stpv'] = states_bth.reshape(B, T * H).cpu().numpy()

    # Per-epoch mean
    for epoch_name, (t_start, t_end) in epoch_dict.items():
        epoch_states = states_bth[:, t_start:t_end, :]  # (B, t_end-t_start, H)
        reps[epoch_name] = epoch_states.mean(dim=1).cpu().numpy()  # (B, H)

    # Last-timestep population vectors for stimulus and go epochs.
    # These are single hidden states h_t, not time-averaged epoch activity.
    for prefix, out_name in (("stim", "stim_last"), ("go", "go_last")):
        matching_epochs = [
            (name, bounds) for name, bounds in epoch_dict.items()
            if name.startswith(prefix)
        ]
        if not matching_epochs:
            continue
        _name, (_t_start, t_end) = matching_epochs[-1]
        reps[out_name] = states_bth[:, t_end - 1, :].cpu().numpy()  # (B, H)

    return reps


def _expected_last_rep_names(task_name: str) -> List[str]:
    """Return cache-required last-timestep PV names for a Go-family task."""
    names = ["go_last"]
    if task_name in ("fdgo", "fdanti", "delaygo", "delayanti"):
        names.insert(0, "stim_last")
    return names


def _last_rep_names_for_epoch_dict(epoch_dict: Dict[str, Tuple[int, int]]) -> List[str]:
    """Return last-timestep PV representation names available for epochs."""
    names = []
    if any(name.startswith("stim") for name in epoch_dict):
        names.append("stim_last")
    if any(name.startswith("go") for name in epoch_dict):
        names.append("go_last")
    return names


def _extract_or_load_umap_reps(
    exp_dir: str,
    probe_task: str,
    task_names: List[str],
    n_directions: int = 8,
    n_per_dir: int = 40,
) -> Optional[Dict]:
    """Load cached umap reps or extract from checkpoints and cache.

    Returns:
        dict with keys:
          'direction_labels': int array (batch,)
          'epoch_names': list of epoch names
          'n_checkpoints': int
          '{epoch_name}_after_task_{i}': np.ndarray per checkpoint per epoch
        Or None if probe_task is not a Go-family task.
    """
    reps_dir = os.path.join(exp_dir, "representations")
    cache_path = os.path.join(reps_dir, f"umap_{probe_task}.npz")

    # --- Try loading cache ---
    if os.path.exists(cache_path):
        print(f"    Loading cached UMAP reps from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        result = {k: data[k] for k in data.files}
        # Convert epoch_names from ndarray back to list
        if 'epoch_names' in result:
            result['epoch_names'] = list(result['epoch_names'])
        epoch_names = result.get('epoch_names', [])
        missing = [name for name in _expected_last_rep_names(probe_task) if name not in epoch_names]
        if not missing:
            return result
        print(f"    Cached UMAP reps missing {missing}; re-extracting and updating cache.")

    # --- Not cached: extract from checkpoints ---
    if probe_task not in _GO_FAMILY:
        print(f"    Skipping UMAP for '{probe_task}' (not a Go-family task)")
        return None

    # Load experiment config
    exp_config_path = os.path.join(exp_dir, "experiment_config.json")
    if not os.path.exists(exp_config_path):
        print(f"    Cannot find {exp_config_path}, skipping UMAP extraction.")
        return None
    with open(exp_config_path, "r", encoding="utf-8") as f:
        exp_config = json.load(f)

    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build controlled trial
    directions = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    trial, direction_labels, epoch_dict = _build_controlled_trial(
        config, probe_task, directions, n_per_dir
    )
    epoch_names = ['stpv'] + list(epoch_dict.keys()) + _last_rep_names_for_epoch_dict(epoch_dict)

    # Reconstruct model shell
    model = _reconstruct_model(exp_config, config, device)

    # Find checkpoints
    ckpt_map = list_checkpoints(exp_dir)
    if not ckpt_map:
        print(f"    No checkpoints found in {exp_dir}, skipping UMAP extraction.")
        return None

    sorted_indices = sorted(ckpt_map.keys())
    n_ckpt = len(sorted_indices)
    print(f"    Extracting UMAP reps from {n_ckpt} checkpoints (device={device})...")

    # Prepare trial tensor
    x_tensor = torch.tensor(trial.x, dtype=torch.float32, device=device)

    # Extract representations
    save_dict = {
        'direction_labels': direction_labels,
        'epoch_names': np.array(epoch_names),
        'n_checkpoints': np.array(n_ckpt),
    }

    for task_idx in sorted_indices:
        load_model(model, exp_dir, task_idx, map_location=device)
        model.eval()

        with torch.no_grad():
            _, states = model(x_tensor, return_all_states=True)  # (T, B, H)

        reps = _extract_epoch_reps(states, epoch_dict)
        for rep_name, rep_array in reps.items():
            save_dict[f"{rep_name}_after_task_{task_idx}"] = rep_array

    # Save cache
    os.makedirs(reps_dir, exist_ok=True)
    np.savez_compressed(cache_path, **save_dict)
    print(f"    Cached UMAP reps to {cache_path}")

    # Convert epoch_names back to list for downstream
    save_dict['epoch_names'] = epoch_names
    return save_dict


# ======================================================================
# 1c. PCA → UMAP
# ======================================================================

def _pca_umap(
    reps_per_ckpt: List[np.ndarray],
    pca_var_threshold: float = 0.90,
) -> Tuple[List[np.ndarray], Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Run PCA → UMAP on concatenated checkpoint reps.

    Args:
        reps_per_ckpt: list of (N, D) arrays, one per checkpoint.
        pca_var_threshold: cumulative variance threshold for PCA.

    Returns:
        Z_list: list of (N, 2) UMAP embeddings per checkpoint.
        pca_k: number of PCA components kept.
        pca_var: actual variance explained.
        eigenvalues: PCA eigenvalues, if PCA was run.
        explained_ratio: PCA explained variance ratios, if PCA was run.
    """
    try:
        import umap as umap_lib
    except ImportError as e:
        raise ImportError(
            "umap-learn is required. Install via 'pip install umap-learn'."
        ) from e

    X_all = np.concatenate(reps_per_ckpt, axis=0).astype(np.float32)
    N = reps_per_ckpt[0].shape[0]
    T = len(reps_per_ckpt)

    # PCA
    pca_k, pca_var = None, None
    eigenvalues, explained_ratio = None, None
    if pca_var_threshold > 0 and X_all.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_cap = min(512, X_all.shape[0] - 1, X_all.shape[1] - 1)
        pca = PCA(n_components=n_cap, svd_solver="randomized", random_state=42)
        pca.fit(X_all)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, pca_var_threshold) + 1)
        k = min(k, n_cap)
        X_pca = pca.transform(X_all)[:, :k]
        pca_k = k
        pca_var = float(cumvar[k - 1])
        eigenvalues = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_
        print(f"      PCA: {X_all.shape[1]}D → {k}D ({pca_var*100:.1f}% var)")
    else:
        X_pca = X_all

    # UMAP
    reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
    Z_all = reducer.fit_transform(X_pca)

    Z_list = [Z_all[i * N:(i + 1) * N] for i in range(T)]
    return Z_list, pca_k, pca_var, eigenvalues, explained_ratio


# ======================================================================
# 1d. Plotting
# ======================================================================

def _pca_subtitle(pca_k: Optional[int], pca_var: Optional[float]) -> str:
    if pca_k is None or pca_var is None:
        return ""
    return f"PCA {pca_k}D: {pca_var * 100:.1f}% var explained"


def _plot_pca_diagnostics(
    eigenvalues: Optional[np.ndarray],
    explained_ratio: Optional[np.ndarray],
    selected_k: Optional[int],
    safe_name: str,
    umap_dir: str,
) -> None:
    """Plot cumulative explained variance and PCA scree diagnostics."""
    if eigenvalues is None or explained_ratio is None or selected_k is None:
        return

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


def _plot_direction_umap(
    Z_list: List[np.ndarray],
    direction_labels: np.ndarray,
    task_indices: List[int],
    rep_name: str,
    probe_task: str,
    umap_dir: str,
    pca_k: Optional[int] = None,
    pca_var: Optional[float] = None,
    max_display_per_class: int = 40,
) -> None:
    """Plot UMAP subplots per checkpoint, colored by stimulus direction."""
    T = len(Z_list)
    unique_dirs = np.unique(direction_labels)
    n_dirs = len(unique_dirs)

    cmap = cm.get_cmap("tab10", max(n_dirs, 10))
    dir_to_color = {int(d): cmap(i) for i, d in enumerate(unique_dirs)}
    dir_angles = np.linspace(0, 360, n_dirs, endpoint=False).astype(int)
    dir_to_label = {int(d): f"{dir_angles[i]}\u00b0" for i, d in enumerate(unique_dirs)}

    # Display subsample mask
    rng = np.random.default_rng(42)
    mask = np.zeros(len(direction_labels), dtype=bool)
    for d in unique_dirs:
        idx = np.where(direction_labels == d)[0]
        chosen = rng.choice(idx, size=min(max_display_per_class, len(idx)), replace=False)
        mask[chosen] = True

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
        z_disp = z[mask]
        labels_disp = direction_labels[mask]
        colors = [dir_to_color[int(lb)] for lb in labels_disp]
        ax.scatter(z_disp[:, 0], z_disp[:, 1], c=colors, s=12, alpha=0.7, linewidths=0)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        apply_paper_axis_style(ax)
        ax.text(0.02, 0.97, f"T{ax_idx + 1}", transform=ax.transAxes,
                va="top", ha="left", fontsize=TICK_LABEL_SIZE, fontweight="bold")

    # Hide unused axes
    for ax in axes_flat[T:]:
        ax.set_visible(False)

    # Direction legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=dir_to_color[int(d)], label=dir_to_label[int(d)])
        for d in unique_dirs
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.8,
        ncol=1,
        title="Direction",
        title_fontsize=LEGEND_TITLE_SIZE,
    )

    subtitle = _pca_subtitle(pca_k, pca_var)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=TICK_LABEL_SIZE,
                 color="gray")

    fig.suptitle(f"{probe_task} — {rep_name}", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 0.88, 0.96])
    out_path = os.path.join(umap_dir, f"umap_{rep_name}_{probe_task}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"    Saved {out_path}")


def _plot_direction_umap_paper_subset(
    Z_list: List[np.ndarray],
    direction_labels: np.ndarray,
    task_indices: List[int],
    rep_name: str,
    probe_task: str,
    umap_dir: str,
    checkpoint_numbers: List[int],
    pca_k: Optional[int] = None,
    pca_var: Optional[float] = None,
    max_display_per_class: int = 40,
) -> None:
    """One-row paper figure for selected RNN checkpoints, color = direction."""
    selected = []
    for ckpt_num in checkpoint_numbers:
        pos = ckpt_num - 1
        if 0 <= pos < len(Z_list):
            selected.append((pos, ckpt_num))

    if len(selected) != len(checkpoint_numbers):
        print(
            f"    Skipping paper UMAP subset for {probe_task}/{rep_name}: "
            f"requested {checkpoint_numbers}, only {len(Z_list)} checkpoints available"
        )
        return

    unique_dirs = np.unique(direction_labels)
    n_dirs = len(unique_dirs)
    cmap = cm.get_cmap("tab10", max(n_dirs, 10))
    dir_to_color = {int(d): cmap(i) for i, d in enumerate(unique_dirs)}
    dir_angles = np.linspace(0, 360, n_dirs, endpoint=False).astype(int)
    dir_to_label = {int(d): f"{dir_angles[i]}\u00b0" for i, d in enumerate(unique_dirs)}

    rng = np.random.default_rng(42)
    mask = np.zeros(len(direction_labels), dtype=bool)
    for d in unique_dirs:
        idx = np.where(direction_labels == d)[0]
        chosen = rng.choice(idx, size=min(max_display_per_class, len(idx)), replace=False)
        mask[chosen] = True

    all_z = np.concatenate(Z_list, axis=0)
    x_min, x_max = all_z[:, 0].min(), all_z[:, 0].max()
    y_min, y_max = all_z[:, 1].min(), all_z[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (pos, label_num) in zip(axes, selected):
        z_disp = Z_list[pos][mask]
        labels_disp = direction_labels[mask]
        colors = [dir_to_color[int(lb)] for lb in labels_disp]
        ax.scatter(z_disp[:, 0], z_disp[:, 1], c=colors, s=12, alpha=0.75, linewidths=0)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        apply_paper_axis_style(ax)
        ax.text(0.03, 0.96, f"T{label_num}", transform=ax.transAxes,
                va="top", ha="left", fontsize=TICK_LABEL_SIZE, fontweight="bold")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=dir_to_color[int(d)], label=dir_to_label[int(d)])
        for d in unique_dirs
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.8,
        ncol=1,
        title="Direction",
        title_fontsize=LEGEND_TITLE_SIZE,
    )

    subtitle = _pca_subtitle(pca_k, pca_var)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=TICK_LABEL_SIZE,
                 color="gray")

    fig.suptitle(f"{probe_task} — {rep_name}", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 0.80, 0.94])
    out_path = os.path.join(umap_dir, f"umap_{rep_name}_{probe_task}_paper.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"    Saved {out_path}")


# ======================================================================
# Entry point
# ======================================================================

def run_sample_umap(
    exp_dir: str,
    probe_tasks: List[str],
    task_names: List[str],
    output_dir: str,
    n_directions: int = 8,
    n_trials_per_dir: int = 40,
    pca_var_threshold: float = 0.90,
) -> None:
    """Run direction-UMAP analysis for RNN Go-family probe tasks.

    Extracts (or loads cached) controlled-direction representations from
    checkpoints, then runs PCA → UMAP per epoch and plots results.

    Args:
        exp_dir: Experiment directory (contains checkpoints + representations/).
        probe_tasks: Which tasks to analyse.
        task_names: Ordered list of all task names in the experiment.
        output_dir: Directory to save UMAP plots.
        n_directions: Number of uniformly-spaced stimulus directions.
        n_trials_per_dir: Repetitions per direction.
        pca_var_threshold: PCA variance threshold (0 to skip PCA).
    """
    umap_dir = os.path.join(output_dir, "sample_umap")
    os.makedirs(umap_dir, exist_ok=True)

    for probe_task in probe_tasks:
        print(f"  [sample_umap] probe: {probe_task}")

        if probe_task not in _GO_FAMILY:
            print(f"    Skipping '{probe_task}' — UMAP only supports Go-family tasks: {sorted(_GO_FAMILY)}")
            continue

        data = _extract_or_load_umap_reps(
            exp_dir, probe_task, task_names,
            n_directions=n_directions, n_per_dir=n_trials_per_dir,
        )
        if data is None:
            continue

        direction_labels = data['direction_labels']
        epoch_names = data['epoch_names']
        if isinstance(epoch_names, np.ndarray):
            epoch_names = list(epoch_names)
        n_ckpt = int(data['n_checkpoints'])

        # Determine checkpoint indices from stored keys
        # Keys are like 'stpv_after_task_0', 'stim1_after_task_0', ...
        task_indices = sorted({
            int(k.split('_after_task_')[-1])
            for k in data.keys()
            if '_after_task_' in k
        })

        for rep_name in epoch_names:
            print(f"    UMAP for {rep_name}...")
            reps_per_ckpt = [
                data[f"{rep_name}_after_task_{t}"] for t in task_indices
            ]

            Z_list, pca_k, pca_var, eigenvalues, explained_ratio = _pca_umap(
                reps_per_ckpt, pca_var_threshold
            )
            safe_name = f"{rep_name}_{probe_task}"
            _plot_pca_diagnostics(eigenvalues, explained_ratio, pca_k, safe_name, umap_dir)

            _plot_direction_umap(
                Z_list, direction_labels, task_indices,
                rep_name, probe_task, umap_dir,
                pca_k=pca_k, pca_var=pca_var,
            )
            _plot_direction_umap_paper_subset(
                Z_list, direction_labels, task_indices,
                rep_name, probe_task, umap_dir,
                checkpoint_numbers=[1, 6, 12, 18],
                pca_k=pca_k, pca_var=pca_var,
            )

    print(f"  [sample_umap] Results saved to {umap_dir}")
