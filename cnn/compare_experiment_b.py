#!/usr/bin/env python
"""Experiment B full report: tabulate every arm (lambda x seed) -- no verdict.

Loads the replay / anchored_replay runs under an experiments root and emits one
combined table (CSV + printed) plus plots of each downstream outcome measure against the
anchoring strength and against the achieved drift. Retained accuracy is shown as
its own column so the reader can see which lambda arms are fairly comparable to
the lambda=0 arm (the matched-accuracy control of Sec. 5.1).

This script deliberately does NOT pick a "matched" arm or declare drift
functional/incidental -- that judgment is left to the user eyeballing the table.

Inputs read per run dir (when present):
  experiment_config.json          -> method, anchor_lambda, anchor_loss, seed, anchor_layers
  comprehensive_evaluation.json    -> overall.mean_accuracy  (retained accuracy, %)
  plasticity_metrics.json          -> per-task best_val_acc / best_val_loss
  drift_analysis/metrics.json      -> final reference drift  1 - cosine_sim_mean
  drift_analysis/health_metrics.json        -> participation ratio, dead-unit fraction
  drift_analysis/subspace_overlap_metrics.json -> mean cross-task overlap
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'Liberation Sans'

ANCHOR_LAYERS_DEFAULT = ["layer3", "layer4"]


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _final_drift(metrics: list, layers: List[str]) -> float:
    """Mean over `layers` of (1 - cosine_sim_mean) at the largest target task."""
    by_layer: Dict[str, list] = defaultdict(list)
    for r in metrics:
        by_layer[r["layer"]].append(r)
    drifts = []
    for ln in layers:
        if ln not in by_layer:
            continue
        rec = max(by_layer[ln], key=lambda r: r["target_task"])
        drifts.append(1.0 - rec["cosine_sim_mean"])
    return _mean(drifts)


def _final_health(health: dict, layers: List[str]) -> Dict[str, float]:
    """Final-checkpoint participation ratio + dead-unit fraction averaged over layers."""
    prs, deads = [], []
    layer_block = health.get("layers", {})
    for ln in layers:
        if ln not in layer_block:
            continue
        prs.append(layer_block[ln]["participation_ratio"][-1])
        deads.append(layer_block[ln]["dead_unit_fraction"][-1])
    return {"participation_ratio": _mean(prs), "dead_unit_fraction": _mean(deads)}


def _mean_overlap(overlap: dict, layers: List[str]) -> float:
    layer_block = overlap.get("layers", {})
    vals = [layer_block[ln]["mean_overlap"] for ln in layers if ln in layer_block]
    return _mean(vals)


def _safe_lambda_label(lam: float) -> str:
    label = f"{lam:.6g}"
    return label.replace("-", "m").replace("+", "").replace(".", "p")


def _load_accuracy_matrix(run_dir: str) -> Optional[np.ndarray]:
    """Load task x training-stage accuracy matrix from performance_history.json."""
    perf = _load_json(os.path.join(run_dir, "performance_history.json"))
    if not perf:
        return None

    raw_task_names = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
    if not raw_task_names:
        return None

    num_tasks = len(raw_task_names)
    num_stages = max(len(perf[n]) for n in raw_task_names)
    matrix = np.full((num_tasks, num_stages), np.nan)
    for i, name in enumerate(raw_task_names):
        for j, entry in enumerate(perf[name]):
            if entry is None:
                continue
            acc = entry.get("accuracy")
            if acc is not None:
                matrix[i, j] = acc
    return matrix


def _mean_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """Average matrices, padding with NaN if runs have different task counts."""
    rows = max(m.shape[0] for m in matrices)
    cols = max(m.shape[1] for m in matrices)
    stacked = np.full((len(matrices), rows, cols), np.nan)
    for i, matrix in enumerate(matrices):
        stacked[i, :matrix.shape[0], :matrix.shape[1]] = matrix
    valid = np.isfinite(stacked)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    counts = valid.sum(axis=0)
    mean = np.full((rows, cols), np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean


def collect_run(run_dir: str) -> Optional[dict]:
    cfg = _load_json(os.path.join(run_dir, "experiment_config.json"))
    if cfg is None:
        return None

    method = cfg.get("method", "?")
    # Experiment B only compares the two replay arms; skip other CL methods
    # (normal/ewc/lwf/gpm) that may share the exp<i>_cnn_* directory prefix.
    if method not in ("replay", "anchored_replay"):
        return None
    anchor_lambda = float(cfg.get("anchor_lambda", 0.0) or 0.0)
    anchor_loss = cfg.get("anchor_loss", "-")
    seed = cfg.get("seed", -1)
    layers_cfg = cfg.get("anchor_layers") or ""
    layers = [s.strip() for s in layers_cfg.split(",") if s.strip()] or ANCHOR_LAYERS_DEFAULT

    row: Dict[str, object] = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "method": method,
        "anchor_loss": anchor_loss if anchor_lambda > 0 else "-",
        "anchor_lambda": anchor_lambda,
        "seed": seed,
        "retained_acc": float("nan"),
        "first_task_acc": float("nan"),
        "final_drift": float("nan"),
        "plasticity_best_val_acc": float("nan"),
        "participation_ratio": float("nan"),
        "dead_unit_fraction": float("nan"),
        "subspace_overlap": float("nan"),
    }

    comp = _load_json(os.path.join(run_dir, "comprehensive_evaluation.json"))
    if comp and "overall" in comp:
        row["retained_acc"] = comp["overall"].get("mean_accuracy", float("nan"))

    plast = _load_json(os.path.join(run_dir, "plasticity_metrics.json"))
    if plast:
        row["plasticity_best_val_acc"] = _mean([e.get("best_val_acc") for e in plast])

    # Fallback: derive ret_acc / fwd_acc from performance_history.json
    perf = _load_json(os.path.join(run_dir, "performance_history.json"))
    if perf:
        tasks_sorted = sorted(perf.keys(), key=lambda k: int(k.split("_")[1]))
        n_stages = len(perf[tasks_sorted[0]]) if tasks_sorted else 0
        if n_stages > 0:
            if math.isnan(row["retained_acc"]):
                # mean accuracy of all tasks evaluated after final training stage
                final_accs = [perf[t][-1]["accuracy"] * 100.0 for t in tasks_sorted
                              if len(perf[t]) >= n_stages]
                row["retained_acc"] = _mean(final_accs)
            # First task accuracy at final checkpoint
            if "task_1" in perf and perf["task_1"]:
                row["first_task_acc"] = perf["task_1"][-1]["accuracy"] * 100.0
            if math.isnan(row["plasticity_best_val_acc"]):
                # diagonal: accuracy on task_k right after training on task_k
                diag = []
                for i, t in enumerate(tasks_sorted):
                    if i < len(perf[t]):
                        diag.append(perf[t][i]["accuracy"] * 100.0)
                row["plasticity_best_val_acc"] = _mean(diag)

    drift_dir = os.path.join(run_dir, "drift_analysis")
    metrics = _load_json(os.path.join(drift_dir, "metrics.json"))
    if metrics:
        row["final_drift"] = _final_drift(metrics, layers)
    health = _load_json(os.path.join(drift_dir, "health_metrics.json"))
    if health:
        fh = _final_health(health, layers)
        row["participation_ratio"] = fh["participation_ratio"]
        row["dead_unit_fraction"] = fh["dead_unit_fraction"]
    overlap = _load_json(os.path.join(drift_dir, "subspace_overlap_metrics.json"))
    if overlap:
        row["subspace_overlap"] = _mean_overlap(overlap, layers)

    return row


def aggregate(rows: List[dict]) -> List[dict]:
    """Mean +/- 95% CI across seeds, grouped by (anchor_loss, anchor_lambda)."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["anchor_loss"], r["anchor_lambda"])].append(r)

    metric_keys = [
        "retained_acc", "first_task_acc", "final_drift", "plasticity_best_val_acc",
        "participation_ratio", "dead_unit_fraction", "subspace_overlap",
    ]
    agg = []
    for (loss, lam), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        entry = {"anchor_loss": loss, "anchor_lambda": lam, "n_seeds": len(grp)}
        for k in metric_keys:
            vals = [g[k] for g in grp if not (isinstance(g[k], float) and math.isnan(g[k]))]
            if vals:
                m = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                    ci = 1.96 * math.sqrt(var) / math.sqrt(len(vals))
                else:
                    ci = 0.0
                entry[f"{k}_mean"] = m
                entry[f"{k}_ci"] = ci
            else:
                entry[f"{k}_mean"] = float("nan")
                entry[f"{k}_ci"] = float("nan")
        agg.append(entry)
    return agg


def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-run table saved to {path}")


def print_table(agg: List[dict]) -> None:
    cols = [
        ("anchor_loss", "loss", "{}"),
        ("anchor_lambda", "lambda", "{:.3g}"),
        ("n_seeds", "n", "{}"),
        ("retained_acc_mean", "ret_acc%", "{:.2f}"),
        ("first_task_acc_mean", "t1_acc%", "{:.2f}"),
        ("final_drift_mean", "drift", "{:.3f}"),
        ("plasticity_best_val_acc_mean", "fwd_acc%", "{:.2f}"),
        ("participation_ratio_mean", "PR", "{:.1f}"),
        ("dead_unit_fraction_mean", "dead", "{:.3f}"),
        ("subspace_overlap_mean", "overlap", "{:.3f}"),
    ]
    header = " | ".join(f"{h:>9}" for _, h, _ in cols)
    print("\n" + "=" * len(header))
    print("EXPERIMENT B REPORT (mean across seeds; +/-95% CI in CSV)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for e in agg:
        cells = []
        for key, _, fmt in cols:
            v = e.get(key)
            try:
                cells.append(f"{fmt.format(v):>9}")
            except (ValueError, TypeError):
                cells.append(f"{str(v):>9}")
        print(" | ".join(cells))
    print("=" * len(header))
    print("Note: retained_acc is the matched-accuracy control. Compare downstream")
    print("columns (fwd_acc, PR, dead, overlap) only across arms with similar ret_acc.\n")


def _style_ax(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    """Apply consistent styling to an axis."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_accuracy_matrix(
    matrix: np.ndarray,
    output_path: str,
) -> None:
    n_tasks, n_stages = matrix.shape
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    tick_positions = list(range(n_stages))
    tick_labels = [str(i + 1) for i in range(n_stages)]
    if n_stages > 6:
        mid = (n_stages - 1) // 2
        tick_positions = [0, mid, n_stages - 1]
        tick_labels = ["1", str(mid + 1), str(n_stages)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    row_ticks = [t for t in tick_positions if t < n_tasks]
    ax.set_yticks(row_ticks)
    ax.set_yticklabels([str(t + 1) for t in row_ticks])
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def _plot_lambda_curves(
    lambda_matrices: Dict[float, np.ndarray],
    output_dir: str,
) -> None:
    lambdas = sorted(lambda_matrices)
    if not lambdas:
        return

    cmap = plt.colormaps.get_cmap("Blues")
    color_values = np.linspace(0.35, 0.95, len(lambdas))
    colors = {lam: cmap(color_values[i]) for i, lam in enumerate(lambdas)}

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for lam in lambdas:
        matrix = lambda_matrices[lam]
        diag_len = min(matrix.shape)
        xs = np.arange(1, diag_len + 1)
        ys = np.diag(matrix[:diag_len, :diag_len])
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5,
                color=colors[lam], label=f"{lam:.6g}")
    _style_ax(ax, "Task", "Accuracy", "Accuracy Matrix Diagonal")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="lambda", fontsize=9, title_fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_diagonal_by_lambda.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Plot saved to {path}")

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for lam in lambdas:
        matrix = lambda_matrices[lam]
        xs = np.arange(1, matrix.shape[1] + 1)
        ys = matrix[0, :]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5,
                color=colors[lam], label=f"{lam:.6g}")
    _style_ax(ax, "After Training on Task", "Task-1 Accuracy", "First Row of Accuracy Matrix")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="lambda", fontsize=9, title_fontsize=10)
    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_first_row_by_lambda.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Plot saved to {path}")


def make_accuracy_matrix_report(run_dirs: List[str], out_dir: str) -> None:
    """Average accuracy matrices by lambda and plot lambda-wise summaries."""
    matrix_groups: Dict[float, List[np.ndarray]] = defaultdict(list)
    skipped = 0
    for run_dir in run_dirs:
        cfg = _load_json(os.path.join(run_dir, "experiment_config.json"))
        if not cfg:
            skipped += 1
            continue
        if cfg.get("method") not in ("replay", "anchored_replay"):
            skipped += 1
            continue
        matrix = _load_accuracy_matrix(run_dir)
        if matrix is None:
            skipped += 1
            continue
        lam = float(cfg.get("anchor_lambda", 0.0) or 0.0)
        matrix_groups[lam].append(matrix)

    if not matrix_groups:
        print("No performance_history.json files found; skipping accuracy matrix report.")
        return

    matrix_dir = os.path.join(out_dir, "accuracy_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    lambda_matrices: Dict[float, np.ndarray] = {}
    for lam in sorted(matrix_groups):
        mean_matrix = _mean_matrices(matrix_groups[lam])
        lambda_matrices[lam] = mean_matrix
        group_dir = os.path.join(matrix_dir, f"lambda_{_safe_lambda_label(lam)}")
        os.makedirs(group_dir, exist_ok=True)
        np.savetxt(
            os.path.join(group_dir, "accuracy_matrix.csv"),
            mean_matrix,
            delimiter=",",
            fmt="%.8g",
        )
        _plot_accuracy_matrix(
            mean_matrix,
            os.path.join(group_dir, "accuracy_matrix.pdf"),
        )

    _plot_lambda_curves(lambda_matrices, matrix_dir)
    print(
        f"Accuracy matrix report saved to {matrix_dir} "
        f"({sum(len(v) for v in matrix_groups.values())} run(s), {skipped} skipped)."
    )


def make_focus_plots(agg: List[dict], out_dir: str) -> None:
    """Create compact Experiment-B figures with replay as a visual reference."""
    rows = sorted(agg, key=lambda entry: entry["anchor_lambda"])
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)

    baseline = next((entry for entry in rows if entry["anchor_lambda"] == 0.0), None)
    anchored = [entry for entry in rows if entry["anchor_lambda"] > 0.0]
    lambda_rows = [entry for entry in anchored if 0.3 <= entry["anchor_lambda"] <= 1000]
    if not anchored:
        print("No positive anchor lambdas found; skipping focused Experiment-B figures.")
        return

    lambdas = [entry["anchor_lambda"] for entry in lambda_rows]
    task1_color = "#1f77b4"
    forward_color = "#d62728"
    baseline_color = "0.45"

    fig, ax = plt.subplots(figsize=(6.6, 4.25))
    series = (
        ("first_task_acc", "Task-1 accuracy", task1_color),
        ("plasticity_best_val_acc", "Forward accuracy", forward_color),
    )
    for key, label, color in series:
        if baseline is not None and math.isfinite(baseline[f"{key}_mean"]):
            mean = baseline[f"{key}_mean"]
            ci = baseline[f"{key}_ci"]
            ax.axhspan(mean - ci, mean + ci, color=color, alpha=0.10, zorder=0)
            ax.axhline(
                mean, color=color, linestyle="--", linewidth=1.35, alpha=0.8,
                label=f"Replay {label.lower()} ($\\lambda=0$)",
            )
        ys = [entry[f"{key}_mean"] for entry in lambda_rows]
        errors = [entry[f"{key}_ci"] for entry in lambda_rows]
        ax.errorbar(
            lambdas, ys, yerr=errors, label=f"Anchored {label.lower()}",
            color=color, marker="o", markersize=5.8, linestyle="none",
            capsize=3, elinewidth=1.1, zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Anchor strength $\lambda$ (log scale)", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    ax.margins(y=0.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "task1_fwd_acc_vs_lambda.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Plot saved to {path}")

    drift_rows = [entry for entry in anchored if math.isfinite(entry["final_drift_mean"])]
    if not drift_rows:
        print("No final-drift values found; skipping forward-accuracy vs drift figure.")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.25))
    drift_x = [entry["final_drift_mean"] for entry in drift_rows]
    fwd_y = [entry["plasticity_best_val_acc_mean"] for entry in drift_rows]
    fwd_yerr = [entry["plasticity_best_val_acc_ci"] for entry in drift_rows]
    ax.errorbar(
        drift_x, fwd_y, yerr=fwd_yerr,
        color=task1_color, marker="o", markersize=5.8, linestyle="none",
        capsize=3, elinewidth=1.1, label="Replay + representation anchoring", zorder=3,
    )
    if baseline is not None and math.isfinite(baseline["final_drift_mean"]):
        ax.errorbar(
            baseline["final_drift_mean"], baseline["plasticity_best_val_acc_mean"],
            yerr=baseline["plasticity_best_val_acc_ci"],
            color=baseline_color, marker="D", markersize=5.6, linestyle="none",
            capsize=3, elinewidth=1.1, label=r"Replay baseline ($\lambda=0$)", zorder=4,
        )
    ax.set_xlabel(r"Final drift $1 - \langle\cos\rangle$", fontsize=14)
    ax.set_ylabel("Forward accuracy (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.margins(x=0.12, y=0.16)
    fig.tight_layout()
    path = os.path.join(out_dir, "fwd_acc_vs_final_drift.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Plot saved to {path}")


def make_plots(agg: List[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    agg_sorted = sorted(agg, key=lambda e: e["anchor_lambda"])
    lambdas = [e["anchor_lambda"] for e in agg_sorted]
    drifts = [e["final_drift_mean"] for e in agg_sorted]

    downstream = [
        ("retained_acc", "Retained acc (%)"),
        ("first_task_acc", "Task-1 acc (%)"),
        ("plasticity_best_val_acc", "Forward-transfer acc (%)"),
        ("participation_ratio", "Participation ratio"),
        ("dead_unit_fraction", "Dead-unit fraction"),
        ("subspace_overlap", "Subspace overlap"),
    ]

    colors = plt.colormaps.get_cmap("tab10")

    # ── benefit vs lambda ──
    n = len(downstream)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes_flat = axes.flatten()
    for idx, (ax, (key, label)) in enumerate(zip(axes_flat, downstream)):
        ys = [e[f"{key}_mean"] for e in agg_sorted]
        es = [e[f"{key}_ci"] for e in agg_sorted]
        x = [max(l, 1e-3) for l in lambdas]
        ax.errorbar(x, ys, yerr=es, marker="o", capsize=4, color=colors(idx),
                    linewidth=1.8, markersize=6, elinewidth=1.2)
        ax.set_xscale("log")
        _style_ax(ax, r"Anchor $\lambda$", label)
    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Metrics vs Anchoring Strength", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(out_dir, "benefit_vs_lambda.pdf")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {p1}")

    # ── benefit vs achieved drift ──
    drift_metrics = [m for m in downstream if m[0] != "retained_acc"]
    nd = len(drift_metrics)
    fig, axes = plt.subplots(2, (nd + 1) // 2, figsize=(5 * ((nd + 1) // 2), 9))
    axes_flat = axes.flatten()
    for idx, (ax, (key, label)) in enumerate(zip(axes_flat, drift_metrics)):
        ys = [e[f"{key}_mean"] for e in agg_sorted]
        es = [e[f"{key}_ci"] for e in agg_sorted]
        ax.errorbar(drifts, ys, yerr=es, marker="o", capsize=4, linestyle="none",
                    color=colors(idx + 1), markersize=7, elinewidth=1.2)
        _style_ax(ax, r"Final drift  $1 - \langle\cos\rangle$", label)
    for ax in axes_flat[nd:]:
        ax.set_visible(False)
    fig.suptitle("Metrics vs Achieved Drift", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = os.path.join(out_dir, "benefit_vs_drift.pdf")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {p2}")


def main():
    ap = argparse.ArgumentParser(description="Experiment B full report (no verdict)")
    ap.add_argument("--exp_root", type=str, default="experiments",
                    help="Directory containing the exp<i>_cnn_* run directories")
    ap.add_argument("--glob", type=str, default="exp*_cnn_*",
                    help="Glob (relative to exp_root) matching run directories. "
                         "Non-replay methods are skipped automatically.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for report files (default: <exp_root>/experiment_b_report)")
    args = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.exp_root, args.glob)) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"No run dirs matched {os.path.join(args.exp_root, args.glob)}")

    rows = [r for r in (collect_run(d) for d in run_dirs) if r is not None]
    if not rows:
        raise SystemExit("No runs with experiment_config.json found.")

    out_dir = args.out_dir or os.path.join(args.exp_root, "experiment_b_report")
    os.makedirs(out_dir, exist_ok=True)

    write_csv(rows, os.path.join(out_dir, "experiment_b_per_run.csv"))
    agg = aggregate(rows)
    write_csv(agg, os.path.join(out_dir, "experiment_b_aggregated.csv"))
    print_table(agg)
    make_plots(agg, out_dir)
    make_focus_plots(agg, out_dir)
    make_accuracy_matrix_report(run_dirs, out_dir)


if __name__ == "__main__":
    main()
