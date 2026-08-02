#!/usr/bin/env python
"""Aggregate RNN Experiment B runs across seeds and anchor strengths."""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Liberation Sans"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_accuracy_matrix(run_dir):
    perf = load_json(os.path.join(run_dir, "performance_history.json"))
    if not perf:
        return None
    task_names = list(perf)
    n_stages = max(len(perf[name]) for name in task_names)
    matrix = np.full((len(task_names), n_stages), np.nan)
    for row, task_name in enumerate(task_names):
        for column, entry in enumerate(perf[task_name]):
            if entry is not None and entry.get("accuracy") is not None:
                matrix[row, column] = float(entry["accuracy"])
    return matrix


def mean_matrices(matrices):
    rows = max(matrix.shape[0] for matrix in matrices)
    columns = max(matrix.shape[1] for matrix in matrices)
    stacked = np.full((len(matrices), rows, columns), np.nan)
    for index, matrix in enumerate(matrices):
        stacked[index, :matrix.shape[0], :matrix.shape[1]] = matrix
    return np.nanmean(stacked, axis=0)


def final_drift(run_dir, probe_task):
    metrics = load_json(os.path.join(
        run_dir, "drift_analysis", "reference_drift_metrics.json"
    )) or []
    records = [item for item in metrics if item.get("probe_task") == probe_task]
    if not records:
        return float("nan")
    final = max(records, key=lambda item: item["target_task"])
    return 1.0 - float(final["cosine_sim_mean"])


def finite_mean(values):
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else float("nan")


def collect_run(run_dir):
    config = load_json(os.path.join(run_dir, "experiment_config.json"))
    matrix = load_accuracy_matrix(run_dir)
    if not config or matrix is None:
        return None
    method = config.get("method")
    if method not in ("replay", "anchored_replay"):
        return None

    anchor_lambda = float(config.get("anchor_lambda", 0.0) or 0.0)
    diagonal = np.diag(matrix[:min(matrix.shape), :min(matrix.shape)])
    final_column = matrix[:, -1]
    task_names = config.get("tasks") or []
    probe_task = task_names[0] if task_names else next(iter(
        load_json(os.path.join(run_dir, "performance_history.json"))
    ))
    return {
        "run": os.path.basename(run_dir.rstrip("/")),
        "method": method,
        "anchor_loss": config.get("anchor_loss", "mse") if anchor_lambda > 0 else "-",
        "anchor_lambda": anchor_lambda,
        "seed": config.get("seed", -1),
        "n_tasks": matrix.shape[0],
        "retained_acc": finite_mean(final_column) * 100.0,
        "task1_acc": float(matrix[0, -1]) * 100.0,
        "forward_acc": finite_mean(diagonal) * 100.0,
        "final_drift": final_drift(run_dir, probe_task),
    }


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["anchor_loss"], row["anchor_lambda"])].append(row)

    result = []
    for (loss, anchor_lambda), group in sorted(
        groups.items(), key=lambda item: item[0][1]
    ):
        entry = {
            "anchor_loss": loss,
            "anchor_lambda": anchor_lambda,
            "n_seeds": len(group),
        }
        for metric in ("retained_acc", "task1_acc", "forward_acc", "final_drift"):
            values = [row[metric] for row in group if math.isfinite(row[metric])]
            if values:
                mean = float(np.mean(values))
                ci = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            else:
                mean = ci = float("nan")
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_ci"] = ci
        result.append(entry)
    return result


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def lambda_label(value):
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def plot_accuracy_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_box_aspect(1)
    n_tasks = matrix.shape[0]
    ticks = list(range(n_tasks)) if n_tasks <= 6 else [0, (n_tasks - 1) // 2, n_tasks - 1]
    ax.set_xticks(ticks, [str(value + 1) for value in ticks])
    ax.set_yticks(ticks, [str(value + 1) for value in ticks])
    ax.set_xlabel("After Training on Task")
    ax.set_ylabel("Evaluated Task")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_matrix_curves(lambda_matrices, output_dir):
    lambdas = sorted(lambda_matrices)
    colors = plt.colormaps["Blues"](np.linspace(0.35, 0.95, len(lambdas)))
    specifications = (
        ("accuracy_diagonal_by_lambda.pdf", "Task", "Forward Accuracy", lambda matrix: np.diag(matrix)),
        ("accuracy_first_row_by_lambda.pdf", "After Training on Task", "Task-1 Accuracy", lambda matrix: matrix[0]),
    )
    for filename, xlabel, ylabel, extractor in specifications:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for color, anchor_lambda in zip(colors, lambdas):
            values = extractor(lambda_matrices[anchor_lambda])
            ax.plot(
                np.arange(1, len(values) + 1), values, marker="o", markersize=4,
                linewidth=1.5, color=color, label=f"{anchor_lambda:g}",
            )
        style_axis(ax, xlabel, ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="lambda", frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def make_accuracy_report(run_dirs, output_dir):
    groups = defaultdict(list)
    for run_dir in run_dirs:
        config = load_json(os.path.join(run_dir, "experiment_config.json")) or {}
        if config.get("method") not in ("replay", "anchored_replay"):
            continue
        matrix = load_accuracy_matrix(run_dir)
        if matrix is not None:
            groups[float(config.get("anchor_lambda", 0.0) or 0.0)].append(matrix)
    if not groups:
        return

    matrix_dir = os.path.join(output_dir, "accuracy_matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    means = {}
    for anchor_lambda in sorted(groups):
        matrix = mean_matrices(groups[anchor_lambda])
        means[anchor_lambda] = matrix
        group_dir = os.path.join(matrix_dir, f"lambda_{lambda_label(anchor_lambda)}")
        os.makedirs(group_dir, exist_ok=True)
        np.savetxt(os.path.join(group_dir, "accuracy_matrix.csv"), matrix, delimiter=",", fmt="%.8g")
        plot_accuracy_matrix(
            matrix, os.path.join(group_dir, "accuracy_matrix.pdf"),
        )
    plot_matrix_curves(means, matrix_dir)


def make_focus_plots(summary, output_dir):
    baseline = next((row for row in summary if row["anchor_lambda"] == 0.0), None)
    positive = [row for row in summary if row["anchor_lambda"] > 0.0]
    if positive:
        fig, ax = plt.subplots(figsize=(6.6, 4.25))
        series = (
            ("task1_acc", "Task-1 accuracy", "#1f77b4"),
            ("forward_acc", "Forward accuracy", "#d62728"),
        )
        for metric, label, color in series:
            if baseline and math.isfinite(baseline[f"{metric}_mean"]):
                mean = baseline[f"{metric}_mean"]
                ci = baseline[f"{metric}_ci"]
                ax.axhspan(mean - ci, mean + ci, color=color, alpha=0.10)
                ax.axhline(mean, color=color, linestyle="--", linewidth=1.3,
                           label=f"Replay {label.lower()} (lambda=0)")
            ax.errorbar(
                [row["anchor_lambda"] for row in positive],
                [row[f"{metric}_mean"] for row in positive],
                yerr=[row[f"{metric}_ci"] for row in positive],
                color=color, marker="o", linestyle="none", capsize=3,
                label=f"Anchored {label.lower()}",
            )
        ax.set_xscale("log")
        style_axis(ax, "Anchor strength lambda (log scale)", "Accuracy (%)")
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "task1_fwd_acc_vs_lambda.pdf"), bbox_inches="tight")
        plt.close(fig)

    drift_rows = [row for row in summary if math.isfinite(row["final_drift_mean"])]
    if drift_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.25))
        anchored = [row for row in drift_rows if row["anchor_lambda"] > 0]
        ax.errorbar(
            [row["final_drift_mean"] for row in anchored],
            [row["forward_acc_mean"] for row in anchored],
            yerr=[row["forward_acc_ci"] for row in anchored],
            color="#1f77b4", marker="o", linestyle="none", capsize=3,
            label="Replay + state anchoring",
        )
        if baseline and math.isfinite(baseline["final_drift_mean"]):
            ax.errorbar(
                baseline["final_drift_mean"], baseline["forward_acc_mean"],
                yerr=baseline["forward_acc_ci"], color="0.45", marker="D",
                linestyle="none", capsize=3, label="Replay baseline (lambda=0)",
            )
        style_axis(ax, "Final STPV drift  1 - <cos>", "Forward Accuracy (%)")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fwd_acc_vs_final_drift.pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="RNN Experiment B comparison report")
    parser.add_argument("--exp_root", default="experiments")
    parser.add_argument("--glob", default="exp*b_rnn_*")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    run_dirs = sorted(
        path for path in glob.glob(os.path.join(args.exp_root, args.glob))
        if os.path.isdir(path)
    )
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.glob}")
    rows = [row for row in (collect_run(path) for path in run_dirs) if row is not None]
    if not rows:
        raise SystemExit("No completed replay/anchored_replay runs found")

    output_dir = args.out_dir or os.path.join(args.exp_root, "experiment_b_report")
    os.makedirs(output_dir, exist_ok=True)
    summary = aggregate_rows(rows)
    write_csv(os.path.join(output_dir, "experiment_b_per_run.csv"), rows)
    write_csv(os.path.join(output_dir, "experiment_b_aggregated.csv"), summary)
    make_accuracy_report(run_dirs, output_dir)
    make_focus_plots(summary, output_dir)
    print(f"RNN Experiment B report saved to {output_dir}")


if __name__ == "__main__":
    main()
