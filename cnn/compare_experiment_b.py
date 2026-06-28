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


def make_plots(agg: List[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    agg_sorted = sorted(agg, key=lambda e: e["anchor_lambda"])
    lambdas = [e["anchor_lambda"] for e in agg_sorted]
    drifts = [e["final_drift_mean"] for e in agg_sorted]

    downstream = [
        ("plasticity_best_val_acc", "Forward-transfer best val acc (%)"),
        ("participation_ratio", "Participation ratio"),
        ("dead_unit_fraction", "Dead-unit fraction"),
        ("subspace_overlap", "Cross-task subspace overlap"),
        ("retained_acc", "Retained accuracy (%)"),
    ]

    # vs lambda
    fig, axes = plt.subplots(1, len(downstream), figsize=(5 * len(downstream), 5))
    for ax, (key, label) in zip(axes, downstream):
        ys = [e[f"{key}_mean"] for e in agg_sorted]
        es = [e[f"{key}_ci"] for e in agg_sorted]
        x = [max(l, 1e-3) for l in lambdas]  # 0 -> small for log axis
        ax.errorbar(x, ys, yerr=es, marker="o", capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("anchor lambda (0 shown as 1e-3)")
        ax.set_ylabel(label)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "benefit_vs_lambda.pdf")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Plot saved to {p1}")

    # vs achieved drift
    fig, axes = plt.subplots(1, len(downstream) - 1, figsize=(5 * (len(downstream) - 1), 5))
    for ax, (key, label) in zip(axes, downstream[:-1]):
        ys = [e[f"{key}_mean"] for e in agg_sorted]
        es = [e[f"{key}_ci"] for e in agg_sorted]
        ax.errorbar(drifts, ys, yerr=es, marker="o", capsize=4, linestyle="none")
        ax.set_xlabel("Final reference drift  1 - <cos>")
        ax.set_ylabel(label)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "benefit_vs_drift.pdf")
    fig.savefig(p2)
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


if __name__ == "__main__":
    main()
