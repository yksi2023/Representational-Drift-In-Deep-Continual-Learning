#!/usr/bin/env python
"""Export the numerical arrays used by a CNN aggregate report's plots."""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CNN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CNN_DIR))
import aggregate_seeds as aggregate


def write_rows(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the aggregated numbers behind CNN report plots."
    )
    parser.add_argument("--exp_root", default=str(CNN_DIR / "experiments"))
    parser.add_argument("--prefix", default="exp1_cnn_")
    parser.add_argument("--methods", default="normal,replay,ewc,lwf,gpm")
    parser.add_argument("--layers", default="layer1,layer2,layer3,layer4")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def export_gap_drift(seed_dirs, layers, method_dir):
    values = defaultdict(list)
    for seed_dir in seed_dirs:
        result = aggregate.load_gap_drift(seed_dir, layers) or {}
        for layer, (gaps, means) in result.items():
            for gap, mean in zip(gaps, means):
                values[(layer, gap)].append(mean)

    rows = [
        {
            "layer": layer,
            "task_gap": gap,
            "n_seeds": len(vals),
            "sample_pv_pearson_mean": float(np.mean(vals)),
            "sample_pv_pearson_std": float(np.std(vals)),
        }
        for (layer, gap), vals in sorted(values.items())
    ]
    write_rows(
        os.path.join(method_dir, "gap_drift_sample_pv.csv"),
        ["layer", "task_gap", "n_seeds", "sample_pv_pearson_mean", "sample_pv_pearson_std"],
        rows,
    )


def main():
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    output_dir = args.output_dir or os.path.join(
        args.exp_root, "exp1_aggregate_report", "plot_data"
    )

    for method in methods:
        seed_dirs = aggregate.discover_seed_dirs(args.exp_root, args.prefix, method)
        if not seed_dirs:
            print(f"[{method}] no seed directories found")
            continue

        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        matrices = [aggregate.load_accuracy_matrix(path) for path in seed_dirs]
        matrices = [matrix for matrix in matrices if matrix is not None]
        if matrices:
            np.savetxt(
                os.path.join(method_dir, "accuracy_matrix.csv"),
                np.nanmean(np.stack(matrices), axis=0), delimiter=",", fmt="%.8g",
            )

        for layer in layers:
            matrices = [aggregate.load_similarity_matrix(path, layer) for path in seed_dirs]
            matrices = [matrix for matrix in matrices if matrix is not None]
            if matrices:
                np.savetxt(
                    os.path.join(method_dir, f"similarity_matrix_{layer}.csv"),
                    np.nanmean(np.stack(matrices), axis=0), delimiter=",", fmt="%.8g",
                )

        export_gap_drift(seed_dirs, layers, method_dir)
        print(f"[{method}] exported plot data to {method_dir}")


if __name__ == "__main__":
    main()
