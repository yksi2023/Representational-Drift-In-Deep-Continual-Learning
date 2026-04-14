"""Analyze representational drift across RNN continual learning checkpoints.

This script provides comprehensive drift analysis including:
1. Baseline drift metrics (cosine similarity, L2 distance over tasks)
2. Model pairwise cosine similarity matrices
3. Sample-wise similarity matrices
4. Temporal hidden state similarity matrices
5. Performance plots (from saved performance_history.json)

Representations are loaded from pre-saved .npz files (generated during training),
so no GPU re-extraction is needed.

Usage:
    python analyze_drift.py --exp_dir experiments/exp2_rnn_normal
    python analyze_drift.py --exp_dir experiments/exp2_rnn_ewc --probe_tasks fdgo reactgo
"""
import argparse
import json
import os
import sys

from src.analysis import (
    run_baseline_drift,
    run_model_similarity,
    run_sample_similarity,
    run_temporal_similarity,
    plot_rnn_performance,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze representational drift in RNN sequential learning")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (contains performance_history.json and representations/)")
    parser.add_argument("--probe_tasks", type=str, nargs="+", default=None,
                        help="Task names to probe for drift analysis. Default: first task in the sequence.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save analysis results (defaults to exp_dir/drift_analysis)")
    parser.add_argument("--neuron_ratio", type=float, default=1.0,
                        help="Ratio of neurons to randomly sample for drift analysis (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for neuron sampling reproducibility")
    parser.add_argument("--skip_sample_sim", action="store_true",
                        help="Skip sample similarity matrices (can be slow for many tasks)")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="RNN hidden size (needed to reshape flat reps for temporal analysis)")
    return parser.parse_args()


def load_task_names(exp_dir: str):
    """Load ordered task names from experiment config or performance history."""
    # Try experiment_config.json first (has the original task order)
    config_path = os.path.join(exp_dir, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "tasks" in config:
            return config["tasks"]

    # Fallback: infer from performance_history.json key order
    perf_path = os.path.join(exp_dir, "performance_history.json")
    if os.path.exists(perf_path):
        with open(perf_path, "r", encoding="utf-8") as f:
            perf = json.load(f)
        return list(perf.keys())

    raise FileNotFoundError(
        f"Cannot determine task names. Neither experiment_config.json nor "
        f"performance_history.json found in {exp_dir}"
    )


def main():
    args = parse_args()

    # Validate experiment directory
    if not os.path.isdir(args.exp_dir):
        print(f"Error: experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    reps_dir = os.path.join(args.exp_dir, "representations")
    if not os.path.isdir(reps_dir):
        print(f"Error: representations/ not found in {args.exp_dir}")
        sys.exit(1)

    # Validate neuron_ratio
    if not 0.0 < args.neuron_ratio <= 1.0:
        print("Error: --neuron_ratio must be in range (0.0, 1.0]")
        sys.exit(1)

    # Load task names
    task_names = load_task_names(args.exp_dir)
    print(f"Task sequence ({len(task_names)}): {task_names}")

    # Determine probe tasks
    if args.probe_tasks is None:
        # Default: first task (most interesting for forgetting analysis)
        args.probe_tasks = [task_names[0]]
    for pt in args.probe_tasks:
        npz_path = os.path.join(reps_dir, f"{pt}.npz")
        if not os.path.exists(npz_path):
            print(f"Error: probe task '{pt}' not found in {reps_dir}")
            sys.exit(1)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "drift_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RNN DRIFT ANALYSIS")
    print(f"Experiment dir: {args.exp_dir}")
    print(f"Probe tasks: {args.probe_tasks}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # 1. Baseline drift analysis
    print("\n[1/5] Running baseline drift analysis...")
    run_baseline_drift(
        exp_dir=args.exp_dir,
        probe_tasks=args.probe_tasks,
        task_names=task_names,
        output_dir=args.output_dir,
        neuron_ratio=args.neuron_ratio,
        seed=args.seed,
    )

    # 2. Model pairwise cosine similarity matrices
    print("\n[2/5] Running model cosine similarity analysis...")
    run_model_similarity(
        exp_dir=args.exp_dir,
        probe_tasks=args.probe_tasks,
        task_names=task_names,
        output_dir=args.output_dir,
    )

    # 3. Sample-wise similarity matrices
    if not args.skip_sample_sim:
        print("\n[3/5] Running sample similarity analysis...")
        run_sample_similarity(
            exp_dir=args.exp_dir,
            probe_tasks=args.probe_tasks,
            task_names=task_names,
            output_dir=args.output_dir,
        )
    else:
        print("\n[3/5] Skipping sample similarity (--skip_sample_sim).")

    # 4. Temporal hidden state similarity
    print("\n[4/5] Running temporal hidden state similarity...")
    run_temporal_similarity(
        exp_dir=args.exp_dir,
        probe_tasks=args.probe_tasks,
        task_names=task_names,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
    )

    # 5. Performance plots
    print("\n[5/5] Generating performance plots...")
    try:
        plot_rnn_performance(args.exp_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping performance plots: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
