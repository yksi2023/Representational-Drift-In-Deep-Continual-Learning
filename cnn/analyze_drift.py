"""Analyze representational drift across continual learning checkpoints.

Pipeline:
  0. Build a one-shot representation cache: for every checkpoint, run a single
     forward pass on the probe loader and collect activations for every
     requested layer. Optional per-layer neuron subsampling (same indices
     across all checkpoints).
  1. Baseline drift (cosine / L2 / shuffled control vs first checkpoint).
  2. Model pairwise cosine similarity matrices + decay profile vs task gap.
  3. Sample-wise cosine similarity matrices (per checkpoint, per layer).
  4. Coding / null subspace drift decomposition (PCA of baseline reps).
  5. Pearson correlation of Sample-PV and ERV vs task gap.
  6. Performance plots from saved training metrics.
"""
import argparse
import json
import os
from typing import List

import torch

from src.models import MODEL_DEFAULTS, build_model
from src.checkpoints import list_checkpoints
from src.analysis import (
    build_reps_cache,
    run_baseline_drift,
    run_model_similarity,
    run_sample_similarity,
    run_subspace_drift,
    run_gap_drift,
    plot_cnn_performance,
)
from src.eval import plot_performance_from_files
from datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze representational drift")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="network.1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save all analysis results (defaults to ckpt_dir/drift_analysis)")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--neuron_ratio", type=float, default=1.0,
                        help="Ratio of neurons to randomly sample (0.0, 1.0], default: 1.0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for neuron sampling reproducibility")
    parser.add_argument("--probe_type", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--subspace_threshold", type=float, default=0.99,
                        help="Cumulative variance threshold defining coding subspace")
    parser.add_argument("--skip_sample_sim", action="store_true",
                        help="Skip sample similarity matrices (costly for many checkpoints)")
    parser.add_argument("--skip_distributions", action="store_true",
                        help="Skip per-task activation histograms in baseline drift")
    return parser.parse_args()


def setup_environment(args):
    if not 0.0 < args.neuron_ratio <= 1.0:
        raise ValueError("--neuron_ratio must be in range (0.0, 1.0]")

    # Recover dataset / model / class count from the training config so we
    # always instantiate the same architecture the checkpoint was saved with.
    config_path = os.path.join(args.ckpt_dir, "experiment_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. Re-run training with the current run_experiment.py "
            f"so it writes dataset/model/num_classes into the config.")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_name = cfg["dataset"]
    defaults = MODEL_DEFAULTS[dataset_name]
    model_name = cfg.get("model", defaults["model"])
    num_classes = cfg.get("num_classes", defaults["num_classes"])
    img_size = cfg.get("img_size", defaults["img_size"])

    data_manager = build_dataset(
        dataset_name,
        num_classes=num_classes,
        img_size=img_size,
        val_ratio=cfg.get("val_ratio", 0.1),
    )
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=not cfg.get("no_pretrained", False),
        freeze_layers=cfg.get("freeze_layers", ""),
        freeze_until=cfg.get("freeze_until"),
    )
    print(f"Dataset: {dataset_name} (num_classes={num_classes}, img_size={img_size})")
    print(f"Model:   {model_name}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "drift_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]

    meta_path = os.path.join(args.ckpt_dir, "model_after_task_1.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find baseline metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        increment = json.load(f)["training_params"]["increment"]

    probe_loader = data_manager.get_loader(
        mode=args.probe_type,
        label=range(increment),
        batch_size=args.batch_size,
        shuffle=False,
    )

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")

    return model, probe_loader, layer_names, device


def main():
    args = parse_args()
    model, probe_loader, layer_names, device = setup_environment(args)

    print("=" * 60)
    print("DRIFT ANALYSIS")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Layers: {layer_names}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # 0. Build one-shot representation cache
    print("\n[0/6] Building representation cache (one pass per checkpoint)...")
    reps_cache, labels, neuron_indices = build_reps_cache(
        model=model,
        probe_loader=probe_loader,
        ckpt_dir=args.ckpt_dir,
        layer_names=layer_names,
        device=device,
        max_batches=args.max_batches,
        use_amp=args.amp,
        neuron_ratio=args.neuron_ratio,
        seed=args.seed,
    )
    print(f"  Cache built: {len(reps_cache)} checkpoints × {len(layer_names)} layers, "
          f"N={labels.shape[0]} probe samples")

    # 1. Baseline drift
    print("\n[1/6] Running baseline drift analysis...")
    run_baseline_drift(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
        plot_distributions=not args.skip_distributions,
    )

    # 2. Model pairwise cosine similarity
    print("\n[2/6] Running model similarity analysis...")
    run_model_similarity(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
    )

    # 3. Sample-wise similarity
    if not args.skip_sample_sim:
        print("\n[3/6] Running sample similarity analysis...")
        run_sample_similarity(
            reps_cache=reps_cache,
            labels=labels,
            layer_names=layer_names,
            output_dir=args.output_dir,
        )
    else:
        print("\n[3/6] Skipping sample similarity (--skip_sample_sim).")

    # 4. Coding / null subspace drift decomposition
    print("\n[4/6] Running coding/null subspace drift analysis...")
    run_subspace_drift(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
        threshold=args.subspace_threshold,
    )

    # 5. Gap-based vector drift (Sample-PV + ERV Pearson vs task gap)
    print("\n[5/6] Running gap-based vector drift analysis...")
    run_gap_drift(
        reps_cache=reps_cache,
        layer_names=layer_names,
        output_dir=args.output_dir,
    )

    # 6. Performance plots (line plots + task x stage accuracy heatmap)
    print("\n[6/6] Generating performance plots...")
    try:
        plot_performance_from_files(args.ckpt_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping line plots: {e}")
    try:
        plot_cnn_performance(args.ckpt_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping accuracy matrix: {e}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
