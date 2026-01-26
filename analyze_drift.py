"""Analyze representational drift across continual learning checkpoints.

This script provides comprehensive drift analysis including:
1. Baseline drift metrics (cosine similarity, L2 distance over tasks)
2. Model pairwise similarity matrices
3. Sample-wise similarity matrices
4. Performance plots (from saved training metrics)
"""
import argparse
import json
import os
from typing import List

import torch

from src.models import FashionMNISTModel, ResNet18_Tiny, PretrainedResNet18
from src.checkpoints import list_checkpoints
from src.analysis import run_baseline_drift, run_model_similarity, run_sample_similarity
from src.eval import plot_performance_from_files
from datasets import IncrementalFashionMNIST, IncrementalTinyImageNet


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sample-wise representational drift")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="network.1") 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save all analysis results (defaults to ckpt_dir/drift_analysis)")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--neuron_ratio", type=float, default=1.0,
                        help="Ratio of neurons to randomly sample for drift analysis (0.0-1.0, default: 1.0 for all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for neuron sampling reproducibility")
    parser.add_argument("--probe_type", type=str, default="test", choices=["test","train"])
    return parser.parse_args()


def setup_environment(args):
    """Initialize model, data manager, and common settings."""
    # Validate neuron_ratio
    if not 0.0 < args.neuron_ratio <= 1.0:
        raise ValueError("--neuron_ratio must be in range (0.0, 1.0]")

    # Initialize data manager and model
    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST()
        model = FashionMNISTModel(output_size=10)
    elif args.dataset == "tiny_imagenet":
        data_manager = IncrementalTinyImageNet()
        model = PretrainedResNet18(num_classes=200)
    else:
        raise ValueError("Invalid dataset")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "drift_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Parse layer names
    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]
    
    # Read metadata to determine increment
    meta_path = os.path.join(args.ckpt_dir, "model_after_task_1.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find baseline metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        increment = json.load(f)["training_params"]["increment"]
    
    # Create probe loader (shuffle=False for consistent sample order)
    probe_loader = data_manager.get_loader(
        mode=args.probe_type, 
        label=range(increment),
        batch_size=args.batch_size,
        shuffle=False 
    )
    
    # Verify checkpoints exist
    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")
    
    return model, probe_loader, layer_names, device


def main():
    args = parse_args()
    model, probe_loader, layer_names, device = setup_environment(args)
    
    print("="*60)
    print("DRIFT ANALYSIS")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Layers: {layer_names}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    # 1. Baseline drift analysis
    print("\n[1/4] Running baseline drift analysis...")
    neuron_indices = run_baseline_drift(
        model=model,
        probe_loader=probe_loader,
        ckpt_dir=args.ckpt_dir,
        layer_names=layer_names,
        output_dir=args.output_dir,
        device=device,
        max_batches=args.max_batches,
        use_amp=args.amp,
        neuron_ratio=args.neuron_ratio,
        seed=args.seed,
    )
    
    # 2. Model pairwise similarity matrices
    print("\n[2/4] Running model similarity analysis...")
    run_model_similarity(
        model=model,
        probe_loader=probe_loader,
        ckpt_dir=args.ckpt_dir,
        layer_names=layer_names,
        output_dir=args.output_dir,
        device=device,
        max_batches=args.max_batches,
        use_amp=args.amp,
        neuron_indices=neuron_indices,
    )
    
    # 3. Sample-wise similarity matrices
    print("\n[3/4] Running sample similarity analysis...")
    run_sample_similarity(
        model=model,
        probe_loader=probe_loader,
        ckpt_dir=args.ckpt_dir,
        layer_names=layer_names,
        output_dir=args.output_dir,
        device=device,
        max_batches=args.max_batches,
        use_amp=args.amp,
        neuron_indices=neuron_indices,
    )
    
    # 4. Performance plots
    print("\n[4/4] Generating performance plots...")
    try:
        plot_performance_from_files(args.ckpt_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"  Skipping performance plots: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
