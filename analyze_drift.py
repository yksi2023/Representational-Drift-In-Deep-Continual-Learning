import argparse
import json
import os
from typing import List, Dict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.models import FashionMNISTModel, ResNet18_Tiny, PretrainedResNet18
from src.checkpoints import list_checkpoints, load_model
from src.representations import extract_representations
from src.drift_metrics import compute_metrics
from datasets import IncrementalFashionMNIST, IncrementalTinyImageNet

def plot_drift_results(results: List[Dict], output_path: str):
    """plot drift and save"""
    tasks = [r['target_task'] for r in results]
    layers = sorted(list(set(r['layer'] for r in results)))
    
    # prepare canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for layer in layers:
        layer_data = [r for r in results if r['layer'] == layer]
        # 按任务排序
        layer_data.sort(key=lambda x: x['target_task'])
        
        xs = [d['target_task'] for d in layer_data]
        
        # Cosine Sim
        cos_means = [d['cosine_sim_mean'] for d in layer_data]
        cos_stds = [d['cosine_sim_std'] for d in layer_data]
        ax1.errorbar(xs, cos_means, yerr=cos_stds, label=layer, capsize=5, marker='o')
        
        # L2 Dist
        l2_means = [d['l2_dist_mean'] for d in layer_data]
        l2_stds = [d['l2_dist_std'] for d in layer_data]
        ax2.errorbar(xs, l2_means, yerr=l2_stds, label=layer, capsize=5, marker='o')

    ax1.set_title("Cosine Similarity Decay")
    ax1.set_xlabel("Task Index")
    ax1.set_ylabel("Cosine Similarity to Baseline")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title("L2 Distance Drift")
    ax2.set_xlabel("Task Index")
    ax2.set_ylabel("L2 Distance from Baseline")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Drift plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze sample-wise representational drift")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="network.1") # 例如: "network.0, network.1"
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_img", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    # Initialize data, model and output paths
    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST()
        model = FashionMNISTModel(output_size=10)
    elif args.dataset == "tiny_imagenet":
        data_manager = IncrementalTinyImageNet()
        model = PretrainedResNet18(num_classes=200)
    else:
        raise ValueError("Invalid dataset")
    
    if args.output_json is None:
        args.output_json = os.path.join(args.ckpt_dir, "drift_stats.json")
    if args.output_img is None:
        args.output_img = os.path.join(args.ckpt_dir, "drift_plot.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]

    # Get probe data
    # read metadata to determine which classes are included
    meta_path = os.path.join(args.ckpt_dir, "model_after_task_1.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find baseline metadata at {meta_path}")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        increment = json.load(f)["training_params"]["increment"]
    
    #  Key point: shuffle=False ensures sample order is strictly consistent across Checkpoints
    probe_loader = data_manager.get_loader(
        mode="test", 
        label=range(increment), # only look at Task 1 classes
        batch_size=args.batch_size,
        shuffle=False 
    )

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")

    sorted_task_indices = sorted(ckpts.keys())
    baseline_idx = sorted_task_indices[0]
    
    print(f"Using Task {baseline_idx} as baseline.")
    
    # Extract baseline features
    load_model(model, args.ckpt_dir, baseline_idx, map_location=device)
    print(f"Extracting baseline features from Task {baseline_idx}...")
    baseline_reps = extract_representations(
        model, probe_loader, layer_names, 
        device=device, max_batches=args.max_batches, use_amp=args.amp
    )
    
    # Compare subsequent tasks
    results = []
    
    # Initial point: Baseline compares to itself (as the starting point for the chart)
    for layer in layer_names:
        results.append({
            "baseline_task": baseline_idx,
            "target_task": baseline_idx,
            "layer": layer,
            "cosine_sim_mean": 1.0,
            "cosine_sim_std": 0.0,
            "l2_dist_mean": 0.0,
            "l2_dist_std": 0.0,
        })

    for task_idx in sorted_task_indices:
        if task_idx == baseline_idx:
            continue
            
        print(f"Comparing Task {task_idx} against Baseline...")
        load_model(model, args.ckpt_dir, task_idx, map_location=device)
        
        current_reps = extract_representations(
            model, probe_loader, layer_names, 
            device=device, max_batches=args.max_batches, use_amp=args.amp
        )
        
        for layer in layer_names:
            feat_base = baseline_reps[layer]
            feat_curr = current_reps[layer]
            
            metrics = compute_metrics(feat_base, feat_curr)
            
            results.append({
                "baseline_task": baseline_idx,
                "target_task": task_idx,
                "layer": layer,
                **metrics
            })

    # Save statistics and plot
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {args.output_json}")

    plot_drift_results(results, args.output_img)

if __name__ == "__main__":
    main()