import argparse
import json
from typing import List

import torch
from torch.utils.data import DataLoader

from src.models import FashionMNISTModel, ResNet18_Tiny
from src.checkpoints import list_checkpoints, load_model
from src.representations import extract_representations
from src.drift_metrics import compute_all_metrics
from datasets import IncrementalFashionMNIST, IncrementalTinyImageNet


def main():
    parser = argparse.ArgumentParser(description="Analyze representational drift across checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="network.0,network.2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "tiny_imagenet"])
    args = parser.parse_args()

    if args.dataset == "fashion_mnist":
        data_manager = IncrementalFashionMNIST()
        model = FashionMNISTModel(output_size=10)
    elif args.dataset == "tiny_imagenet":
        data_manager = IncrementalTinyImageNet()
        model = ResNet18_Tiny()
    else:
        raise ValueError("Invalid dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_names: List[str] = [s.strip() for s in args.layers.split(",") if s.strip()]

    # Build a probe dataset (use a fixed subset from all classes)
    
    probe_loader = data_manager.get_loader(mode="test", label=range(10), batch_size=args.batch_size)

     # final head size doesn't affect hidden layers we hook
    model.to(device)

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}")

    # Extract representations for each checkpoint
    reps_per_task = {}
    for task_idx, _path in ckpts.items():
        load_model(model, args.ckpt_dir, task_idx, map_location=device)
        reps = extract_representations(model, probe_loader, layer_names, device=device, max_batches=args.max_batches)
        reps_per_task[task_idx] = {k: v.clone() for k, v in reps.items()}

    # Compute pairwise drift w.r.t. the first checkpoint as baseline
    baseline_idx = min(reps_per_task.keys())
    baseline = reps_per_task[baseline_idx]

    results = []
    for task_idx in sorted(reps_per_task.keys()):
        if task_idx == baseline_idx:
            continue
        for layer in layer_names:
            a = baseline[layer]
            b = reps_per_task[task_idx][layer]
            metrics = compute_all_metrics(a, b)
            results.append({
                "baseline_task": baseline_idx,
                "target_task": task_idx,
                "layer": layer,
                **metrics,
            })

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        for row in results:
            print(row)


if __name__ == "__main__":
    main()


