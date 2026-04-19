"""One-shot representation cache.

Extracts activations for every (checkpoint, layer) pair in a single pass per
checkpoint and stores them in memory. Downstream drift analyses consume this
cache instead of re-running the model.

Also captures probe-sample labels (once) and optionally subsamples neurons per
layer (using baseline dimensions, reused across all checkpoints for
consistency).
"""
from typing import Dict, List, Optional, Tuple
import random

import torch

from src.checkpoints import list_checkpoints, load_model
from src.representations import register_activation_hooks


@torch.no_grad()
def build_reps_cache(
    model: torch.nn.Module,
    probe_loader: torch.utils.data.DataLoader,
    ckpt_dir: str,
    layer_names: List[str],
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    neuron_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Extract representations for every checkpoint × layer in one forward pass per checkpoint.

    Args:
        model: Model architecture (weights overwritten per checkpoint).
        probe_loader: DataLoader yielding (inputs, labels) with shuffle=False.
        ckpt_dir: Directory of checkpoints (model_after_task_*.pth).
        layer_names: Module names to hook.
        device: Torch device.
        max_batches: Optional batch limit.
        use_amp: Use CUDA mixed precision forward.
        neuron_ratio: Fraction in (0, 1] of neurons to randomly keep per layer.
            Indices are drawn once from baseline shape and reused for all
            checkpoints so that dimensions align.
        seed: RNG seed for neuron sampling.

    Returns:
        reps_cache: {task_idx: {layer_name: Tensor(N, D_eff) on CPU float32}}
        labels: Tensor(N,) int labels in probe order.
        neuron_indices: {layer_name: Tensor or None} indices used per layer.
    """
    if not 0.0 < neuron_ratio <= 1.0:
        raise ValueError("neuron_ratio must be in (0, 1]")

    random.seed(seed)
    torch.manual_seed(seed)

    ckpts = list_checkpoints(ckpt_dir)
    sorted_task_indices = sorted(ckpts.keys())
    if not sorted_task_indices:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    model.eval()
    reps_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    labels_cached: Optional[torch.Tensor] = None
    neuron_indices: Dict[str, Optional[torch.Tensor]] = {ln: None for ln in layer_names}

    baseline_idx = sorted_task_indices[0]

    for task_idx in sorted_task_indices:
        print(f"  [cache] extracting reps for task {task_idx}...")
        load_model(model, ckpt_dir, task_idx, map_location=device)

        activations, handles = register_activation_hooks(model, layer_names)
        collected: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
        batch_labels: List[torch.Tensor] = []
        need_labels = labels_cached is None

        try:
            for batch_idx, (inputs, lbls) in enumerate(probe_loader):
                inputs = inputs.to(device, non_blocking=True)
                if need_labels:
                    batch_labels.append(lbls.detach().cpu())

                if use_amp and device.type == "cuda":
                    with torch.amp.autocast(device_type=device.type):
                        _ = model(inputs)
                else:
                    _ = model(inputs)

                for ln in layer_names:
                    collected[ln].append(activations[ln].float())

                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        finally:
            for h in handles:
                h.remove()

        if need_labels:
            labels_cached = torch.cat(batch_labels, dim=0)

        reps_this = {
            ln: (torch.cat(v, dim=0) if v else torch.empty(0))
            for ln, v in collected.items()
        }

        # Draw neuron indices from baseline shapes, then reuse for all checkpoints.
        if task_idx == baseline_idx and neuron_ratio < 1.0:
            print(f"  [cache] sampling {neuron_ratio * 100:.1f}% neurons per layer (seed={seed})")
            for ln in layer_names:
                num_neurons = reps_this[ln].shape[1]
                num_sample = max(1, int(num_neurons * neuron_ratio))
                idx = torch.tensor(random.sample(range(num_neurons), num_sample))
                neuron_indices[ln] = idx
                print(f"    {ln}: {num_sample}/{num_neurons} neurons kept")

        for ln in layer_names:
            if neuron_indices[ln] is not None:
                reps_this[ln] = reps_this[ln][:, neuron_indices[ln]]

        reps_cache[task_idx] = reps_this

    assert labels_cached is not None
    return reps_cache, labels_cached, neuron_indices
