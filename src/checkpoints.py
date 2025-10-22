import os
import json
from typing import Any, Dict, Optional

import torch


def _task_ckpt_path(save_dir: str, task_idx: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"model_after_task_{task_idx}.pth")


def _task_meta_path(save_dir: str, task_idx: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"model_after_task_{task_idx}.json")


def save_model(
    model: torch.nn.Module,
    save_dir: str,
    task_idx: int,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model state dict and metadata for a given task index.

    Returns the checkpoint path.
    """
    ckpt_path = _task_ckpt_path(save_dir, task_idx)
    meta_path = _task_meta_path(save_dir, task_idx)
    torch.save(model.state_dict(), ckpt_path)
    metadata: Dict[str, Any] = {"task_idx": task_idx}
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return ckpt_path


def load_model(
    model: torch.nn.Module,
    save_dir: str,
    task_idx: int,
    map_location: Optional[str] = None,
) -> torch.nn.Module:
    """Load model state dict for a given task index into the provided model."""
    ckpt_path = _task_ckpt_path(save_dir, task_idx)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def list_checkpoints(save_dir: str) -> Dict[int, str]:
    """Return a mapping from task_idx to checkpoint path found in save_dir."""
    if not os.path.isdir(save_dir):
        return {}
    mapping: Dict[int, str] = {}
    for fname in os.listdir(save_dir):
        if fname.startswith("model_after_task_") and fname.endswith(".pth"):
            try:
                idx_str = fname[len("model_after_task_") : -len(".pth")]
                idx = int(idx_str)
                mapping[idx] = os.path.join(save_dir, fname)
            except ValueError:
                continue
    return dict(sorted(mapping.items()))


