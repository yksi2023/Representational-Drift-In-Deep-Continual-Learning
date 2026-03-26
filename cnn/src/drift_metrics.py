import torch
import torch.nn.functional as F
from typing import Dict, List

def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """
    Compute element-wise metrics between two representations of the same samples.
    
    Args:
        a: Tensor of shape [N, D] (Baseline activations)
        b: Tensor of shape [N, D] (Current activations)
        
    Returns:
        Dictionary containing mean and std of cosine similarity and L2 distance.
    """
    if a.size() != b.size():
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    
    if a.numel() == 0:
        return {
            "cosine_sim_mean": float("nan"),
            "cosine_sim_std": float("nan"),
            "l2_dist_mean": float("nan"),
            "l2_dist_std": float("nan"),
        }

    # 1. Cosine Similarity (逐样本计算)
    # dim=1 表示沿着特征维度计算，返回 shape [N]
    cos_sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    
    # 2. L2 Distance (逐样本计算)
    # (a - b) shape [N, D] -> norm(dim=1) -> shape [N]
    l2_dist = torch.norm(a - b, p=2, dim=1)

    return {
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_std": cos_sim.std().item(),
        "l2_dist_mean": l2_dist.mean().item(),
        "l2_dist_std": l2_dist.std().item()
    }


def compute_pairwise_similarity_matrix(reps_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between representations from different checkpoints.
    
    Args:
        reps_list: List of T tensors, each of shape [N, D].
                   The i-th tensor is the representation from the i-th checkpoint.
                   
    Returns:
        Similarity matrix of shape [T, T].
        matrix[i, j] = mean cosine similarity between reps_list[i] and reps_list[j] across all N samples.
    """
    num_tasks = len(reps_list)
    sim_matrix = torch.zeros(num_tasks, num_tasks)
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                # Cosine similarity per sample, then average
                cos_sim = F.cosine_similarity(reps_list[i], reps_list[j], dim=1, eps=1e-8)
                sim_matrix[i, j] = cos_sim.mean().item()
    
    return sim_matrix