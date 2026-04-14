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
        Dictionary containing mean and std of cosine similarity, L2 distance,
        and shuffled baseline similarity.
    """
    if a.size() != b.size():
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if a.numel() == 0:
        return {
            "cosine_sim_mean": float("nan"),
            "cosine_sim_std": float("nan"),
            "l2_dist_mean": float("nan"),
            "l2_dist_std": float("nan"),
            "shuffled_sim_mean": float("nan"),
            "shuffled_sim_std": float("nan"),
        }

    cos_sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    l2_dist = torch.norm(a - b, p=2, dim=1)

    # Shuffled baseline: shuffle features within each sample
    N, D = b.shape
    rand_idx = torch.rand(N, D, device=b.device).argsort(dim=1)
    b_shuffled = torch.gather(b, 1, rand_idx)
    shuffled_sim = F.cosine_similarity(a, b_shuffled, dim=1, eps=1e-8)

    return {
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_std": cos_sim.std().item(),
        "l2_dist_mean": l2_dist.mean().item(),
        "l2_dist_std": l2_dist.std().item(),
        "shuffled_sim_mean": shuffled_sim.mean().item(),
        "shuffled_sim_std": shuffled_sim.std().item(),
    }


def compute_pairwise_similarity_matrix(reps_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between representations from different checkpoints.

    Args:
        reps_list: List of T tensors, each of shape [N, D].

    Returns:
        Similarity matrix of shape [T, T].
    """
    num_tasks = len(reps_list)
    sim_matrix = torch.zeros(num_tasks, num_tasks)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                cos_sim = F.cosine_similarity(reps_list[i], reps_list[j], dim=1, eps=1e-8)
                sim_matrix[i, j] = cos_sim.mean().item()

    return sim_matrix


def compute_pairwise_pearson_matrix(reps_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise Pearson correlation matrix between checkpoint representations.

    Pearson correlation = cosine similarity of mean-centred vectors.

    Args:
        reps_list: List of T tensors, each of shape [N, D].

    Returns:
        Correlation matrix of shape [T, T].
    """
    num_tasks = len(reps_list)
    corr_matrix = torch.zeros(num_tasks, num_tasks)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                a = reps_list[i] - reps_list[i].mean(dim=1, keepdim=True)
                b = reps_list[j] - reps_list[j].mean(dim=1, keepdim=True)
                corr = F.cosine_similarity(a, b, dim=1, eps=1e-8)
                corr_matrix[i, j] = corr.mean().item()

    return corr_matrix


def compute_linear_cka(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute linear CKA between two representation matrices.

    Args:
        a: Tensor of shape [N, D]
        b: Tensor of shape [N, D]
        eps: Numerical stability term

    Returns:
        Scalar CKA value in [0, 1].
    """
    if a.size() != b.size():
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if a.numel() == 0 or a.size(0) < 2:
        return float("nan")

    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)

    cross = torch.matmul(a_centered, b_centered.t())
    self_a = torch.matmul(a_centered, a_centered.t())
    self_b = torch.matmul(b_centered, b_centered.t())

    numerator = (cross ** 2).sum()
    denominator = torch.sqrt((self_a ** 2).sum() * (self_b ** 2).sum()) + eps
    return (numerator / denominator).item()


def compute_pairwise_cka_matrix(reps_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise linear CKA matrix between checkpoint representations.

    Args:
        reps_list: List of T tensors, each of shape [N, D].

    Returns:
        CKA matrix of shape [T, T].
    """
    num_tasks = len(reps_list)
    cka_matrix = torch.zeros(num_tasks, num_tasks)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                cka_matrix[i, j] = 1.0
            else:
                cka_matrix[i, j] = compute_linear_cka(reps_list[i], reps_list[j])

    return cka_matrix
