from typing import Dict, Tuple

import torch


def mean_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 distance between feature means.

    a, b: [N, D]
    """
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    return torch.norm(mu_a - mu_b, p=2).item()


def cosine_between_means(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between feature means.

    Returns in [-1, 1]. Higher is more similar; drift can be 1 - cosine.
    """
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    denom = (mu_a.norm(p=2) * mu_b.norm(p=2)).clamp_min(1e-12)
    return (mu_a @ mu_b / denom).item()


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    one_n = torch.full((n, n), 1.0 / n, device=K.device, dtype=K.dtype)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear CKA between representations a and b.

    a, b: [N, D]
    """
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    A = a - a.mean(dim=0)
    B = b - b.mean(dim=0)
    Ka = A @ A.t()
    Kb = B @ B.t()
    Ka = _center_gram(Ka)
    Kb = _center_gram(Kb)
    hsic_ab = (Ka * Kb).sum()
    hsic_aa = (Ka * Ka).sum().clamp_min(1e-12)
    hsic_bb = (Kb * Kb).sum().clamp_min(1e-12)
    return (hsic_ab / torch.sqrt(hsic_aa * hsic_bb)).item()


def compute_all_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Convenience wrapper to compute all drift-related metrics.

    Returns a dict with keys: mean_shift, cosine_between_means, linear_cka
    """
    return {
        "mean_shift": mean_shift(a, b),
        "cosine_between_means": cosine_between_means(a, b),
        "linear_cka": linear_cka(a, b),
    }


