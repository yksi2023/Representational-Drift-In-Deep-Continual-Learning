import torch
import numpy as np
import random


def set_seed(seed: int):
    """Set random seed for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
