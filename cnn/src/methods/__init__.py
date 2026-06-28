from src.methods.base import BaseContinualMethod
from src.methods.normal import NormalMethod
from src.methods.replay import ReplayMethod
from src.methods.anchored_replay import AnchoredReplayMethod  # [exp-b]
from src.methods.ewc import EWCMethod
from src.methods.gpm import GPMMethod
from src.methods.lwf import LwFMethod

__all__ = [
    'BaseContinualMethod',
    'NormalMethod',
    'ReplayMethod',
    'AnchoredReplayMethod',  # [exp-b]
    'EWCMethod',
    'GPMMethod',
    'LwFMethod',
]

METHOD_REGISTRY = {
    'normal': NormalMethod,
    'replay': ReplayMethod,
    'anchored_replay': AnchoredReplayMethod,  # [exp-b]
    'ewc': EWCMethod,
    'gpm': GPMMethod,
    'lwf': LwFMethod,
}

def get_method(name: str):
    """Get continual learning method class by name."""
    name = name.lower()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]
