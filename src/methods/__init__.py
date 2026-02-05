from src.methods.base import BaseContinualMethod
from src.methods.normal import NormalMethod
from src.methods.replay import ReplayMethod
from src.methods.ewc import EWCMethod
from src.methods.gpm import GPMMethod

__all__ = [
    'BaseContinualMethod',
    'NormalMethod',
    'ReplayMethod',
    'EWCMethod',
    'GPMMethod',
]

METHOD_REGISTRY = {
    'normal': NormalMethod,
    'replay': ReplayMethod,
    'ewc': EWCMethod,
    'gpm': GPMMethod,
}

def get_method(name: str):
    """Get continual learning method class by name."""
    name = name.lower()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]
