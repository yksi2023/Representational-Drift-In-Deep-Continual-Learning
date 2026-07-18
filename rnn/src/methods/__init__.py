from src.methods.normal import NormalMethod
from src.methods.ewc import EWCMethod
from src.methods.replay import ReplayMethod
from src.methods.anchored_replay import AnchoredReplayMethod
from src.methods.lwf import LwFMethod
from src.methods.hypernet import HyperNetMethod
from src.methods.duncker import DunckerMethod

_METHOD_MAP = {
    'normal': NormalMethod,
    'ewc': EWCMethod,
    'replay': ReplayMethod,
    'anchored_replay': AnchoredReplayMethod,
    'lwf': LwFMethod,
    'hypernet': HyperNetMethod,
    'duncker': DunckerMethod,
}

def get_method(name: str):
    """Factory function to get a continual learning method class by name."""
    key = name.lower()
    if key not in _METHOD_MAP:
        raise ValueError(f"Unknown method '{name}'. Available: {list(_METHOD_MAP.keys())}")
    return _METHOD_MAP[key]
