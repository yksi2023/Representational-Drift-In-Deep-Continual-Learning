"""
Training utilities - backward compatibility module.

The training functions have been refactored into src/methods/.
This module provides backward-compatible imports.
"""

# Re-export for backward compatibility
from src.methods.normal import NormalMethod
from src.methods.replay import ReplayMethod
from src.methods.ewc import EWCMethod
from src.methods.gpm import GPMMethod

__all__ = [
    'NormalMethod',
    'ReplayMethod', 
    'EWCMethod',
    'GPMMethod',
]
