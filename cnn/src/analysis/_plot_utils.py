"""Shared plotting utilities for CNN drift analysis."""
from typing import List, Tuple


def sparse_ticks(n: int) -> Tuple[List[int], List[str]]:
    """Return (positions, labels) showing only start, middle, and end ticks.

    positions: 0-indexed integers.
    labels: 1-indexed strings (plain integers, no T prefix).
    """
    if n <= 3:
        return list(range(n)), [str(i + 1) for i in range(n)]
    mid = (n - 1) // 2
    return [0, mid, n - 1], [str(1), str(mid + 1), str(n)]
