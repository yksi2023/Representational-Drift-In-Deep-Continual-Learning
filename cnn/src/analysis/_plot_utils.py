"""Shared plotting utilities for CNN drift analysis."""
from typing import Iterable, List, Tuple


AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 26
LEGEND_FONT_SIZE = 24
LEGEND_TITLE_SIZE = 26
TITLE_SIZE = 30
SINGLE_FIGSIZE = (7.2, 7.2)
WIDE_FIGSIZE = (8.8, 5.8)
SMALL_LEGEND_FONT_SIZE = 16
SMALL_LEGEND_TITLE_SIZE = 18


def sparse_ticks(n: int) -> Tuple[List[int], List[str]]:
    """Return (positions, labels) showing only start, middle, and end ticks.

    positions: 0-indexed integers.
    labels: 1-indexed strings (plain integers, no T prefix).
    """
    if n <= 3:
        return list(range(n)), [str(i + 1) for i in range(n)]
    mid = (n - 1) // 2
    return [0, mid, n - 1], [str(1), str(mid + 1), str(n)]


def sparse_value_ticks(values: Iterable[int]) -> Tuple[List[int], List[str]]:
    """Return sparse ticks for actual x values such as task gaps."""
    vals = sorted(set(int(v) for v in values))
    if len(vals) <= 3:
        return vals, [str(v) for v in vals]
    mid = (len(vals) - 1) // 2
    ticks = [vals[0], vals[mid], vals[-1]]
    return ticks, [str(v) for v in ticks]


def apply_paper_axis_style(ax, legend: bool = False, legend_kwargs=None) -> None:
    """Apply large paper-friendly axis and optional legend fonts."""
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if legend:
        kwargs = {"fontsize": LEGEND_FONT_SIZE, "title_fontsize": LEGEND_TITLE_SIZE}
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def savefig_compact(fig, path: str) -> None:
    """Save with minimal whitespace while preserving labels."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
