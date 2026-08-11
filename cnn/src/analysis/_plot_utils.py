"""Shared plotting utilities for CNN drift analysis."""
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 26
LEGEND_FONT_SIZE = 24
LEGEND_TITLE_SIZE = 26
TITLE_SIZE = 30
SINGLE_FIGSIZE = (7.2, 7.2)
WIDE_FIGSIZE = (8.8, 5.8)
SMALL_LEGEND_FONT_SIZE = 16
SMALL_LEGEND_TITLE_SIZE = 18

CATEGORICAL_PALETTE = (
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#17BECF",  # cyan
)
OPEN_MARKER_SIZE = 22
OPEN_MARKER_LINEWIDTH = 1.2

_LAYER_PALETTE = ("#9ECAE1", "#4292C6", "#756BB1", "#3F007D")
_LAYER_MARKERS = ("o", "s", "^", "D")
_CANONICAL_LAYER_COLORS = {
    f"layer{index + 1}": color for index, color in enumerate(_LAYER_PALETTE)
}
_CANONICAL_LAYER_MARKERS = {
    f"layer{index + 1}": marker for index, marker in enumerate(_LAYER_MARKERS)
}


def categorical_colors(n: int) -> List[str]:
    """Return the first `n` high-saturation, high-contrast categorical colors.

    Raises ValueError if `n` exceeds the number of preset colors, since
    colors would otherwise have to repeat and become ambiguous.
    """
    if n > len(CATEGORICAL_PALETTE):
        raise ValueError(
            f"categorical_colors: requested {n} colors but only "
            f"{len(CATEGORICAL_PALETTE)} are defined in CATEGORICAL_PALETTE."
        )
    return list(CATEGORICAL_PALETTE[:n])


def layer_sequential_color_map(
    layer_names: Sequence[str], cmap_name: str = "Blues"
) -> Dict[str, object]:
    """Map layers to a single-hue light-to-dark gradient (e.g. Blues)."""
    n = len(layer_names)
    shades = plt.colormaps.get_cmap(cmap_name)(
        np.linspace(0.35, 0.85, n) if n > 1 else np.array([0.7])
    )
    return {name: shades[i] for i, name in enumerate(layer_names)}


def layer_color_map(layer_names: Sequence[str]) -> Dict[str, str]:
    """Map CNN layers to an ordered light-blue-to-deep-purple palette."""
    colors: Dict[str, str] = {}
    for index, layer_name in enumerate(layer_names):
        short_name = layer_name.rsplit(".", 1)[-1]
        colors[layer_name] = _CANONICAL_LAYER_COLORS.get(
            short_name, _LAYER_PALETTE[min(index, len(_LAYER_PALETTE) - 1)]
        )
    return colors


def layer_marker_map(layer_names: Sequence[str]) -> Dict[str, str]:
    """Map CNN layers to distinct markers that remain legible in grayscale."""
    markers: Dict[str, str] = {}
    for index, layer_name in enumerate(layer_names):
        short_name = layer_name.rsplit(".", 1)[-1]
        markers[layer_name] = _CANONICAL_LAYER_MARKERS.get(
            short_name, _LAYER_MARKERS[min(index, len(_LAYER_MARKERS) - 1)]
        )
    return markers


def layer_line_kwargs(color: str, marker: str) -> Dict[str, object]:
    """Return the shared high-contrast style for a CNN layer curve."""
    return {
        "color": color,
        "marker": marker,
        "linewidth": 2.0,
        "markersize": 5.5,
        "markeredgecolor": "white",
        "markeredgewidth": 0.7,
    }


def layer_errorbar_kwargs(color: str, marker: str) -> Dict[str, object]:
    """Return layer-curve styling with subdued uncertainty bars."""
    return {
        **layer_line_kwargs(color, marker),
        "ecolor": to_rgba(color, 0.48),
        "elinewidth": 1.0,
        "capthick": 1.0,
        "capsize": 3,
    }


def layer_display_name(layer: str) -> str:
    """Convert an internal layer name to a paper-friendly display label.

    Examples:
        layer1            -> Stage 1
        backbone.layer2   -> Stage 2
        backbone.stages.0 -> Stage 1
        layer4            -> Stage 4
    """
    short = layer.rsplit(".", 1)[-1]
    if short.startswith("layer"):
        num = short[len("layer"):]
        return f"Stage {num}"
    if short.startswith("stages."):
        num = short[len("stages."):]
        return f"Stage {int(num) + 1}"
    if short.startswith("stages"):
        return f"Stage {short[len('stages'):]}"
    # Fallback: try to extract a trailing number
    import re
    m = re.search(r"(\d+)$", short)
    if m:
        return f"Stage {m.group(1)}"
    return short


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
