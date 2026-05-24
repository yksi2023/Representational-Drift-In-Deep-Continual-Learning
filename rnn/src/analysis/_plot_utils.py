"""Shared plotting utilities for RNN drift analysis."""
from typing import Iterable, List


AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 26
LEGEND_FONT_SIZE = 24
LEGEND_TITLE_SIZE = 26
TITLE_SIZE = 30

# ---------------------------------------------------------------------------
# Default 18-task sequence (DEFAULT_TASKS order in datasets.py).
# Tick labels on all plots use T1..T18; mapping to task names:
#
#   T1  = fdgo             Go, fixation-onset cue
#   T2  = reactgo          Go, stimulus-onset cue
#   T3  = delaygo          Go with delay period
#   T4  = fdanti           Anti-go, fixation-onset cue
#   T5  = reactanti        Anti-go, stimulus-onset cue
#   T6  = delayanti        Anti-go with delay period
#   T7  = dm1              Decision-making, modality 1
#   T8  = dm2              Decision-making, modality 2
#   T9  = contextdm1       Context-dependent DM, modality 1
#   T10 = contextdm2       Context-dependent DM, modality 2
#   T11 = multidm          Multi-sensory integration DM
#   T12 = delaydm1         Delayed DM, modality 1
#   T13 = delaydm2         Delayed DM, modality 2
#   T14 = contextdelaydm1  Context-dependent delayed DM, modality 1
#   T15 = contextdelaydm2  Context-dependent delayed DM, modality 2
#   T16 = multidelaydm     Multi-sensory delayed DM
#   T17 = dmsgo            DMS Go (match-to-sample)
#   T18 = dmsnogo          DMS No-go
# ---------------------------------------------------------------------------


def t_labels(task_names: List[str]) -> List[str]:
    """Convert a list of raw task names to 1-indexed integer tick labels."""
    return [str(i + 1) for i in range(len(task_names))]


def sparse_ticks(n: int):
    """Return (positions, labels) showing only start, middle, and end ticks.

    positions: 0-indexed integers.
    labels: 1-indexed strings (plain integers, no T prefix).
    """
    if n <= 3:
        return list(range(n)), [str(i + 1) for i in range(n)]
    mid = (n - 1) // 2
    return [0, mid, n - 1], [str(1), str(mid + 1), str(n)]


def sparse_value_ticks(values: Iterable[int]):
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
