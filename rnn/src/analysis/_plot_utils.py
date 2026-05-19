"""Shared plotting utilities for RNN drift analysis."""
from typing import List

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
    """Convert a list of raw task names to short T1..TN tick labels."""
    return [f"T{i + 1}" for i in range(len(task_names))]
