"""Compatibility wrapper for the old baseline_drift module name."""

from .reference_drift import (
    _load_reps_from_npz,
    plot_drift_results,
    run_baseline_drift,
    run_reference_drift,
)

__all__ = [
    "_load_reps_from_npz",
    "plot_drift_results",
    "run_baseline_drift",
    "run_reference_drift",
]
