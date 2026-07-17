"""Compatibility imports for the former Hybrid Method 3 module path."""

from senhance.pipeline.hybrid.method3.alignment import (
    AlignedPair,
    DelayEstimate,
    FixedDelayAligner,
    estimate_delay_cross_correlation,
    measure_impulse_delay,
)

__all__ = [
    "AlignedPair",
    "DelayEstimate",
    "FixedDelayAligner",
    "estimate_delay_cross_correlation",
    "measure_impulse_delay",
]
