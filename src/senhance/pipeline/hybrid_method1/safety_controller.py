"""Compatibility imports for the former Hybrid Method 1 module path."""

from senhance.pipeline.hybrid.method1.safety_controller import (
    GainStageResult,
    SafetyGainController,
    apply_gain_floor,
    limit_gain_change,
    smooth_gain_across_frequency,
    temporal_exponential_smooth,
)

__all__ = [
    "GainStageResult",
    "SafetyGainController",
    "apply_gain_floor",
    "limit_gain_change",
    "smooth_gain_across_frequency",
    "temporal_exponential_smooth",
]
