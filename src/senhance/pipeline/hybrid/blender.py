"""Compatibility imports for the former Hybrid Method 3 module path."""

from senhance.pipeline.hybrid.method3.blender import (
    FixedWaveformBlender,
    WaveformBlendResult,
    WaveformBlendStatistics,
    blend_aligned_waveforms,
)

__all__ = [
    "FixedWaveformBlender",
    "WaveformBlendResult",
    "WaveformBlendStatistics",
    "blend_aligned_waveforms",
]
