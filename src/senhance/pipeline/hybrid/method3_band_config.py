"""Compatibility imports for the former Hybrid Method 3 module path."""

from senhance.pipeline.hybrid.method3.method3_band_config import (
    CANONICAL_BIN_WIDTH_HZ,
    CANONICAL_FRAME_SIZE,
    CANONICAL_NUM_BINS,
    CANONICAL_NYQUIST_HZ,
    CANONICAL_SAMPLE_RATE,
    BandBlendVariantConfig,
    BandEvaluationConfig,
    BandListeningSetConfig,
    Method3BandConfig,
    load_method3_band_config,
)

__all__ = [
    "CANONICAL_BIN_WIDTH_HZ",
    "CANONICAL_FRAME_SIZE",
    "CANONICAL_NUM_BINS",
    "CANONICAL_NYQUIST_HZ",
    "CANONICAL_SAMPLE_RATE",
    "BandBlendVariantConfig",
    "BandEvaluationConfig",
    "BandListeningSetConfig",
    "Method3BandConfig",
    "load_method3_band_config",
]
