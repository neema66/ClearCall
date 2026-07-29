"""Compatibility imports for the former Hybrid Method 3 module path."""

from senhance.pipeline.hybrid.method3.paired_stft import (
    PairedSpectrumFrame,
    PairedSTFTCore,
    SpectrumProcessor,
    select_candidate_spectrum,
    select_reference_spectrum,
)

__all__ = [
    "PairedSpectrumFrame",
    "PairedSTFTCore",
    "SpectrumProcessor",
    "select_candidate_spectrum",
    "select_reference_spectrum",
]
