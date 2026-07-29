"""Independent Hybrid Method 1 and Method 3 package namespace.

The two implementations live in separate subpackages.  Legacy Method 3
package-level exports remain available through lazy attribute loading so that
importing Method 1 never imports Method 3 as a side effect.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBPACKAGES = ("method1", "method3")
_LEGACY_METHOD3_EXPORTS = (
    "AlignedPair",
    "DelayEstimate",
    "FixedDelayAligner",
    "FixedWaveformBlender",
    "HybridAlignmentConfig",
    "HybridConfig",
    "HybridSTFTConfig",
    "Method3ListeningSetConfig",
    "Method3DSPBackendConfig",
    "Method3DSPConfig",
    "Method3WaveformConfig",
    "PairedSTFTCore",
    "PairedSpectrumFrame",
    "SpectrumProcessor",
    "WaveformBlendResult",
    "WaveformBlendStatistics",
    "blend_aligned_waveforms",
    "estimate_delay_cross_correlation",
    "load_hybrid_config",
    "load_method3_config",
    "measure_impulse_delay",
    "select_candidate_spectrum",
    "select_reference_spectrum",
)

__all__ = [*_SUBPACKAGES, *_LEGACY_METHOD3_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load a hybrid subpackage or legacy Method 3 export only when requested."""

    if name in _SUBPACKAGES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LEGACY_METHOD3_EXPORTS:
        method3 = import_module(f"{__name__}.method3")
        value = getattr(method3, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazily available names to interactive tooling."""

    return sorted(set(globals()) | set(__all__))
