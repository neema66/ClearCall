"""Independent Hybrid Method 3 DSP-plus-DL blending implementation."""

from senhance.pipeline.hybrid.method3.alignment import (
    AlignedPair,
    DelayEstimate,
    FixedDelayAligner,
    estimate_delay_cross_correlation,
    measure_impulse_delay,
)
from senhance.pipeline.hybrid.method3.config import (
    HybridAlignmentConfig,
    HybridConfig,
    HybridSTFTConfig,
    load_hybrid_config,
)
from senhance.pipeline.hybrid.method3.blender import (
    FixedWaveformBlender,
    WaveformBlendResult,
    WaveformBlendStatistics,
    blend_aligned_waveforms,
)
from senhance.pipeline.hybrid.method3.method3_config import (
    Method3DSPBackendConfig,
    Method3DSPConfig,
    Method3ListeningSetConfig,
    Method3WaveformConfig,
    load_method3_config,
)
from senhance.pipeline.hybrid.method3.paired_stft import (
    PairedSpectrumFrame,
    PairedSTFTCore,
    SpectrumProcessor,
    select_candidate_spectrum,
    select_reference_spectrum,
)

__all__ = [
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
]
