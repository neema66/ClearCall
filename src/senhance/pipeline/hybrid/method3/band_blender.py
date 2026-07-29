"""Independent Method 3 Version 2 DSP/DL fixed-frequency-band wrapper."""

from __future__ import annotations

import dataclasses

import numpy as np

from senhance.pipeline.hybrid.method3.bands import (
    FixedBandDefinition,
    FixedBandSpectrumProcessor,
)
from senhance.pipeline.hybrid.method3.config import HybridConfig
from senhance.pipeline.hybrid.method3.method3_band_config import Method3BandConfig
from senhance.pipeline.hybrid.method3.paired_stft import PairedSTFTCore


@dataclasses.dataclass(frozen=True)
class BandBlendStatistics:
    """Read-only output diagnostics; no clipping or normalization is applied."""

    variant_id: str
    blend_domain: str
    phase_source: str
    frequency_smoothing_bins: int
    sample_count: int
    peak_abs: float
    clipping_threshold: float
    clipped_sample_count: int
    clipped_sample_fraction: float
    minimum_dl_weight: float
    maximum_dl_weight: float


@dataclasses.dataclass(frozen=True)
class BandBlendResult:
    """One exact-length fixed-band output and its diagnostics."""

    audio: np.ndarray
    statistics: BandBlendStatistics
    delay_samples: int


class FixedFrequencyBandBlender:
    """Apply fixed band policy over the neutral paired-STFT array contract."""

    def __init__(
        self,
        hybrid_config: HybridConfig,
        method_config: Method3BandConfig,
        *,
        variant_id: str,
    ) -> None:
        if not isinstance(hybrid_config, HybridConfig):
            raise TypeError("hybrid_config must be HybridConfig")
        if not isinstance(method_config, Method3BandConfig):
            raise TypeError("method_config must be Method3BandConfig")
        hybrid_config.validate()
        method_config.validate()
        variant = method_config.variant(variant_id)

        self.hybrid_config = hybrid_config
        self.method_config = method_config
        self.variant = variant
        self.definition = FixedBandDefinition.from_config(
            method_config,
            variant,
            hybrid_config,
        )
        self.processor = FixedBandSpectrumProcessor(self.definition)
        self._core = PairedSTFTCore(hybrid_config)

    def reset(self) -> None:
        """Clear shared framing state; the fixed spectral policy is stateless."""

        self._core.reset()
        self.processor.reset()

    def process_array(
        self,
        dsp: np.ndarray,
        dl: np.ndarray,
        *,
        sample_rate: int,
    ) -> BandBlendResult:
        """Return one aligned, exact-length fixed-band blend and diagnostics."""

        output = self._core.process_array(
            dsp,
            dl,
            self.processor,
            sample_rate=sample_rate,
        )
        if not np.all(np.isfinite(output)):
            raise ValueError("fixed-band array output must contain only finite samples")
        output = np.ascontiguousarray(output, dtype=np.float32)
        sample_count = int(output.size)
        absolute = np.abs(output.astype(np.float64))
        peak_abs = float(np.max(absolute)) if sample_count else 0.0
        threshold = self.method_config.evaluation.clipping_threshold
        clipped_count = int(np.count_nonzero(absolute >= threshold))
        weights = self.definition.effective_dl_weight_by_bin
        statistics = BandBlendStatistics(
            variant_id=self.variant.id,
            blend_domain=self.variant.blend_domain,
            phase_source=self.variant.phase_source,
            frequency_smoothing_bins=self.variant.frequency_smoothing_bins,
            sample_count=sample_count,
            peak_abs=peak_abs,
            clipping_threshold=threshold,
            clipped_sample_count=clipped_count,
            clipped_sample_fraction=(clipped_count / sample_count if sample_count else 0.0),
            minimum_dl_weight=float(np.min(weights)),
            maximum_dl_weight=float(np.max(weights)),
        )
        return BandBlendResult(
            audio=output,
            statistics=statistics,
            delay_samples=self.hybrid_config.alignment.delay_samples,
        )
