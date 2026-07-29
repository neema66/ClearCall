"""Independent Method 3 Version 1 DSP/DL waveform blending."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from senhance.pipeline.hybrid.method3.alignment import FixedDelayAligner
from senhance.pipeline.hybrid.method3.config import HybridConfig


@dataclasses.dataclass(frozen=True)
class WaveformBlendStatistics:
    """Read-only signal diagnostics; the blender never clips or normalizes."""

    alpha: float
    sample_count: int
    peak_abs: float
    clipping_threshold: float
    clipped_sample_count: int
    clipped_sample_fraction: float


@dataclasses.dataclass(frozen=True)
class WaveformBlendResult:
    """One exact-length blended array and its diagnostics."""

    audio: np.ndarray
    statistics: WaveformBlendStatistics
    delay_samples: int


class FixedWaveformBlender:
    """Stateless, causally aligned Method 3 Version 1.

    The caller must supply already-computed DSP and DL arrays. The concrete
    DSP implementation is selected by outer orchestration, so it can be
    replaced without changing this math core. This class applies only the
    reviewed residual DSP-to-DL delay and the fixed waveform equation.
    """

    def __init__(
        self,
        config: HybridConfig,
        *,
        alpha: Any,
        clipping_threshold: Any = 1.0,
    ) -> None:
        if not isinstance(config, HybridConfig):
            raise TypeError("config must be HybridConfig")
        config.validate()
        self.config = config
        self.alpha = _normalize_alpha(alpha)
        self.clipping_threshold = _normalize_clipping_threshold(clipping_threshold)
        self._aligner = FixedDelayAligner(config.alignment.delay_samples)

    def process_array(
        self,
        dsp: np.ndarray,
        dl: np.ndarray,
        *,
        sample_rate: int,
    ) -> WaveformBlendResult:
        """Align once and blend without clipping, normalization, or DSP calls."""

        self._validate_sample_rate(sample_rate)
        aligned = self._aligner.align(dsp, dl)
        return blend_aligned_waveforms(
            aligned.reference,
            aligned.candidate,
            alpha=self.alpha,
            clipping_threshold=self.clipping_threshold,
            delay_samples=aligned.delay_samples,
        )

    def _validate_sample_rate(self, sample_rate: int) -> None:
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate,
            (int, np.integer),
        ):
            raise TypeError("sample_rate must be an integer")
        if int(sample_rate) != self.config.sample_rate:
            raise ValueError(
                f"Method 3 sample rate must be {self.config.sample_rate} Hz, " f"got {sample_rate}"
            )


def blend_aligned_waveforms(
    aligned_dsp: np.ndarray,
    aligned_dl: np.ndarray,
    *,
    alpha: Any,
    clipping_threshold: Any = 1.0,
    delay_samples: int = 0,
) -> WaveformBlendResult:
    """Blend already-aligned arrays as ``alpha*DL + (1-alpha)*DSP``.

    Endpoint branches intentionally return bit-exact copies. Intermediate
    blends are evaluated in float64 and converted once to contiguous float32.
    """

    weight = _normalize_alpha(alpha)
    threshold = _normalize_clipping_threshold(clipping_threshold)
    pair = FixedDelayAligner(0).align(aligned_dsp, aligned_dl)
    delay = _normalize_delay_metadata(delay_samples)

    if weight == 0.0:
        output = pair.reference.copy()
    elif weight == 1.0:
        output = pair.candidate.copy()
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            blended = weight * pair.candidate.astype(np.float64)
            blended += (1.0 - weight) * pair.reference.astype(np.float64)
            output = np.asarray(blended, dtype=np.float32)
        if not np.all(np.isfinite(output)):
            raise ValueError("Method 3 blend exceeds the float32 finite range")
        output = np.ascontiguousarray(output)

    if not np.all(np.isfinite(output)):
        raise ValueError("Method 3 output must contain only finite samples")
    sample_count = int(output.size)
    peak_abs = float(np.max(np.abs(output))) if sample_count else 0.0
    clipped_sample_count = int(np.count_nonzero(np.abs(output.astype(np.float64)) >= threshold))
    statistics = WaveformBlendStatistics(
        alpha=weight,
        sample_count=sample_count,
        peak_abs=peak_abs,
        clipping_threshold=threshold,
        clipped_sample_count=clipped_sample_count,
        clipped_sample_fraction=(clipped_sample_count / sample_count if sample_count else 0.0),
    )
    return WaveformBlendResult(
        audio=np.ascontiguousarray(output, dtype=np.float32),
        statistics=statistics,
        delay_samples=delay,
    )


def _normalize_alpha(value: Any) -> float:
    alpha = _normalize_real_scalar(value, "alpha")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must satisfy 0.0 <= alpha <= 1.0")
    return alpha


def _normalize_clipping_threshold(value: Any) -> float:
    threshold = _normalize_real_scalar(value, "clipping_threshold")
    if threshold <= 0.0:
        raise ValueError("clipping_threshold must be positive")
    return threshold


def _normalize_real_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _normalize_delay_metadata(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("delay_samples must be an integer")
    return int(value)
