"""Causal fixed-delay alignment and explicit offline delay diagnostics."""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import DTypeLike
from scipy.signal import correlate, correlation_lags  # type: ignore[import-untyped]


@dataclasses.dataclass(frozen=True)
class AlignedPair:
    """Equal-length reference/candidate arrays after delaying the faster path.

    Method 3 supplies the selected DSP output as ``reference`` and the DL
    output as ``candidate``.  The neutral names keep the aligner reusable by
    other hybrid methods without hiding Method 3's mandatory DSP branch.
    """

    reference: np.ndarray
    candidate: np.ndarray
    delay_samples: int


@dataclasses.dataclass(frozen=True)
class DelayEstimate:
    """Read-only diagnostic result; it never updates production alignment."""

    delay_samples: int
    normalized_peak: float
    max_abs_delay: int | None
    peak_at_search_boundary: bool
    ambiguous_peak: bool


class FixedDelayAligner:
    """Apply one configured signed residual delay without advancing a path.

    Sign convention::

        delay_samples = candidate event index - reference event index

    Positive means the candidate lags, so the reference path is delayed.
    Negative means the candidate leads, so the candidate path is delayed.
    Delaying a path prepends zeros and drops the same number of tail samples.
    Both outputs always retain length ``N``.
    """

    def __init__(self, delay_samples: int) -> None:
        self.delay_samples = _require_signed_integer(delay_samples, "delay_samples")

    def align(self, reference: np.ndarray, candidate: np.ndarray) -> AlignedPair:
        reference_samples, candidate_samples = _validate_pair(
            reference,
            candidate,
            dtype=np.float32,
        )
        sample_count = reference_samples.size
        delay = self.delay_samples

        aligned_reference = reference_samples.copy()
        aligned_candidate = candidate_samples.copy()
        if delay > 0:
            aligned_reference.fill(0.0)
            if delay < sample_count:
                aligned_reference[delay:] = reference_samples[: sample_count - delay]
        elif delay < 0:
            candidate_delay = -delay
            aligned_candidate.fill(0.0)
            if candidate_delay < sample_count:
                aligned_candidate[candidate_delay:] = candidate_samples[
                    : sample_count - candidate_delay
                ]

        return AlignedPair(
            reference=np.ascontiguousarray(aligned_reference, dtype=np.float32),
            candidate=np.ascontiguousarray(aligned_candidate, dtype=np.float32),
            delay_samples=delay,
        )


def estimate_delay_cross_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_abs_delay: int | None = None,
) -> DelayEstimate:
    """Estimate residual candidate lag using full, demeaned cross-correlation.

    This is a diagnostic only. Production code must use a reviewed fixed value
    from ``config/hybrid.yaml`` and must not call this function per clip.
    """

    reference_samples, candidate_samples = _validate_pair(
        reference,
        candidate,
        dtype=np.float64,
    )
    if reference_samples.size == 0:
        raise ValueError("Delay diagnostic requires non-empty audio")

    reference_centered = _normalized_centered(reference_samples, "reference")
    candidate_centered = _normalized_centered(candidate_samples, "candidate")
    normalization = float(np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered))
    if not np.isfinite(normalization) or normalization <= np.finfo(np.float64).eps:
        raise ValueError("Delay diagnostic requires non-silent, non-constant audio")

    correlation = correlate(
        candidate_centered,
        reference_centered,
        mode="full",
        method="fft",
    )
    lags = correlation_lags(
        candidate_centered.size,
        reference_centered.size,
        mode="full",
    )

    limit = None
    if max_abs_delay is not None:
        limit = _require_signed_integer(max_abs_delay, "max_abs_delay")
        if limit < 0:
            raise ValueError("max_abs_delay must be non-negative")
        keep = np.abs(lags) <= limit
        correlation = correlation[keep]
        lags = lags[keep]
        if correlation.size == 0:
            raise ValueError("max_abs_delay excludes every correlation lag")

    absolute = np.abs(correlation)
    peak_value = float(np.max(absolute))
    peak_indices = np.flatnonzero(np.isclose(absolute, peak_value, rtol=1e-10, atol=1e-12))
    # Prefer the candidate closest to zero if a periodic signal has tied peaks.
    selected_index = int(peak_indices[np.argmin(np.abs(lags[peak_indices]))])
    delay = int(lags[selected_index])
    normalized_peak = float(correlation[selected_index] / normalization)
    return DelayEstimate(
        delay_samples=delay,
        normalized_peak=normalized_peak,
        max_abs_delay=limit,
        peak_at_search_boundary=(limit is not None and abs(delay) == limit),
        ambiguous_peak=peak_indices.size > 1,
    )


def measure_impulse_delay(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Return ``candidate peak index - reference peak index`` for impulses."""

    reference_samples, candidate_samples = _validate_pair(
        reference,
        candidate,
        dtype=np.float64,
    )
    if reference_samples.size == 0:
        raise ValueError("Impulse diagnostic requires non-empty audio")
    reference_index = _unique_absolute_peak(reference_samples, "reference")
    candidate_index = _unique_absolute_peak(candidate_samples, "candidate")
    return candidate_index - reference_index


def _unique_absolute_peak(samples: np.ndarray, label: str) -> int:
    magnitude = np.abs(samples)
    peak = float(np.max(magnitude))
    if peak <= np.finfo(np.float64).eps:
        raise ValueError(f"Impulse diagnostic requires non-silent {label} audio")
    indices = np.flatnonzero(np.isclose(magnitude, peak, rtol=1e-10, atol=1e-12))
    if indices.size != 1:
        raise ValueError(f"Impulse diagnostic requires one unique {label} peak")
    return int(indices[0])


def _normalized_centered(samples: np.ndarray, label: str) -> np.ndarray:
    scale = float(np.max(np.abs(samples)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError(f"Delay diagnostic requires non-silent {label} audio")
    normalized = samples / scale
    centered = normalized - np.mean(normalized)
    if not np.any(np.abs(centered) > np.finfo(np.float64).eps):
        raise ValueError(f"Delay diagnostic requires non-constant {label} audio")
    return centered


def _validate_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    dtype: DTypeLike,
) -> tuple[np.ndarray, np.ndarray]:
    reference_samples = _validate_audio(reference, "reference", dtype=dtype)
    candidate_samples = _validate_audio(candidate, "candidate", dtype=dtype)
    if reference_samples.shape != candidate_samples.shape:
        raise ValueError(
            "reference and candidate arrays must have equal lengths: "
            f"reference={reference_samples.size}, candidate={candidate_samples.size}"
        )
    return reference_samples, candidate_samples


def _validate_audio(
    audio: np.ndarray,
    label: str,
    *,
    dtype: DTypeLike,
) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"{label} audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise ValueError(f"{label} audio must be one-dimensional mono")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"{label} audio must have a real floating dtype")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} audio must contain only finite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.array(audio, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{label} audio is outside the finite {dtype} range")
    return samples


def _require_signed_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)
