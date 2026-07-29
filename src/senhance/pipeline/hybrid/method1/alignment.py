"""Method-1-owned fixed alignment and offline delay diagnostics."""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import DTypeLike
from scipy.signal import correlate, correlation_lags  # type: ignore[import-untyped]


@dataclasses.dataclass(frozen=True)
class AlignedNoisyDLPair:
    """Equal-length noisy/DL arrays after delaying only the faster path."""

    noisy: np.ndarray
    dl: np.ndarray
    dl_minus_noisy_delay_samples: int


@dataclasses.dataclass(frozen=True)
class NoisyDLDelayEstimate:
    """Read-only diagnostic; production alignment never learns per clip."""

    dl_minus_noisy_delay_samples: int
    normalized_peak: float
    max_abs_delay: int | None
    peak_at_search_boundary: bool
    ambiguous_peak: bool


class FixedNoisyDLAligner:
    """Apply a configured delay using ``DL event index - noisy event index``.

    Positive delay means DL lags and the noisy path is delayed. Negative delay
    means DL leads and the DL path is delayed. Both results retain length N.
    """

    def __init__(self, dl_minus_noisy_delay_samples: int) -> None:
        self.delay_samples = _integer(
            dl_minus_noisy_delay_samples,
            "dl_minus_noisy_delay_samples",
        )

    def align(self, noisy: np.ndarray, dl: np.ndarray) -> AlignedNoisyDLPair:
        noisy_samples, dl_samples = _validate_pair(noisy, dl, dtype=np.float32)
        sample_count = noisy_samples.size
        aligned_noisy = noisy_samples.copy()
        aligned_dl = dl_samples.copy()
        if self.delay_samples > 0:
            aligned_noisy.fill(0.0)
            if self.delay_samples < sample_count:
                aligned_noisy[self.delay_samples :] = noisy_samples[
                    : sample_count - self.delay_samples
                ]
        elif self.delay_samples < 0:
            dl_delay = -self.delay_samples
            aligned_dl.fill(0.0)
            if dl_delay < sample_count:
                aligned_dl[dl_delay:] = dl_samples[: sample_count - dl_delay]
        return AlignedNoisyDLPair(
            noisy=np.ascontiguousarray(aligned_noisy, dtype=np.float32),
            dl=np.ascontiguousarray(aligned_dl, dtype=np.float32),
            dl_minus_noisy_delay_samples=self.delay_samples,
        )


def estimate_noisy_dl_delay(
    noisy: np.ndarray,
    dl: np.ndarray,
    *,
    max_abs_delay: int | None = None,
) -> NoisyDLDelayEstimate:
    """Estimate residual DL lag by demeaned cross-correlation for review only."""

    noisy_samples, dl_samples = _validate_pair(noisy, dl, dtype=np.float64)
    if noisy_samples.size == 0:
        raise ValueError("Method 1 delay diagnostic requires non-empty audio")
    noisy_centered = _normalized_centered(noisy_samples, "noisy")
    dl_centered = _normalized_centered(dl_samples, "DL")
    normalization = float(np.linalg.norm(noisy_centered) * np.linalg.norm(dl_centered))
    if not np.isfinite(normalization) or normalization <= np.finfo(np.float64).eps:
        raise ValueError("Method 1 delay diagnostic requires non-silent audio")

    values = correlate(dl_centered, noisy_centered, mode="full", method="fft")
    lags = correlation_lags(dl_centered.size, noisy_centered.size, mode="full")
    limit = None
    if max_abs_delay is not None:
        limit = _integer(max_abs_delay, "max_abs_delay")
        if limit < 0:
            raise ValueError("max_abs_delay must be non-negative")
        keep = np.abs(lags) <= limit
        values = values[keep]
        lags = lags[keep]
        if values.size == 0:
            raise ValueError("max_abs_delay excludes every correlation lag")

    absolute = np.abs(values)
    peak = float(np.max(absolute))
    peak_indices = np.flatnonzero(np.isclose(absolute, peak, rtol=1.0e-10, atol=1.0e-12))
    selected = int(peak_indices[np.argmin(np.abs(lags[peak_indices]))])
    delay = int(lags[selected])
    return NoisyDLDelayEstimate(
        dl_minus_noisy_delay_samples=delay,
        normalized_peak=float(values[selected] / normalization),
        max_abs_delay=limit,
        peak_at_search_boundary=limit is not None and abs(delay) == limit,
        ambiguous_peak=peak_indices.size > 1,
    )


def measure_noisy_dl_impulse_delay(noisy: np.ndarray, dl: np.ndarray) -> int:
    """Return ``DL peak index - noisy peak index`` for a synthetic impulse."""

    noisy_samples, dl_samples = _validate_pair(noisy, dl, dtype=np.float64)
    if noisy_samples.size == 0:
        raise ValueError("Method 1 impulse diagnostic requires non-empty audio")
    return _unique_peak(dl_samples, "DL") - _unique_peak(noisy_samples, "noisy")


def _unique_peak(samples: np.ndarray, label: str) -> int:
    magnitude = np.abs(samples)
    peak = float(np.max(magnitude))
    if peak <= np.finfo(np.float64).eps:
        raise ValueError(f"Method 1 impulse diagnostic requires non-silent {label} audio")
    indices = np.flatnonzero(np.isclose(magnitude, peak, rtol=1.0e-10, atol=1.0e-12))
    if indices.size != 1:
        raise ValueError(f"Method 1 impulse diagnostic requires one unique {label} peak")
    return int(indices[0])


def _normalized_centered(samples: np.ndarray, label: str) -> np.ndarray:
    scale = float(np.max(np.abs(samples)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError(f"Method 1 delay diagnostic requires non-silent {label} audio")
    centered = samples / scale
    centered -= np.mean(centered)
    if not np.any(np.abs(centered) > np.finfo(np.float64).eps):
        raise ValueError(f"Method 1 delay diagnostic requires non-constant {label} audio")
    return centered


def _validate_pair(
    noisy: np.ndarray,
    dl: np.ndarray,
    *,
    dtype: DTypeLike,
) -> tuple[np.ndarray, np.ndarray]:
    noisy_samples = _audio(noisy, "noisy", dtype=dtype)
    dl_samples = _audio(dl, "DL", dtype=dtype)
    if noisy_samples.shape != dl_samples.shape:
        raise ValueError(
            "Method 1 noisy and DL arrays must have equal lengths: "
            f"noisy={noisy_samples.size}, dl={dl_samples.size}"
        )
    return noisy_samples, dl_samples


def _audio(audio: np.ndarray, label: str, *, dtype: DTypeLike) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"Method 1 {label} audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise ValueError(f"Method 1 {label} audio must be one-dimensional mono")
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"Method 1 {label} audio must have a real floating dtype")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Method 1 {label} audio must contain only finite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.array(audio, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"Method 1 {label} audio exceeds the finite numeric range")
    return samples


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)
