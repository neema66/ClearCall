"""Stateful DSP keep-map controls owned only by Hybrid Method 1."""

from __future__ import annotations

import dataclasses

import numpy as np

from senhance.pipeline.hybrid.method1.config import (
    Method1SafetyConfig,
    Method1VariantConfig,
)


@dataclasses.dataclass(frozen=True)
class GainStageResult:
    """All cumulative gain stages for testing, ablation, and diagnostics."""

    raw_gain: np.ndarray
    temporal_gain: np.ndarray
    frequency_gain: np.ndarray
    floored_gain: np.ndarray
    final_gain: np.ndarray
    temporal_changed_bin_count: int
    frequency_changed_bin_count: int
    floor_raised_bin_count: int
    drop_limited_bin_count: int
    rise_limited_bin_count: int


class SafetyGainController:
    """Apply the selected cumulative safety stages with explicit neutral state.

    Temporal and rate-limit histories initialize to all-one gain. This
    conservative startup avoids suddenly suppressing the beginning of a clip.
    ``reset`` restores that behavior deterministically.
    """

    def __init__(
        self,
        safety: Method1SafetyConfig,
        variant: Method1VariantConfig,
    ) -> None:
        if not isinstance(safety, Method1SafetyConfig):
            raise TypeError("safety must be Method1SafetyConfig")
        if not isinstance(variant, Method1VariantConfig):
            raise TypeError("variant must be Method1VariantConfig")
        safety.validate()
        variant.validate()
        self.safety = safety
        self.variant = variant
        self._previous_temporal: np.ndarray | None = None
        self._previous_final: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_temporal = None
        self._previous_final = None

    def apply(self, raw_gain: np.ndarray) -> GainStageResult:
        raw = _gain(raw_gain, "raw_gain")
        if self.variant.temporal_smoothing:
            previous_temporal = (
                np.ones_like(raw) if self._previous_temporal is None else self._previous_temporal
            )
            temporal = temporal_exponential_smooth(
                raw,
                previous_temporal,
                self.safety.temporal_smoothing,
            )
            self._previous_temporal = temporal.copy()
        else:
            temporal = raw.copy()

        if self.variant.frequency_smoothing:
            frequency = smooth_gain_across_frequency(
                temporal,
                self.safety.frequency_kernel_bins,
            )
        else:
            frequency = temporal.copy()

        if self.variant.gain_floor:
            floored = apply_gain_floor(frequency, self.safety.gain_floor)
        else:
            floored = frequency.copy()

        drop_limited = np.zeros(raw.shape, dtype=bool)
        rise_limited = np.zeros(raw.shape, dtype=bool)
        if self.variant.rate_limits:
            previous_final = (
                np.ones_like(raw) if self._previous_final is None else self._previous_final
            )
            lower = np.maximum(
                0.0,
                previous_final - self.safety.max_gain_drop_per_frame,
            )
            upper = np.minimum(
                1.0,
                previous_final + self.safety.max_gain_rise_per_frame,
            )
            drop_limited = floored < lower
            rise_limited = floored > upper
            final = np.minimum(np.maximum(floored, lower), upper)
            self._previous_final = final.copy()
        else:
            final = floored.copy()

        for value in (temporal, frequency, floored, final):
            np.clip(value, 0.0, 1.0, out=value)
            if not np.all(np.isfinite(value)):
                raise ValueError("Method 1 safety controller produced non-finite gains")
        return GainStageResult(
            raw_gain=np.ascontiguousarray(raw),
            temporal_gain=np.ascontiguousarray(temporal),
            frequency_gain=np.ascontiguousarray(frequency),
            floored_gain=np.ascontiguousarray(floored),
            final_gain=np.ascontiguousarray(final),
            temporal_changed_bin_count=_changed(raw, temporal),
            frequency_changed_bin_count=_changed(temporal, frequency),
            floor_raised_bin_count=int(np.count_nonzero(floored > frequency)),
            drop_limited_bin_count=int(np.count_nonzero(drop_limited)),
            rise_limited_bin_count=int(np.count_nonzero(rise_limited)),
        )


def temporal_exponential_smooth(
    current: np.ndarray,
    previous: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    current_gain = _gain(current, "current")
    previous_gain = _gain(previous, "previous")
    if current_gain.shape != previous_gain.shape:
        raise ValueError("current and previous gains must have equal shapes")
    if isinstance(coefficient, bool) or not isinstance(coefficient, float):
        raise TypeError("temporal coefficient must be a floating-point value")
    if not np.isfinite(coefficient) or coefficient < 0.0 or coefficient >= 1.0:
        raise ValueError("temporal coefficient must satisfy 0.0 <= value < 1.0")
    result = coefficient * previous_gain + (1.0 - coefficient) * current_gain
    return np.ascontiguousarray(result, dtype=np.float64)


def smooth_gain_across_frequency(gain: np.ndarray, kernel_bins: int) -> np.ndarray:
    values = _gain(gain, "gain")
    if isinstance(kernel_bins, bool) or not isinstance(kernel_bins, (int, np.integer)):
        raise TypeError("kernel_bins must be an integer")
    width = int(kernel_bins)
    if width <= 0 or width % 2 == 0:
        raise ValueError("kernel_bins must be a positive odd integer")
    if width == 1 or values.size == 0:
        return values.copy()
    if width > values.size:
        raise ValueError("kernel_bins cannot exceed the gain-map bin count")
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.shape != values.shape:
        raise AssertionError("frequency smoother changed gain-map length")
    return np.ascontiguousarray(smoothed, dtype=np.float64)


def apply_gain_floor(gain: np.ndarray, floor: float) -> np.ndarray:
    values = _gain(gain, "gain")
    if isinstance(floor, bool) or not isinstance(floor, float):
        raise TypeError("gain floor must be a floating-point value")
    if not np.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError("gain floor must satisfy 0.0 <= value <= 1.0")
    return np.ascontiguousarray(np.maximum(values, floor), dtype=np.float64)


def limit_gain_change(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    max_drop: float,
    max_rise: float,
) -> np.ndarray:
    current_gain = _gain(current, "current")
    previous_gain = _gain(previous, "previous")
    if current_gain.shape != previous_gain.shape:
        raise ValueError("current and previous gains must have equal shapes")
    for name, value in (("max_drop", max_drop), ("max_rise", max_rise)):
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(f"{name} must be a floating-point value")
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must satisfy 0.0 <= value <= 1.0")
    lower = np.maximum(0.0, previous_gain - max_drop)
    upper = np.minimum(1.0, previous_gain + max_rise)
    return np.ascontiguousarray(
        np.minimum(np.maximum(current_gain, lower), upper),
        dtype=np.float64,
    )


def _gain(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be a one-dimensional floating array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return result


def _changed(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isclose(left, right, rtol=0.0, atol=1.0e-15)))
