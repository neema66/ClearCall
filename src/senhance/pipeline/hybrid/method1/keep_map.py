"""Numerically safe raw DeepFilterNet keep-map estimation for Method 1."""

from __future__ import annotations

import dataclasses

import numpy as np

from senhance.pipeline.hybrid.method1.config import Method1KeepMapConfig


@dataclasses.dataclass(frozen=True)
class RawKeepMapResult:
    """One bounded gain map plus counts explaining its numerical decisions."""

    gain: np.ndarray
    bin_count: int
    low_energy_neutral_bin_count: int
    clipped_above_one_bin_count: int
    zero_gain_bin_count: int


def estimate_raw_keep_map(
    noisy_spectrum: np.ndarray,
    dl_spectrum: np.ndarray,
    config: Method1KeepMapConfig,
) -> RawKeepMapResult:
    """Estimate ``|DL|/|noisy|`` with unbiased guarded division.

    The literal ``|DL|/(|noisy|+epsilon)`` biases identical nonzero spectra
    below one. Method 1 instead treats epsilon as an activity threshold:
    divide only where noisy magnitude is safely nonzero, initialize every
    other bin to neutral gain one, sanitize, then clamp to ``[0, 1]``.
    """

    if not isinstance(config, Method1KeepMapConfig):
        raise TypeError("config must be Method1KeepMapConfig")
    config.validate()
    noisy = _spectrum(noisy_spectrum, "noisy")
    dl = _spectrum(dl_spectrum, "DL")
    if noisy.shape != dl.shape:
        raise ValueError(
            "Method 1 noisy and DL spectra must have equal shapes: "
            f"noisy={noisy.shape}, dl={dl.shape}"
        )
    noisy_magnitude = np.abs(noisy)
    dl_magnitude = np.abs(dl)
    division_threshold = max(config.epsilon, config.low_energy_threshold)
    active = noisy_magnitude > division_threshold
    neutral = (noisy_magnitude <= config.low_energy_threshold) & (
        dl_magnitude <= config.low_energy_threshold
    )
    above_one = dl_magnitude > noisy_magnitude

    gain = np.ones(noisy.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gain[active] = dl_magnitude[active] / noisy_magnitude[active]
    gain = np.nan_to_num(gain, nan=1.0, posinf=1.0, neginf=0.0)
    np.clip(gain, 0.0, 1.0, out=gain)
    if not np.all(np.isfinite(gain)):
        raise AssertionError("Method 1 keep-map sanitization failed")
    return RawKeepMapResult(
        gain=np.ascontiguousarray(gain),
        bin_count=int(gain.size),
        low_energy_neutral_bin_count=int(np.count_nonzero(neutral)),
        clipped_above_one_bin_count=int(np.count_nonzero(above_one & ~neutral)),
        zero_gain_bin_count=int(np.count_nonzero(gain == 0.0)),
    )


def _spectrum(value: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Method 1 {label} spectrum must be a numpy.ndarray")
    if value.ndim != 1:
        raise ValueError(f"Method 1 {label} spectrum must be one-dimensional")
    if not np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"Method 1 {label} spectrum must be complex floating")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Method 1 {label} spectrum must contain finite values")
    return np.array(value, order="C", copy=True)
