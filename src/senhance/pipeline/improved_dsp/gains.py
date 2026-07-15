"""Gain estimators and soft fusion for the improved DSP pipeline."""

from __future__ import annotations

import numpy as np


def spectral_subtraction_gain(
    noisy_magnitude: np.ndarray,
    noise_magnitude: np.ndarray,
    oversubtraction_factor: float,
    spectral_floor: float,
) -> np.ndarray:
    """Return a bounded spectral-subtraction keep gain."""

    magnitude = np.asarray(noisy_magnitude, dtype=np.float32)
    noise = np.asarray(noise_magnitude, dtype=np.float32)
    if magnitude.shape != noise.shape:
        raise ValueError("noisy_magnitude and noise_magnitude must have equal shapes")
    if magnitude.ndim != 1:
        raise ValueError("Gain inputs must be one-dimensional spectra")

    gain = np.ones_like(magnitude)
    nonzero = magnitude > np.float32(1e-12)
    gain[nonzero] = (
        magnitude[nonzero] - np.float32(oversubtraction_factor) * noise[nonzero]
    ) / magnitude[nonzero]
    gain = np.nan_to_num(
        gain,
        nan=spectral_floor,
        posinf=1.0,
        neginf=spectral_floor,
    )
    return np.clip(gain, spectral_floor, 1.0).astype(np.float32, copy=False)


class DecisionDirectedWienerGain:
    """Stateful decision-directed Wiener keep-gain estimator."""

    def __init__(self, num_freq_bins: int, smoothing_factor: float) -> None:
        if num_freq_bins <= 0:
            raise ValueError("num_freq_bins must be positive")
        if not 0.0 <= smoothing_factor < 1.0:
            raise ValueError("smoothing_factor must satisfy 0 <= value < 1")

        self.num_freq_bins = num_freq_bins
        self.smoothing_factor = smoothing_factor
        self._previous_clean_power = np.zeros(num_freq_bins, dtype=np.float32)

    def compute(self, noisy_power: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
        """Compute one Wiener gain and update previous-clean-power state."""

        noisy = np.asarray(noisy_power, dtype=np.float32)
        noise = np.asarray(noise_power, dtype=np.float32)
        expected = (self.num_freq_bins,)
        if noisy.shape != expected or noise.shape != expected:
            raise ValueError(
                f"Expected noisy/noise power shape {expected}, got {noisy.shape}/{noise.shape}"
            )
        if np.any(noisy < 0.0) or np.any(noise < 0.0):
            raise ValueError("Power spectra cannot contain negative values")

        denominator = noise + np.float32(1e-10)
        posteriori_snr = np.maximum(noisy / denominator, 0.0)
        alpha = np.float32(self.smoothing_factor)
        priori_snr = alpha * (self._previous_clean_power / denominator) + (
            np.float32(1.0) - alpha
        ) * np.maximum(posteriori_snr - np.float32(1.0), 0.0)

        gain = priori_snr / (priori_snr + np.float32(1.0))
        gain = np.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0)
        gain = np.clip(gain, 0.0, 1.0).astype(np.float32, copy=False)
        self._previous_clean_power = (gain * gain * noisy).astype(np.float32, copy=False)
        return gain

    def reset(self) -> None:
        """Clear the previous-frame clean-power estimate."""

        self._previous_clean_power.fill(0.0)


def soft_fuse_gains(
    subtraction_gain: np.ndarray,
    wiener_gain: np.ndarray,
    fusion_strength: float,
) -> np.ndarray:
    """Retain a configurable fraction of the Wiener attenuation."""

    subtraction = np.asarray(subtraction_gain, dtype=np.float32)
    wiener = np.asarray(wiener_gain, dtype=np.float32)
    if subtraction.shape != wiener.shape:
        raise ValueError("subtraction_gain and wiener_gain must have equal shapes")
    if subtraction.ndim != 1:
        raise ValueError("Gain inputs must be one-dimensional spectra")
    if not 0.0 <= fusion_strength <= 1.0:
        raise ValueError("fusion_strength must satisfy 0 <= value <= 1")

    strength = np.float32(fusion_strength)
    fused = subtraction * ((np.float32(1.0) - strength) + strength * wiener)
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)
