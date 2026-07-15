"""MCRA-style speech-presence-aware noise-power estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senhance.pipeline.improved_dsp.config import MCRAConfig


@dataclass(frozen=True)
class MCRAEstimate:
    """Per-frame outputs from the MCRA estimator."""

    noise_power: np.ndarray
    speech_presence_probability: np.ndarray


class MCRANoiseEstimator:
    """Causal minima-controlled recursive averaging noise estimator."""

    _EPSILON = np.float32(1e-12)

    def __init__(
        self,
        num_freq_bins: int,
        minimum_window_frames: int,
        config: MCRAConfig | None = None,
    ) -> None:
        if num_freq_bins <= 0:
            raise ValueError("num_freq_bins must be positive")
        if minimum_window_frames <= 0:
            raise ValueError("minimum_window_frames must be positive")

        self.num_freq_bins = num_freq_bins
        self.minimum_window_frames = minimum_window_frames
        self.config = config or MCRAConfig()

        self._smoothed_power = np.zeros(num_freq_bins, dtype=np.float32)
        self._minimum_power = np.zeros(num_freq_bins, dtype=np.float32)
        self._temporary_minimum = np.zeros(num_freq_bins, dtype=np.float32)
        self._noise_power = np.zeros(num_freq_bins, dtype=np.float32)
        self._speech_probability = np.zeros(num_freq_bins, dtype=np.float32)
        self._frames_seen = 0
        self._frames_in_subwindow = 0

    def update(self, noisy_power: np.ndarray) -> MCRAEstimate:
        """Update the estimate from one nonnegative noisy power spectrum."""

        power = np.asarray(noisy_power, dtype=np.float32)
        if power.ndim != 1 or power.shape[0] != self.num_freq_bins:
            raise ValueError(
                f"Expected power spectrum shape ({self.num_freq_bins},), got {power.shape}"
            )
        if not np.all(np.isfinite(power)):
            raise ValueError("noisy_power must contain only finite values")
        if np.any(power < 0.0):
            raise ValueError("noisy_power cannot contain negative values")

        local_power = self._smooth_frequency(power)

        if self._frames_seen == 0:
            self._smoothed_power = local_power.copy()
            self._minimum_power = local_power.copy()
            self._temporary_minimum = local_power.copy()
            self._noise_power = power.copy()
            self._speech_probability.fill(0.0)
            self._frames_seen = 1
            self._frames_in_subwindow = 1
            return self._current_estimate()

        alpha_s = np.float32(self.config.power_smoothing)
        self._smoothed_power = (
            alpha_s * self._smoothed_power + (np.float32(1.0) - alpha_s) * local_power
        ).astype(np.float32, copy=False)

        if self._frames_in_subwindow >= self.minimum_window_frames:
            self._minimum_power = np.minimum(self._temporary_minimum, self._smoothed_power)
            self._temporary_minimum = self._smoothed_power.copy()
            self._frames_in_subwindow = 1
        else:
            self._minimum_power = np.minimum(self._minimum_power, self._smoothed_power)
            self._temporary_minimum = np.minimum(self._temporary_minimum, self._smoothed_power)
            self._frames_in_subwindow += 1

        ratio = self._smoothed_power / np.maximum(self._minimum_power, self._EPSILON)
        indicator = (ratio > self.config.speech_ratio_threshold).astype(np.float32)
        alpha_p = np.float32(self.config.speech_probability_smoothing)
        self._speech_probability = (
            alpha_p * self._speech_probability + (np.float32(1.0) - alpha_p) * indicator
        ).astype(np.float32, copy=False)
        np.clip(self._speech_probability, 0.0, 1.0, out=self._speech_probability)

        alpha_d = np.float32(self.config.noise_smoothing)
        controlled_smoothing = alpha_d + (np.float32(1.0) - alpha_d) * self._speech_probability
        self._noise_power = (
            controlled_smoothing * self._noise_power
            + (np.float32(1.0) - controlled_smoothing) * power
        ).astype(np.float32, copy=False)
        np.maximum(self._noise_power, 0.0, out=self._noise_power)

        self._frames_seen += 1
        return self._current_estimate()

    def _smooth_frequency(self, power: np.ndarray) -> np.ndarray:
        smoothed = power.copy()
        if self.num_freq_bins < 3:
            return smoothed

        left, center, right = self.config.local_frequency_weights
        smoothed[1:-1] = (
            np.float32(left) * power[:-2]
            + np.float32(center) * power[1:-1]
            + np.float32(right) * power[2:]
        )
        return smoothed

    def _current_estimate(self) -> MCRAEstimate:
        calibrated = (np.float32(self.config.power_calibration) * self._noise_power).astype(
            np.float32, copy=False
        )
        calibrated = np.maximum(calibrated, 0.0)
        return MCRAEstimate(
            noise_power=calibrated,
            speech_presence_probability=self._speech_probability,
        )

    def reset(self) -> None:
        """Clear all stream-dependent estimator state."""

        self._smoothed_power.fill(0.0)
        self._minimum_power.fill(0.0)
        self._temporary_minimum.fill(0.0)
        self._noise_power.fill(0.0)
        self._speech_probability.fill(0.0)
        self._frames_seen = 0
        self._frames_in_subwindow = 0
