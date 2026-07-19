"""
Noise floor estimation.

Implements a simplified minimum-statistics-style estimator (Martin, 2001):
a smoothed power spectrum is tracked per frequency bin, and the noise
power estimate is the minimum of that smoothed power over a sliding time
window, continuously updated frame by frame. Because speech power is
almost always >= the underlying noise power, the minimum over a window
long enough to contain at least one noise-only stretch per bin (typically
~1.0-1.5s, covering natural pauses/gaps in speech) tracks the noise floor
without needing an initial silent calibration period.

ENSC 429 connection: this is where random signal analysis (power spectral
density estimation of a stochastic noise process) becomes concrete code.

Reference:
    R. Martin, "Noise power spectral density estimation based on optimal
    smoothing and minimum statistics," IEEE Trans. Speech Audio Process.,
    vol. 9, no. 5, pp. 504-512, 2001.
"""

from __future__ import annotations

import numpy as np


class NoiseEstimator:
    """
    Tracks a running estimate of the noise magnitude spectrum via
    minimum statistics.

    Simplification vs. the full Martin (2001) method: the true algorithm
    derives a time- and frequency-varying bias-correction factor from the
    statistics of the smoothed periodogram (its variance relative to the
    true PSD), since the minimum of a set of random variables is a biased
    (too-low) estimate of their mean. Deriving that adaptive factor is out
    of scope here; instead a single fixed `bias_correction` multiplier is
    applied to the tracked minimum, which is a coarser but much simpler
    approximation appropriate for this project's scope.

    Note: the "correct" theoretical bias_correction (compensating purely
    for the minimum-of-random-variables effect) is > 1.0. In practice this
    estimate also feeds spectral subtraction and the Wiener filter
    downstream, both of which apply their own suppression on top of it --
    so the empirically best-performing value (see docs/evaluation_plan.md)
    ends up < 1.0, deliberately under-correcting at this stage so the
    combined pipeline isn't over-suppressed. This is a system-level
    calibration, not a claim that < 1.0 is the theoretically "correct"
    single-stage bias correction.
    """

    def __init__(
        self,
        num_freq_bins: int,
        window_frames: int = 125,
        smoothing: float = 0.9,
        bias_correction: float = 0.5,
    ):
        """
        Args:
            num_freq_bins: Number of frequency bins per frame (i.e.
                frame_size // 2 + 1 for a real FFT).
            window_frames: Number of past frames over which the running
                minimum is tracked (Martin's "search window"). Longer
                windows are more robust to noise power fluctuations but
                slower to react to a rising noise floor; ~1.0-1.5s worth
                of frames is a reasonable default.
            smoothing: Exponential smoothing factor applied to the power
                spectrum before minimum-tracking (closer to 1.0 = smoother
                but slower to react).
            bias_correction: Fixed multiplier applied to the tracked
                minimum power to compensate for the minimum-of-random-
                variables bias (see class docstring).
        """
        self.window_frames = max(1, window_frames)
        self.smoothing = smoothing
        self.bias_correction = bias_correction
        self._smoothed_power = np.zeros(num_freq_bins, dtype=np.float32)
        # Circular buffer of smoothed power per frame over the search
        # window. Unwritten slots start at +inf so they never win the
        # np.min() below until real frames have filled the window.
        self._min_power_history = np.full(
            (self.window_frames, num_freq_bins), np.inf, dtype=np.float32
        )
        self._write_index = 0
        self._frames_seen = 0
        self._noise_magnitude = np.zeros(num_freq_bins, dtype=np.float32)

    def update(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        Update the noise estimate given the current frame's magnitude
        spectrum, and return the current best estimate.

        Args:
            magnitude_spectrum: |FFT| of the current frame.

        Returns:
            The current noise magnitude estimate (same shape as input).
        """
        power_spectrum = magnitude_spectrum.astype(np.float32) ** 2

        if self._frames_seen == 0:
            # Seed directly from the first frame instead of ramping up
            # from zero, which would otherwise bias the estimate low
            # until the smoother catches up.
            self._smoothed_power = power_spectrum
        else:
            self._smoothed_power = (
                self.smoothing * self._smoothed_power
                + (1 - self.smoothing) * power_spectrum
            )

        self._min_power_history[self._write_index] = self._smoothed_power
        self._write_index = (self._write_index + 1) % self.window_frames
        self._frames_seen += 1

        min_power = np.min(self._min_power_history, axis=0)
        noise_power = min_power * self.bias_correction
        self._noise_magnitude = np.sqrt(noise_power).astype(np.float32)

        return self._noise_magnitude

    def reset(self) -> None:
        """Reset the estimator state (call at the start of a new stream)."""
        self._smoothed_power = np.zeros_like(self._smoothed_power)
        self._min_power_history = np.full_like(self._min_power_history, np.inf)
        self._write_index = 0
        self._frames_seen = 0
        self._noise_magnitude = np.zeros_like(self._noise_magnitude)
