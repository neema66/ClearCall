"""
Noise floor estimation.

Implements a simple minimum-statistics-style estimator: assumes the first
N frames of a stream are noise-only (a common simplification for a course
project -- see docs/evaluation_plan.md for how this is validated against
the DNS Challenge dataset, which provides known clean/noisy pairs).

ENSC 429 connection: this is where random signal analysis (power spectral
density estimation of a stochastic noise process) becomes concrete code.

TODO (stretch goal): replace the fixed "first N frames" assumption with a
proper running minimum-statistics tracker (Martin, 2001) that adapts to
non-stationary noise throughout the call, not just at the start.
"""

from __future__ import annotations

import numpy as np


class NoiseEstimator:
    """Tracks an estimate of the noise magnitude spectrum."""

    def __init__(self, num_freq_bins: int, calibration_frames: int = 10, smoothing: float = 0.9):
        """
        Args:
            num_freq_bins: Number of frequency bins per frame (i.e.
                frame_size // 2 + 1 for a real FFT).
            calibration_frames: Number of initial frames treated as
                noise-only to seed the estimate.
            smoothing: Exponential smoothing factor for updating the
                estimate after calibration (closer to 1.0 = slower to adapt).
        """
        self.calibration_frames = calibration_frames
        self.smoothing = smoothing
        self._noise_magnitude = np.zeros(num_freq_bins, dtype=np.float32)
        self._frames_seen = 0

    def update(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        Update the noise estimate given the current frame's magnitude
        spectrum, and return the current best estimate.

        During the calibration period, this simply averages every frame
        in (assuming they're all noise). After calibration, only frames
        that look "noise-like" (see TODO above -- currently this always
        updates, which is a simplification) are blended in.

        Args:
            magnitude_spectrum: |FFT| of the current frame.

        Returns:
            The current noise magnitude estimate (same shape as input).
        """
        if self._frames_seen < self.calibration_frames:
            # Running average during calibration.
            n = self._frames_seen + 1
            self._noise_magnitude = (
                self._noise_magnitude * self._frames_seen + magnitude_spectrum
            ) / n
        else:
            # TODO: only update when a VAD/energy check says this frame is
            # noise-only, not speech. For now we smooth continuously, which
            # is a known simplification appropriate for the Safe Track scope.
            self._noise_magnitude = (
                self.smoothing * self._noise_magnitude
                + (1 - self.smoothing) * magnitude_spectrum
            )

        self._frames_seen += 1
        return self._noise_magnitude

    def reset(self) -> None:
        """Reset the estimator state (call at the start of a new stream)."""
        self._noise_magnitude = np.zeros_like(self._noise_magnitude)
        self._frames_seen = 0
