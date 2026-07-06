"""
Wiener filter noise suppression (decision-directed a priori SNR estimate).

Implements the classic Ephraim & Malah-style decision-directed approach:
estimate the a priori SNR by smoothing between the previous frame's
a posteriori gain and the current frame's instantaneous SNR, then apply
the corresponding Wiener gain to each frequency bin.

Reference:
    Y. Ephraim and D. Malah, "Speech enhancement using a minimum
    mean-square error short-time spectral amplitude estimator," IEEE
    Trans. Acoust., Speech, Signal Process., 1984.
"""

from __future__ import annotations

import numpy as np


class WienerFilter:
    """
    Stateful Wiener filter -- keeps track of the previous frame's gain to
    compute the decision-directed a priori SNR estimate, so this must be
    instantiated once per audio stream (not once per frame).
    """

    def __init__(self, num_freq_bins: int, smoothing_factor: float = 0.98):
        """
        Args:
            num_freq_bins: Number of frequency bins per frame.
            smoothing_factor: Decision-directed smoothing constant
                (typically 0.9-0.99; higher = smoother but slower to react).
        """
        self.smoothing_factor = smoothing_factor
        self._prev_gain = np.ones(num_freq_bins, dtype=np.float32)
        self._prev_clean_power = np.zeros(num_freq_bins, dtype=np.float32)

    def apply(self, noisy_spectrum: np.ndarray, noise_magnitude: np.ndarray) -> np.ndarray:
        """
        Apply the Wiener gain to one frame's spectrum.

        Args:
            noisy_spectrum: Complex FFT spectrum of the noisy frame.
            noise_magnitude: Estimated noise magnitude spectrum.

        Returns:
            Complex FFT spectrum with the Wiener gain applied.
        """
        noisy_power = np.abs(noisy_spectrum) ** 2
        noise_power = noise_magnitude ** 2

        # Instantaneous a posteriori SNR.
        posteriori_snr = noisy_power / (noise_power + 1e-10)
        posteriori_snr = np.maximum(posteriori_snr, 0.0)

        # Decision-directed a priori SNR estimate, blending the previous
        # frame's clean-speech estimate with the current instantaneous SNR.
        priori_snr = self.smoothing_factor * (
            self._prev_clean_power / (noise_power + 1e-10)
        ) + (1 - self.smoothing_factor) * np.maximum(posteriori_snr - 1, 0.0)

        # Wiener gain derived from the a priori SNR.
        gain = priori_snr / (priori_snr + 1)
        gain = np.clip(gain, 0.0, 1.0)

        enhanced_spectrum = gain * noisy_spectrum

        self._prev_gain = gain
        self._prev_clean_power = (gain ** 2) * noisy_power

        return enhanced_spectrum

    def reset(self) -> None:
        """Reset filter state (call at the start of a new stream)."""
        self._prev_gain = np.ones_like(self._prev_gain)
        self._prev_clean_power = np.zeros_like(self._prev_clean_power)
