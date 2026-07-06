"""
Spectral subtraction noise suppression.

Classic algorithm from Boll (1979): estimate the noise magnitude spectrum,
subtract (an oversubtracted multiple of) it from the noisy magnitude
spectrum, floor the result to avoid negative magnitudes, and recombine
with the original (noisy) phase.

Reference:
    S. Boll, "Suppression of acoustic noise in speech using spectral
    subtraction," IEEE Trans. Acoust., Speech, Signal Process., 1979.
"""

from __future__ import annotations

import numpy as np


def spectral_subtract(
    noisy_spectrum: np.ndarray,
    noise_magnitude: np.ndarray,
    oversubtraction_factor: float = 2.0,
    spectral_floor: float = 0.05,
) -> np.ndarray:
    """
    Apply spectral subtraction to one frame's complex spectrum.

    Args:
        noisy_spectrum: Complex FFT spectrum of the noisy frame.
        noise_magnitude: Estimated noise magnitude spectrum (same shape,
            real-valued), typically from NoiseEstimator.update().
        oversubtraction_factor: Alpha -- multiplier on the noise estimate
            before subtracting. Higher values suppress more noise but
            risk removing speech and introducing "musical noise" artifacts.
        spectral_floor: Beta -- minimum gain applied to any bin, as a
            fraction of the original magnitude, to avoid harsh artifacts
            from over-subtraction (bins are never fully zeroed).

    Returns:
        Complex FFT spectrum with noise suppressed, same shape as input,
        ready to be passed to StreamingSTFT.inverse().
    """
    noisy_magnitude = np.abs(noisy_spectrum)
    noisy_phase = np.angle(noisy_spectrum)

    subtracted_magnitude = noisy_magnitude - oversubtraction_factor * noise_magnitude

    # Floor: never let a bin drop below `spectral_floor` times its
    # original magnitude. This is what prevents "musical noise" (isolated
    # bins going to zero and popping in and out between frames).
    floor = spectral_floor * noisy_magnitude
    enhanced_magnitude = np.maximum(subtracted_magnitude, floor)

    return enhanced_magnitude * np.exp(1j * noisy_phase)
