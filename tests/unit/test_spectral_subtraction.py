"""Unit tests for the spectral subtraction module."""

import numpy as np

from senhance.pipeline.dsp.spectral_subtraction import spectral_subtract


def test_subtraction_reduces_magnitude_when_noise_present():
    rng = np.random.default_rng(0)
    noisy_spectrum = rng.normal(size=257) + 1j * rng.normal(size=257)
    noise_magnitude = np.abs(noisy_spectrum) * 0.5  # assume noise is half the signal

    result = spectral_subtract(noisy_spectrum, noise_magnitude, oversubtraction_factor=2.0)

    # Output magnitude should generally be <= input magnitude (we're
    # removing energy, not adding it).
    assert np.all(np.abs(result) <= np.abs(noisy_spectrum) + 1e-6)


def test_spectral_floor_prevents_zero_bins():
    noisy_spectrum = np.ones(257, dtype=complex) * 1.0
    noise_magnitude = np.ones(257) * 10.0  # noise estimate much larger than signal

    result = spectral_subtract(
        noisy_spectrum, noise_magnitude, oversubtraction_factor=2.0, spectral_floor=0.1
    )

    # Even with heavy over-subtraction, the floor should prevent bins
    # from being fully zeroed (this is what avoids "musical noise").
    assert np.all(np.abs(result) >= 0.1 * np.abs(noisy_spectrum) - 1e-6)
