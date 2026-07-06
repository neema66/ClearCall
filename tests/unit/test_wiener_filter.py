"""Unit tests for the Wiener filter module."""

import numpy as np

from senhance.pipeline.dsp.wiener_filter import WienerFilter


def test_gain_is_bounded_between_zero_and_one():
    wiener = WienerFilter(num_freq_bins=257)
    rng = np.random.default_rng(1)
    noisy_spectrum = rng.normal(size=257) + 1j * rng.normal(size=257)
    noise_magnitude = np.abs(rng.normal(size=257))

    result = wiener.apply(noisy_spectrum, noise_magnitude)

    # Output magnitude should never exceed input magnitude (gain <= 1).
    assert np.all(np.abs(result) <= np.abs(noisy_spectrum) + 1e-6)


def test_reset_clears_state():
    wiener = WienerFilter(num_freq_bins=257)
    rng = np.random.default_rng(2)
    spectrum = rng.normal(size=257) + 1j * rng.normal(size=257)
    noise = np.abs(rng.normal(size=257))
    wiener.apply(spectrum, noise)

    wiener.reset()
    assert np.all(wiener._prev_gain == 1.0)
    assert np.all(wiener._prev_clean_power == 0.0)
