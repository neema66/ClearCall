"""Unit tests for the minimum-statistics noise estimator."""

import numpy as np

from senhance.pipeline.dsp.noise_estimator import NoiseEstimator


def test_converges_to_constant_noise_floor():
    """Baseline correctness: given a stationary noise-only signal, the
    estimate should settle at (bias_correction ** 0.5) * true noise level
    -- the same property the old calibration-based estimator guaranteed
    for noise-only input."""
    num_freq_bins = 8
    noise_level = 2.0
    window_frames = 10

    estimator = NoiseEstimator(num_freq_bins=num_freq_bins, window_frames=window_frames)
    magnitude = np.full(num_freq_bins, noise_level, dtype=np.float32)

    estimate = None
    for _ in range(window_frames + 5):
        estimate = estimator.update(magnitude)

    expected = np.sqrt(estimator.bias_correction) * noise_level
    assert np.allclose(estimate, expected, rtol=1e-5)


def test_returns_correct_shape_and_nonnegative():
    num_freq_bins = 12
    estimator = NoiseEstimator(num_freq_bins=num_freq_bins, window_frames=5)
    rng = np.random.default_rng(4)
    magnitude = np.abs(rng.normal(size=num_freq_bins)).astype(np.float32)

    estimate = estimator.update(magnitude)

    assert estimate.shape == (num_freq_bins,)
    assert np.all(estimate >= 0.0)


def test_no_silent_lead_in_still_estimates_noise_floor():
    """
    Regression test for the original bug: the old estimator assumed the
    first `calibration_frames` frames were noise-only, so if speech
    started immediately (frame 0), speech energy got baked into the
    "noise" estimate. The minimum-statistics tracker should find the
    true noise floor even when frame 0 itself is loud speech, as long as
    noise-only frames appear somewhere within the search window.
    """
    num_freq_bins = 16
    noise_level = 1.0
    speech_level = 15.0
    window_frames = 20

    # smoothing=0.0 isolates the minimum-tracking behavior from the
    # exponential smoothing, so the expected value is exact.
    estimator = NoiseEstimator(
        num_freq_bins=num_freq_bins, window_frames=window_frames, smoothing=0.0
    )

    estimate = None
    for i in range(50):
        # Frame 0 is loud speech -- no silent lead-in, this is the bug scenario.
        level = speech_level if i % 4 == 0 else noise_level
        magnitude = np.full(num_freq_bins, level, dtype=np.float32)
        estimate = estimator.update(magnitude)

    expected = np.sqrt(estimator.bias_correction) * noise_level
    assert np.allclose(estimate, expected, rtol=1e-5)
    # Critically, the estimate must not have drifted toward speech level.
    assert np.all(estimate < 0.5 * speech_level)


def test_reset_clears_state():
    estimator = NoiseEstimator(num_freq_bins=8, window_frames=5)
    rng = np.random.default_rng(3)
    for _ in range(10):
        estimator.update(np.abs(rng.normal(size=8)).astype(np.float32))

    estimator.reset()

    assert estimator._frames_seen == 0
    assert estimator._write_index == 0
    assert np.all(estimator._smoothed_power == 0.0)
    assert np.all(estimator._noise_magnitude == 0.0)
    assert np.all(np.isinf(estimator._min_power_history))
