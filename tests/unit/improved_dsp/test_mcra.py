"""Tests for MCRA noise-power estimation."""

import numpy as np

from senhance.pipeline.improved_dsp.config import MCRAConfig
from senhance.pipeline.improved_dsp.mcra import MCRANoiseEstimator


def test_stationary_noise_converges_to_calibrated_power() -> None:
    config = MCRAConfig(power_calibration=0.5)
    estimator = MCRANoiseEstimator(8, minimum_window_frames=10, config=config)
    power = np.full(8, 4.0, dtype=np.float32)

    for _ in range(30):
        estimate = estimator.update(power)

    assert np.allclose(estimate.noise_power, 2.0, atol=1e-5)
    assert np.all(estimate.speech_presence_probability == 0.0)


def test_speech_burst_is_not_immediately_learned_as_noise() -> None:
    estimator = MCRANoiseEstimator(9, minimum_window_frames=20)
    noise = np.ones(9, dtype=np.float32)
    for _ in range(15):
        estimator.update(noise)

    burst = noise.copy()
    burst[4] = 100.0
    estimate = estimator.update(burst)

    assert estimate.speech_presence_probability[4] > 0.5
    assert estimate.noise_power[4] < 5.0


def test_sustained_noise_step_is_eventually_tracked() -> None:
    estimator = MCRANoiseEstimator(8, minimum_window_frames=8)
    low = np.ones(8, dtype=np.float32)
    high = np.full(8, 9.0, dtype=np.float32)
    for _ in range(20):
        estimator.update(low)
    before = estimator.update(low).noise_power.copy()

    for _ in range(80):
        after = estimator.update(high).noise_power

    assert np.all(after > before * 2.0)
    assert np.all(after <= 4.5 + 1e-4)


def test_probability_is_bounded_and_reset_clears_state() -> None:
    estimator = MCRANoiseEstimator(8, minimum_window_frames=5)
    rng = np.random.default_rng(12)
    for _ in range(30):
        power = np.square(rng.normal(size=8)).astype(np.float32)
        estimate = estimator.update(power)
        assert np.all((estimate.speech_presence_probability >= 0.0))
        assert np.all((estimate.speech_presence_probability <= 1.0))
        assert np.all(np.isfinite(estimate.noise_power))

    estimator.reset()

    assert estimator._frames_seen == 0
    assert estimator._frames_in_subwindow == 0
    assert np.all(estimator._noise_power == 0.0)
    assert np.all(estimator._speech_probability == 0.0)
