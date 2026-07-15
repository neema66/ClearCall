"""Tests for independent improved-DSP gain calculation."""

import numpy as np

from senhance.pipeline.improved_dsp.config import FinalGainSmoothingConfig
from senhance.pipeline.improved_dsp.gains import (
    DecisionDirectedWienerGain,
    soft_fuse_gains,
    spectral_subtraction_gain,
)
from senhance.pipeline.improved_dsp.smoothing import FinalGainSmoother


def test_spectral_subtraction_gain_respects_floor_and_bounds() -> None:
    magnitude = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    noise = np.array([1.0, 10.0, 1.0, 0.0], dtype=np.float32)

    gain = spectral_subtraction_gain(magnitude, noise, 0.3, 0.2)

    assert np.all(gain >= 0.2)
    assert np.all(gain <= 1.0)
    assert np.isclose(gain[1], 0.2)
    assert np.isclose(gain[-1], 1.0)


def test_soft_fusion_endpoints_and_intermediate_order() -> None:
    subtraction = np.array([0.3, 0.7, 1.0], dtype=np.float32)
    wiener = np.array([0.2, 0.5, 0.9], dtype=np.float32)

    spectral_only = soft_fuse_gains(subtraction, wiener, 0.0)
    full_product = soft_fuse_gains(subtraction, wiener, 1.0)
    middle = soft_fuse_gains(subtraction, wiener, 0.5)

    assert np.allclose(spectral_only, subtraction)
    assert np.allclose(full_product, subtraction * wiener)
    assert np.all(middle >= full_product)
    assert np.all(middle <= spectral_only)


def test_wiener_gain_is_finite_bounded_and_resettable() -> None:
    wiener = DecisionDirectedWienerGain(16, smoothing_factor=0.7)
    noisy = np.linspace(0.0, 4.0, 16, dtype=np.float32)
    noise = np.linspace(0.0, 2.0, 16, dtype=np.float32)

    gain = wiener.compute(noisy, noise)

    assert np.all(np.isfinite(gain))
    assert np.all((gain >= 0.0) & (gain <= 1.0))
    assert np.any(wiener._previous_clean_power > 0.0)

    wiener.reset()
    assert np.all(wiener._previous_clean_power == 0.0)


def test_frequency_smoothing_reduces_an_isolated_gain_hole() -> None:
    config = FinalGainSmoothingConfig(
        enabled=True,
        frequency_weights=(0.25, 0.50, 0.25),
        temporal_smoothing=0.0,
    )
    smoother = FinalGainSmoother(7, minimum_gain=0.1, config=config)
    gain = np.ones(7, dtype=np.float32)
    gain[3] = 0.1

    result = smoother.apply(gain)

    assert result[3] > gain[3]
    assert result[2] < gain[2]
    assert result.shape == gain.shape


def test_disabled_smoothing_returns_bounded_input() -> None:
    config = FinalGainSmoothingConfig(enabled=False)
    smoother = FinalGainSmoother(4, minimum_gain=0.2, config=config)

    result = smoother.apply(np.array([-1.0, 0.4, 0.8, 2.0], dtype=np.float32))

    assert np.allclose(result, [0.2, 0.4, 0.8, 1.0])
