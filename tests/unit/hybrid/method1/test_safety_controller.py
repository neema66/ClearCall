"""Unit tests for every ordered DSP keep-map safety control."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method1.safety_controller import (
    SafetyGainController,
    apply_gain_floor,
    limit_gain_change,
    smooth_gain_across_frequency,
    temporal_exponential_smooth,
)


def test_temporal_smoothing_uses_documented_exponential_equation():
    current = np.array([0.0, 0.2, 1.0], dtype=np.float32)
    previous = np.array([1.0, 0.6, 0.0], dtype=np.float64)

    result = temporal_exponential_smooth(current, previous, 0.75)

    np.testing.assert_allclose(result, 0.75 * previous + 0.25 * current, atol=1e-15)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous


@pytest.mark.parametrize("coefficient", [0.0, 0.25, 0.999])
def test_temporal_smoothing_accepts_full_supported_coefficient_range(coefficient):
    current = np.array([0.2], dtype=np.float64)
    previous = np.array([0.8], dtype=np.float64)
    result = temporal_exponential_smooth(current, previous, coefficient)
    assert result[0] == pytest.approx(coefficient * 0.8 + (1.0 - coefficient) * 0.2)


def test_frequency_smoothing_is_edge_padded_non_circular_and_exact_length():
    gain = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    result = smooth_gain_across_frequency(gain, 3)

    np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1.0, 1.0, 1.0], atol=1e-15)
    assert result.shape == gain.shape
    assert result[-1] == 1.0  # DC edge must not wrap into the Nyquist edge.


def test_frequency_smoothing_reduces_an_isolated_hole_and_preserves_constants():
    hole = np.ones(9, dtype=np.float64)
    hole[4] = 0.0
    smoothed = smooth_gain_across_frequency(hole, 5)
    assert smoothed[4] == pytest.approx(0.8)
    assert smoothed[3] == pytest.approx(0.8)
    assert np.mean(np.abs(np.diff(smoothed))) < np.mean(np.abs(np.diff(hole)))

    constant = np.full(481, 0.37, dtype=np.float64)
    np.testing.assert_allclose(
        smooth_gain_across_frequency(constant, 5),
        constant,
        atol=1e-15,
        rtol=0.0,
    )


def test_frequency_width_one_is_an_independent_copy():
    gain = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    result = smooth_gain_across_frequency(gain, 1)
    np.testing.assert_array_equal(result, gain)
    assert not np.shares_memory(result, gain)


def test_gain_floor_only_raises_values_below_floor():
    gain = np.array([0.0, 0.1, 0.2, 0.9], dtype=np.float64)
    result = apply_gain_floor(gain, 0.2)
    np.testing.assert_array_equal(result, np.array([0.2, 0.2, 0.2, 0.9]))


def test_gain_change_limiter_respects_drop_and_rise_exactly():
    previous = np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float64)
    current = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    result = limit_gain_change(current, previous, max_drop=0.15, max_rise=0.10)

    np.testing.assert_allclose(result, [0.85, 0.35, 0.30, 0.10], atol=1e-15)


def test_raw_variant_bypasses_every_control_without_aliasing(method1_config):
    controller = SafetyGainController(
        method1_config.safety,
        method1_config.variant("raw_dl_phase"),
    )
    raw = np.array([0.0, 0.2, 0.9, 1.0], dtype=np.float32)
    raw_before = raw.copy()

    stages = controller.apply(raw)

    for stage in (
        stages.raw_gain,
        stages.temporal_gain,
        stages.frequency_gain,
        stages.floored_gain,
        stages.final_gain,
    ):
        np.testing.assert_array_equal(stage, raw)
        assert stage.flags.c_contiguous
        assert not np.shares_memory(stage, raw)
    np.testing.assert_array_equal(raw, raw_before)
    assert stages.temporal_changed_bin_count == 0
    assert stages.frequency_changed_bin_count == 0
    assert stages.floor_raised_bin_count == 0
    assert stages.drop_limited_bin_count == 0
    assert stages.rise_limited_bin_count == 0


def test_temporal_controller_starts_from_one_tracks_history_and_resets(method1_config):
    safety = dataclasses.replace(method1_config.safety, temporal_smoothing=0.5)
    controller = SafetyGainController(safety, method1_config.variant("temporal_dl_phase"))
    zero = np.zeros(3, dtype=np.float64)

    first = controller.apply(zero)
    second = controller.apply(zero)
    controller.reset()
    after_reset = controller.apply(zero)

    np.testing.assert_array_equal(first.final_gain, np.full(3, 0.5))
    np.testing.assert_array_equal(second.final_gain, np.full(3, 0.25))
    np.testing.assert_array_equal(after_reset.final_gain, first.final_gain)
    assert first.temporal_changed_bin_count == 3


def test_frequency_and_floor_ablation_stages_report_only_actual_changes(method1_config):
    frequency_safety = dataclasses.replace(
        method1_config.safety,
        temporal_smoothing=0.0,
        frequency_kernel_bins=3,
    )
    frequency_controller = SafetyGainController(
        frequency_safety,
        method1_config.variant("temporal_frequency_dl_phase"),
    )
    stages = frequency_controller.apply(np.array([1.0, 1.0, 0.0, 1.0, 1.0]))
    np.testing.assert_allclose(stages.final_gain, [1.0, 2 / 3, 2 / 3, 2 / 3, 1.0])
    assert stages.temporal_changed_bin_count == 0
    assert stages.frequency_changed_bin_count == 3
    assert stages.floor_raised_bin_count == 0

    floor_safety = dataclasses.replace(
        method1_config.safety,
        temporal_smoothing=0.0,
        frequency_kernel_bins=1,
        gain_floor=0.2,
    )
    floor_controller = SafetyGainController(
        floor_safety,
        method1_config.variant("temporal_frequency_floor_dl_phase"),
    )
    floor_stages = floor_controller.apply(np.array([0.0, 0.1, 0.2, 0.8]))
    np.testing.assert_array_equal(floor_stages.final_gain, [0.2, 0.2, 0.2, 0.8])
    assert floor_stages.floor_raised_bin_count == 2


def test_full_controller_rate_limits_against_previous_final_and_resets(method1_config):
    safety = dataclasses.replace(
        method1_config.safety,
        temporal_smoothing=0.0,
        frequency_kernel_bins=1,
        gain_floor=0.0,
        max_gain_drop_per_frame=0.15,
        max_gain_rise_per_frame=0.10,
    )
    controller = SafetyGainController(safety, method1_config.variant("full_dl_phase"))

    first = controller.apply(np.zeros(5, dtype=np.float64))
    second = controller.apply(np.ones(5, dtype=np.float64))
    controller.reset()
    after_reset = controller.apply(np.zeros(5, dtype=np.float64))

    np.testing.assert_allclose(first.final_gain, 0.85)
    assert first.drop_limited_bin_count == 5
    assert first.rise_limited_bin_count == 0
    np.testing.assert_allclose(second.final_gain, 0.95)
    assert second.drop_limited_bin_count == 0
    assert second.rise_limited_bin_count == 5
    np.testing.assert_array_equal(after_reset.final_gain, first.final_gain)


@pytest.mark.parametrize(
    "function,args,exception,message",
    [
        (temporal_exponential_smooth, (np.ones(2), np.ones(3), 0.5), ValueError, "equal shapes"),
        (temporal_exponential_smooth, (np.ones(2), np.ones(2), True), TypeError, "floating"),
        (temporal_exponential_smooth, (np.ones(2), np.ones(2), 1.0), ValueError, "0.0"),
        (smooth_gain_across_frequency, (np.ones(3), 2), ValueError, "odd"),
        (smooth_gain_across_frequency, (np.ones(3), 5), ValueError, "exceed"),
        (smooth_gain_across_frequency, (np.ones(3), True), TypeError, "integer"),
        (apply_gain_floor, (np.ones(3), 1), TypeError, "floating"),
        (apply_gain_floor, (np.ones(3), -0.1), ValueError, "0.0"),
    ],
)
def test_control_helpers_reject_invalid_parameters(function, args, exception, message):
    with pytest.raises(exception, match=message):
        function(*args)


@pytest.mark.parametrize(
    "gain,exception,message",
    [
        ([0.5], TypeError, "numpy.ndarray"),
        (np.ones((1, 2), dtype=np.float64), ValueError, "one-dimensional"),
        (np.ones(2, dtype=np.complex128), ValueError, "floating"),
        (np.array([np.nan], dtype=np.float64), ValueError, "finite"),
        (np.array([-0.1], dtype=np.float64), ValueError, r"\[0, 1\]"),
        (np.array([1.1], dtype=np.float64), ValueError, r"\[0, 1\]"),
    ],
)
def test_controls_reject_invalid_gain_arrays(gain, exception, message):
    with pytest.raises(exception, match=message):
        smooth_gain_across_frequency(gain, 1)


def test_limit_gain_change_rejects_bad_shapes_and_limits():
    with pytest.raises(ValueError, match="equal shapes"):
        limit_gain_change(np.ones(2), np.ones(3), max_drop=0.1, max_rise=0.1)
    with pytest.raises(TypeError, match="max_drop"):
        limit_gain_change(np.ones(2), np.ones(2), max_drop=1, max_rise=0.1)
    with pytest.raises(ValueError, match="max_rise"):
        limit_gain_change(np.ones(2), np.ones(2), max_drop=0.1, max_rise=1.1)


def test_controller_rejects_wrong_constructor_types(method1_config):
    with pytest.raises(TypeError, match="safety"):
        SafetyGainController(object(), method1_config.variants[0])
    with pytest.raises(TypeError, match="variant"):
        SafetyGainController(method1_config.safety, object())
