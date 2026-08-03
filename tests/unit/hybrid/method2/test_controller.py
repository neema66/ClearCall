"""Hybrid Method 2 level-controller behavior and safety tests."""

from __future__ import annotations

import numpy as np
import pytest

from senhance.pipeline.hybrid.method2 import DSPGuidedLevelController


def test_equal_branch_levels_leave_dl_waveform_unchanged(method2_config):
    audio = np.linspace(-0.2, 0.2, 101, dtype=np.float32)
    result = DSPGuidedLevelController(method2_config).apply(audio, audio)

    np.testing.assert_array_equal(result.audio, audio)
    assert result.audio.dtype == np.float32
    assert result.statistics.applied_gain == pytest.approx(1.0)
    assert result.statistics.clipped_sample_count == 0


def test_lower_dsp_level_cannot_turn_dl_down(method2_config):
    dl = np.full(100, 0.2, dtype=np.float32)
    dsp = np.full(100, 0.1, dtype=np.float32)
    result = DSPGuidedLevelController(method2_config).apply(dsp, dl)

    np.testing.assert_array_equal(result.audio, dl)
    assert result.statistics.minimum_gain_limited is True
    assert result.statistics.applied_gain == 1.0


def test_ratio_inside_bounds_scales_only_dl_waveform(method2_config):
    dl = np.linspace(-0.2, 0.2, 257, dtype=np.float32)
    dsp = dl * np.float32(1.02)
    result = DSPGuidedLevelController(method2_config).apply(dsp, dl)

    assert result.statistics.applied_gain == pytest.approx(1.02, rel=1.0e-6)
    np.testing.assert_allclose(
        result.audio,
        dl * np.float32(result.statistics.applied_gain),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_maximum_gain_caps_large_dsp_ratio(method2_config):
    dl = np.full(64, 0.1, dtype=np.float32)
    dsp = np.full(64, 0.5, dtype=np.float32)
    result = DSPGuidedLevelController(method2_config).apply(dsp, dl)

    assert result.statistics.maximum_gain_limited is True
    assert result.statistics.applied_gain == pytest.approx(1.03)
    np.testing.assert_allclose(result.audio, dl * np.float32(1.03))


def test_peak_guard_overrides_upward_level_gain(method2_config):
    dl = np.array([1.0, -0.5], dtype=np.float32)
    dsp = dl * np.float32(2.0)
    result = DSPGuidedLevelController(method2_config).apply(dsp, dl)

    assert result.statistics.peak_limited is True
    assert result.statistics.applied_gain == pytest.approx(0.999)
    assert result.statistics.output_peak_abs == pytest.approx(0.999, abs=1.0e-7)
    assert result.statistics.clipped_sample_count == 0


def test_silence_is_finite_and_bypassed(method2_config):
    silence = np.zeros(480, dtype=np.float32)
    result = DSPGuidedLevelController(method2_config).apply(silence, silence)

    np.testing.assert_array_equal(result.audio, silence)
    assert result.statistics.silence_bypassed is True
    assert result.statistics.applied_gain == 1.0
    assert np.isfinite(result.statistics.raw_ratio)


def test_empty_arrays_preserve_shape_and_dtype(method2_config):
    empty = np.zeros(0, dtype=np.float32)
    result = DSPGuidedLevelController(method2_config).apply(empty, empty)

    assert result.audio.shape == (0,)
    assert result.audio.dtype == np.float32
    assert result.statistics.sample_count == 0


@pytest.mark.parametrize(
    ("dsp", "dl", "exception", "message"),
    [
        ([0.0], np.zeros(1, dtype=np.float32), TypeError, "numpy"),
        (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "one-dimensional",
        ),
        (
            np.zeros(1, dtype=np.int16),
            np.zeros(1, dtype=np.float32),
            TypeError,
            "floating",
        ),
        (
            np.array([np.nan], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.zeros(2, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "equal shapes",
        ),
    ],
)
def test_invalid_branch_outputs_are_rejected(method2_config, dsp, dl, exception, message):
    with pytest.raises(exception, match=message):
        DSPGuidedLevelController(method2_config).apply(dsp, dl)
