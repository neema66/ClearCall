"""Method 3 Version 1 fixed-waveform contract and safety tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method3.alignment import FixedDelayAligner
from senhance.pipeline.hybrid.method3.blender import (
    FixedWaveformBlender,
    blend_aligned_waveforms,
)
from senhance.pipeline.hybrid.method3.config import HybridAlignmentConfig, HybridConfig
from senhance.pipeline.hybrid.method3.method3_config import load_method3_config


def _hybrid_config(delay_samples: int = 0) -> HybridConfig:
    return HybridConfig(
        alignment=HybridAlignmentConfig(delay_samples=delay_samples),
    )


@pytest.mark.parametrize("delay", [-7, 0, 7])
@pytest.mark.parametrize("alpha, selected", [(0.0, "dsp"), (1.0, "dl")])
def test_endpoints_are_bit_exact_causally_aligned_copies(delay, alpha, selected):
    dsp = np.linspace(-0.75, 0.75, 101, dtype=np.float32)
    dl = np.linspace(0.6, -0.6, 101, dtype=np.float32)
    aligned = FixedDelayAligner(delay).align(dsp, dl)
    expected = aligned.reference if selected == "dsp" else aligned.candidate

    result = FixedWaveformBlender(
        _hybrid_config(delay),
        alpha=alpha,
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, expected)
    assert result.audio is not expected
    assert result.delay_samples == delay
    assert result.statistics.alpha == alpha


def test_alpha_half_is_exact_documented_float64_then_float32_arithmetic():
    dsp = np.array([-1.0, -0.2, 0.1, 0.8], dtype=np.float32)
    dl = np.array([0.5, -0.4, 0.9, -0.6], dtype=np.float32)
    expected = (0.5 * dl.astype(np.float64) + 0.5 * dsp.astype(np.float64)).astype(np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0.5,
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, expected)


def test_every_configured_alpha_uses_the_fixed_waveform_equation():
    experiment = load_method3_config("config/hybrid_method_3.yaml")
    rng = np.random.default_rng(429)
    dsp = rng.normal(0.0, 0.2, 997).astype(np.float32)
    dl = rng.normal(0.0, 0.2, 997).astype(np.float32)

    for alpha in experiment.alpha_sweep:
        result = FixedWaveformBlender(
            _hybrid_config(),
            alpha=alpha,
            clipping_threshold=experiment.clipping_threshold,
        ).process_array(dsp, dl, sample_rate=48_000)
        if alpha == 0.0:
            expected = dsp
        elif alpha == 1.0:
            expected = dl
        else:
            expected = (
                alpha * dl.astype(np.float64) + (1.0 - alpha) * dsp.astype(np.float64)
            ).astype(np.float32)
        np.testing.assert_array_equal(result.audio, expected)


@pytest.mark.parametrize("length", [0, 1, 479, 480, 481, 959, 960, 961, 1577])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.7, 1.0])
def test_arbitrary_lengths_are_finite_contiguous_float32(length, alpha):
    rng = np.random.default_rng(1000 + length)
    dsp = rng.normal(0.0, 0.1, length).astype(np.float64)
    dl = rng.normal(0.0, 0.1, length).astype(np.float64)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=alpha,
    ).process_array(dsp, dl, sample_rate=48_000)

    assert result.audio.shape == (length,)
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert np.all(np.isfinite(result.audio))
    assert result.statistics.sample_count == length
    assert np.isfinite(result.statistics.peak_abs)


def test_empty_audio_has_zero_diagnostics():
    empty = np.zeros(0, dtype=np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0.5,
    ).process_array(empty, empty, sample_rate=48_000)

    assert result.statistics.peak_abs == 0.0
    assert result.statistics.clipped_sample_count == 0
    assert result.statistics.clipped_sample_fraction == 0.0


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.7, 1.0])
def test_silence_remains_exact_silence(alpha):
    silence = np.zeros(777, dtype=np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=alpha,
    ).process_array(silence, silence, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, silence)


def test_clipping_is_counted_at_both_boundaries_without_modifying_output():
    audio = np.array([-1.25, -1.0, -0.999, 0.0, 0.999, 1.0, 1.25], dtype=np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0,
    ).process_array(audio, np.zeros_like(audio), sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, audio)
    assert result.statistics.peak_abs == pytest.approx(1.25)
    assert result.statistics.clipping_threshold == 1.0
    assert result.statistics.clipped_sample_count == 4
    assert result.statistics.clipped_sample_fraction == pytest.approx(4 / 7)


def test_custom_clipping_threshold_is_diagnostic_only():
    audio = np.array([-0.5, -0.49, 0.49, 0.5], dtype=np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=1.0,
        clipping_threshold=0.5,
    ).process_array(np.zeros_like(audio), audio, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, audio)
    assert result.statistics.clipped_sample_count == 2


def test_configured_alignment_prevents_cancellation_on_delayed_tone():
    delay = 9
    time = np.arange(1000, dtype=np.float64)
    dsp = np.sin(2.0 * np.pi * 1000.0 * time / 48_000.0).astype(np.float32)
    dl = np.zeros_like(dsp)
    dl[delay:] = dsp[:-delay]
    aligned = FixedDelayAligner(delay).align(dsp, dl)

    result = FixedWaveformBlender(
        _hybrid_config(delay),
        alpha=0.5,
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_array_equal(aligned.reference, aligned.candidate)
    np.testing.assert_array_equal(result.audio, aligned.candidate)
    assert np.argmax(np.abs(result.audio)) == np.argmax(np.abs(aligned.candidate))


@pytest.mark.parametrize("delay", [-9, 0, 9])
def test_impulse_has_expected_aligned_onset_and_no_length_change(delay):
    dsp = np.zeros(100, dtype=np.float32)
    dl = np.zeros(100, dtype=np.float32)
    dsp[30] = 0.8
    dl[30 + delay] = 0.8

    result = FixedWaveformBlender(
        _hybrid_config(delay),
        alpha=0.5,
    ).process_array(dsp, dl, sample_rate=48_000)

    assert result.audio.shape == dsp.shape
    assert int(np.argmax(np.abs(result.audio))) == 30 + max(delay, 0)
    assert result.audio[30 + max(delay, 0)] == pytest.approx(0.8)


def test_opposite_polarity_probe_reports_expected_arithmetic_cancellation():
    dsp = np.linspace(-0.8, 0.8, 100, dtype=np.float32)
    dl = -dsp

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0.5,
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, np.zeros_like(dsp))


def test_repeated_processing_is_bit_identical_and_stateless():
    rng = np.random.default_rng(88)
    dsp = rng.normal(size=1703).astype(np.float32)
    dl = rng.normal(size=1703).astype(np.float32)
    blender = FixedWaveformBlender(_hybrid_config(), alpha=0.7)

    first = blender.process_array(dsp, dl, sample_rate=48_000)
    second = blender.process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_array_equal(first.audio, second.audio)
    assert first.statistics == second.statistics


def test_processing_does_not_mutate_or_alias_caller_arrays():
    dsp = np.linspace(-0.2, 0.2, 500, dtype=np.float32)
    dl = -dsp
    dsp_before = dsp.copy()
    dl_before = dl.copy()

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0.0,
    ).process_array(dsp, dl, sample_rate=48_000)
    result.audio[0] = 0.123

    np.testing.assert_array_equal(dsp, dsp_before)
    np.testing.assert_array_equal(dl, dl_before)


@pytest.mark.parametrize("alpha", [0, 1, np.int64(0), np.float32(0.5)])
def test_common_real_numeric_alpha_scalars_are_accepted(alpha):
    audio = np.zeros(4, dtype=np.float32)

    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=alpha,
    ).process_array(audio, audio, sample_rate=48_000)

    assert 0.0 <= result.statistics.alpha <= 1.0


@pytest.mark.parametrize("alpha", [True, np.bool_(False), "0.5", 0.5 + 0j, None])
def test_nonreal_or_boolean_alpha_is_rejected(alpha):
    with pytest.raises(TypeError, match="alpha must be a real numeric scalar"):
        FixedWaveformBlender(_hybrid_config(), alpha=alpha)


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_out_of_range_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="0.0 <= alpha <= 1.0"):
        FixedWaveformBlender(_hybrid_config(), alpha=alpha)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must be finite"):
        FixedWaveformBlender(_hybrid_config(), alpha=alpha)


@pytest.mark.parametrize(
    "threshold, exception, message",
    [
        (True, TypeError, "real numeric scalar"),
        ("1.0", TypeError, "real numeric scalar"),
        (0.0, ValueError, "must be positive"),
        (-1.0, ValueError, "must be positive"),
        (float("inf"), ValueError, "must be finite"),
    ],
)
def test_invalid_clipping_threshold_is_rejected(threshold, exception, message):
    with pytest.raises(exception, match=message):
        FixedWaveformBlender(
            _hybrid_config(),
            alpha=0.5,
            clipping_threshold=threshold,
        )


@pytest.mark.parametrize(
    "dsp, dl, exception, message",
    [
        (np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32), ValueError, "equal"),
        (np.zeros((1, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), ValueError, "mono"),
        (np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (np.zeros(4, dtype=np.complex64), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (
            np.array([np.nan], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_audio_is_rejected(dsp, dl, exception, message):
    with pytest.raises(exception, match=message):
        FixedWaveformBlender(
            _hybrid_config(),
            alpha=0.5,
        ).process_array(dsp, dl, sample_rate=48_000)


def test_float32_input_overflow_is_rejected():
    huge = np.array([np.finfo(np.float64).max], dtype=np.float64)

    with pytest.raises(ValueError, match="finite.*range"):
        FixedWaveformBlender(
            _hybrid_config(),
            alpha=0.5,
        ).process_array(huge, huge, sample_rate=48_000)


@pytest.mark.parametrize("sample_rate", [16_000, 47_999, 48_001])
def test_wrong_sample_rate_is_rejected(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="must be 48000 Hz"):
        FixedWaveformBlender(
            _hybrid_config(),
            alpha=0.5,
        ).process_array(audio, audio, sample_rate=sample_rate)


@pytest.mark.parametrize("sample_rate", [48_000.0, True, "48000"])
def test_noninteger_sample_rate_is_rejected(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(TypeError, match="sample_rate must be an integer"):
        FixedWaveformBlender(
            _hybrid_config(),
            alpha=0.5,
        ).process_array(audio, audio, sample_rate=sample_rate)


def test_low_level_blend_validates_delay_metadata():
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(TypeError, match="delay_samples must be an integer"):
        blend_aligned_waveforms(audio, audio, alpha=0.5, delay_samples=True)


def test_blender_requires_hybrid_config():
    with pytest.raises(TypeError, match="config must be HybridConfig"):
        FixedWaveformBlender(None, alpha=0.5)


def test_result_metadata_is_frozen():
    audio = np.zeros(4, dtype=np.float32)
    result = FixedWaveformBlender(
        _hybrid_config(),
        alpha=0.5,
    ).process_array(audio, audio, sample_rate=48_000)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.statistics.alpha = 0.1
