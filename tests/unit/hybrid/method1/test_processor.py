"""Whole-array Method 1, reconstruction-phase, and diagnostic tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from senhance.pipeline.hybrid.method1.processor import (
    Method1SafetyLayer,
    reconstruct_with_gain,
)


@pytest.mark.parametrize(
    "phase_source",
    ["noisy", "dl"],
)
def test_reconstruct_with_gain_selects_requested_phase(phase_source):
    noisy = np.array([1.0 + 1.0j, -2.0j, 1.0 + 0.0j], dtype=np.complex128)
    dl = np.array([-2.0 + 0.0j, 2.0 + 2.0j, 0.0 + 0.0j], dtype=np.complex128)
    gain = np.array([0.5, 0.25, 0.75], dtype=np.float64)

    output = reconstruct_with_gain(
        noisy,
        dl,
        gain,
        phase_source=phase_source,
        low_energy_threshold=1.0e-7,
    )

    noisy_magnitude = np.abs(noisy)
    noisy_phase = noisy / noisy_magnitude
    if phase_source == "noisy":
        expected_phase = noisy_phase
    else:
        expected_phase = np.array(
            [dl[0] / abs(dl[0]), dl[1] / abs(dl[1]), noisy_phase[2]],
            dtype=np.complex128,
        )
    np.testing.assert_allclose(output, gain * noisy_magnitude * expected_phase, atol=1e-15)
    assert output.dtype == np.complex128
    assert output.flags.c_contiguous


def test_dl_phase_falls_back_to_noisy_when_dl_phase_is_undefined():
    noisy = np.array([1.0j, -2.0, 1.0 + 1.0j], dtype=np.complex128)
    dl = np.array([0.0, 1.0e-9j, -1.0j], dtype=np.complex128)
    gain = np.ones(3, dtype=np.float64)

    output = reconstruct_with_gain(
        noisy,
        dl,
        gain,
        phase_source="dl",
        low_energy_threshold=1.0e-7,
    )

    np.testing.assert_allclose(output[:2], noisy[:2], atol=1e-15)
    assert output[2] == pytest.approx(-abs(noisy[2]) * 1.0j)


@pytest.mark.parametrize(
    "noisy,dl,gain,phase,threshold,exception,message",
    [
        (
            [1.0 + 0j],
            np.ones(1, dtype=np.complex128),
            np.ones(1),
            "dl",
            1.0e-7,
            TypeError,
            "numpy.ndarray",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(3, dtype=np.complex128),
            np.ones(2),
            "dl",
            1.0e-7,
            ValueError,
            "equal shapes",
        ),
        (
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.complex128),
            np.ones(2),
            "dl",
            1.0e-7,
            ValueError,
            "complex",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            [1.0, 1.0],
            "dl",
            1.0e-7,
            TypeError,
            "gain",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones((1, 2)),
            "dl",
            1.0e-7,
            TypeError,
            "gain",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones(3),
            "dl",
            1.0e-7,
            ValueError,
            "shape/dtype",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.array([0.0, 1.1]),
            "dl",
            1.0e-7,
            ValueError,
            "bounded",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones(2),
            "other",
            1.0e-7,
            ValueError,
            "phase_source",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones(2),
            "dl",
            -1.0,
            ValueError,
            "non-negative",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones(2),
            "dl",
            True,
            ValueError,
            "non-negative",
        ),
    ],
)
def test_reconstruction_rejects_invalid_contracts(
    noisy,
    dl,
    gain,
    phase,
    threshold,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        reconstruct_with_gain(
            noisy,
            dl,
            gain,
            phase_source=phase,
            low_energy_threshold=threshold,
        )


@pytest.mark.parametrize("length", [0, 1, 479, 480, 481, 959, 960, 961, 1577])
def test_every_variant_preserves_identical_signal_and_exact_array_contract(
    method1_config,
    length,
):
    rng = np.random.default_rng(5000 + length)
    signal = rng.normal(0.0, 0.1, length).astype(np.float32)

    for variant in method1_config.variants:
        result = Method1SafetyLayer(method1_config, variant_id=variant.id).process_array(
            signal,
            signal,
            sample_rate=48_000,
        )
        assert result.audio.shape == signal.shape
        assert result.audio.dtype == np.float32
        assert result.audio.flags.c_contiguous
        assert np.all(np.isfinite(result.audio))
        np.testing.assert_allclose(result.audio, signal, atol=2e-6, rtol=0.0)
        assert result.statistics.variant_id == variant.id
        assert result.statistics.reconstruction_phase == variant.reconstruction_phase
        assert result.statistics.sample_count == length
        assert result.dl_minus_noisy_delay_samples == 0


@pytest.mark.parametrize("variant_id", ["raw_dl_phase", "full_dl_phase", "full_noisy_phase"])
@pytest.mark.parametrize("length", [0, 1, 480, 481, 1440])
def test_silence_is_exact_zero_and_statistics_are_finite(method1_config, variant_id, length):
    silence = np.zeros(length, dtype=np.float32)
    result = Method1SafetyLayer(method1_config, variant_id=variant_id).process_array(
        silence,
        silence,
        sample_rate=48_000,
    )
    np.testing.assert_array_equal(result.audio, silence)
    stats = result.statistics
    for field in (
        "raw_gain_min",
        "raw_gain_max",
        "raw_gain_mean",
        "final_gain_min",
        "final_gain_max",
        "final_gain_mean",
        "peak_abs",
        "clipped_sample_fraction",
    ):
        assert math.isfinite(getattr(stats, field))
    assert stats.peak_abs == 0.0
    assert stats.clipped_sample_count == 0


def test_empty_input_has_no_frames_and_neutral_empty_statistics(method1_config):
    empty = np.zeros(0, dtype=np.float32)
    result = Method1SafetyLayer(method1_config, variant_id="full_dl_phase").process_array(
        empty,
        empty,
        sample_rate=48_000,
    )
    stats = result.statistics
    assert stats.frame_count == 0
    assert stats.gain_bin_count == 0
    assert stats.raw_gain_min == stats.raw_gain_max == stats.raw_gain_mean == 1.0
    assert stats.final_gain_min == stats.final_gain_max == stats.final_gain_mean == 1.0
    assert stats.final_gain_trace_sha256 == (
        "e3b0c44298fc1c149afbf4c8996fb924" "27ae41e4649b934ca495991b7852b855"
    )


def test_raw_phase_variants_make_the_phase_choice_observable(method1_config):
    sample_count = 4_800
    time = np.arange(sample_count, dtype=np.float64) / 48_000.0
    noisy = (0.4 * np.sin(2.0 * np.pi * 1_000.0 * time)).astype(np.float32)
    dl = (-0.5 * noisy).astype(np.float32)

    dl_phase = Method1SafetyLayer(method1_config, variant_id="raw_dl_phase").process_array(
        noisy,
        dl,
        sample_rate=48_000,
    )
    noisy_phase = Method1SafetyLayer(
        method1_config,
        variant_id="raw_noisy_phase",
    ).process_array(noisy, dl, sample_rate=48_000)

    np.testing.assert_allclose(dl_phase.audio, dl, atol=3e-6, rtol=0.0)
    np.testing.assert_allclose(noisy_phase.audio, -dl, atol=3e-6, rtol=0.0)
    assert not np.array_equal(dl_phase.audio, noisy_phase.audio)


@pytest.mark.parametrize("variant_id", ["raw_dl_phase", "raw_noisy_phase"])
def test_raw_reconstruction_caps_dl_amplification_at_noisy_magnitude(
    method1_config,
    variant_id,
):
    rng = np.random.default_rng(433)
    noisy = rng.normal(0.0, 0.1, 2401).astype(np.float32)
    dl = 2.0 * noisy

    result = Method1SafetyLayer(method1_config, variant_id=variant_id).process_array(
        noisy,
        dl,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(result.audio, noisy, atol=2e-6, rtol=0.0)
    assert result.statistics.clipped_above_one_bin_count > 0
    assert not np.allclose(result.audio, dl)


def test_full_variant_reports_gain_activity_and_bounded_final_map(method1_config):
    rng = np.random.default_rng(434)
    noisy = rng.normal(0.0, 0.1, 480 * 30).astype(np.float32)
    # A short moving average is a deterministic frequency-selective surrogate
    # for a DL output. Unlike all-zero DL, it creates a nonuniform raw map so
    # the frequency smoother has real bin-to-bin structure to regularize.
    dl = np.convolve(
        noisy.astype(np.float64),
        np.full(9, 1.0 / 9.0, dtype=np.float64),
        mode="same",
    ).astype(np.float32)

    result = Method1SafetyLayer(method1_config, variant_id="full_dl_phase").process_array(
        noisy,
        dl,
        sample_rate=48_000,
    )
    stats = result.statistics

    assert stats.frame_count == math.ceil(noisy.size / 480) + 1
    assert stats.gain_bin_count == stats.frame_count * 481
    assert stats.raw_gain_min < 0.2
    assert stats.temporal_changed_bin_count > 0
    assert stats.frequency_changed_bin_count > 0
    assert stats.floor_raised_bin_count > 0
    assert 0.0 <= stats.raw_gain_min <= stats.raw_gain_max <= 1.0
    assert method1_config.safety.gain_floor <= stats.final_gain_min
    assert stats.final_gain_max <= 1.0
    assert len(stats.final_gain_trace_sha256) == 64


def test_output_is_not_normalized_or_clipped_and_clipping_is_only_counted(method1_config):
    signal = np.tile(np.array([-1.2, 1.2], dtype=np.float32), 721)
    result = Method1SafetyLayer(method1_config, variant_id="raw_noisy_phase").process_array(
        signal,
        signal,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(result.audio, signal, atol=3e-6, rtol=0.0)
    assert result.statistics.peak_abs == pytest.approx(1.2, abs=3e-6)
    assert result.statistics.clipped_sample_count > 0
    assert result.statistics.clipped_sample_fraction > 0.0


@pytest.mark.parametrize("variant_id", ["raw_dl_phase", "full_dl_phase", "full_noisy_phase"])
def test_repeated_calls_are_bit_identical_and_gain_hashes_reset(method1_config, variant_id):
    rng = np.random.default_rng(435)
    noisy = rng.normal(0.0, 0.1, 1903).astype(np.float32)
    dl = (0.6 * noisy + rng.normal(0.0, 0.005, noisy.size)).astype(np.float32)
    layer = Method1SafetyLayer(method1_config, variant_id=variant_id)

    first = layer.process_array(noisy, dl, sample_rate=48_000)
    second = layer.process_array(noisy, dl, sample_rate=48_000)

    np.testing.assert_array_equal(first.audio, second.audio)
    assert first.statistics == second.statistics


def test_clip_b_after_clip_a_matches_fresh_layer(method1_config):
    rng = np.random.default_rng(436)
    noisy_a = rng.normal(size=1601).astype(np.float32)
    dl_a = (0.4 * noisy_a).astype(np.float32)
    noisy_b = rng.normal(size=997).astype(np.float32)
    dl_b = (-0.7 * noisy_b).astype(np.float32)
    reused = Method1SafetyLayer(method1_config, variant_id="full_dl_phase")
    reused.process_array(noisy_a, dl_a, sample_rate=48_000)
    after_a = reused.process_array(noisy_b, dl_b, sample_rate=48_000)
    fresh = Method1SafetyLayer(method1_config, variant_id="full_dl_phase").process_array(
        noisy_b,
        dl_b,
        sample_rate=48_000,
    )
    np.testing.assert_array_equal(after_a.audio, fresh.audio)
    assert after_a.statistics == fresh.statistics


def test_whole_array_layer_does_not_mutate_or_alias_inputs(method1_config):
    rng = np.random.default_rng(437)
    noisy = rng.normal(size=1001).astype(np.float64)
    dl = rng.normal(size=1001).astype(np.float64)
    noisy_before = noisy.copy()
    dl_before = dl.copy()

    result = Method1SafetyLayer(method1_config, variant_id="full_noisy_phase").process_array(
        noisy,
        dl,
        sample_rate=48_000,
    )

    np.testing.assert_array_equal(noisy, noisy_before)
    np.testing.assert_array_equal(dl, dl_before)
    assert not np.shares_memory(result.audio, noisy)
    assert not np.shares_memory(result.audio, dl)


def test_layer_constructor_and_array_boundary_are_strict(method1_config):
    with pytest.raises(TypeError, match="Method1Config"):
        Method1SafetyLayer(object(), variant_id="full_dl_phase")
    with pytest.raises(ValueError, match="unknown Method 1 variant"):
        Method1SafetyLayer(method1_config, variant_id="missing")

    layer = Method1SafetyLayer(method1_config, variant_id="raw_dl_phase")
    audio = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="equal lengths"):
        layer.process_array(audio, np.zeros(9, dtype=np.float32), sample_rate=48_000)
    with pytest.raises(ValueError, match="sample rate"):
        layer.process_array(audio, audio, sample_rate=16_000)
    with pytest.raises(TypeError, match="sample_rate"):
        layer.process_array(audio, audio, sample_rate=48_000.0)
