"""Method 3 Version 2 fixed-frequency-band wrapper acceptance tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method3.alignment import FixedDelayAligner
from senhance.pipeline.hybrid.method3.band_blender import FixedFrequencyBandBlender
from senhance.pipeline.hybrid.method3.blender import FixedWaveformBlender
from senhance.pipeline.hybrid.method3.config import HybridAlignmentConfig, HybridConfig
from senhance.pipeline.hybrid.method3.method3_band_config import (
    BandBlendVariantConfig,
    Method3BandConfig,
    load_method3_band_config,
)


METHOD_CONFIG = load_method3_band_config("config/hybrid_method_3_bands.yaml")
VARIANT_IDS = tuple(variant.id for variant in METHOD_CONFIG.variants)
COMPLEX_VARIANT_IDS = tuple(
    variant.id for variant in METHOD_CONFIG.variants if variant.blend_domain == "complex"
)
BOUNDARY_LENGTHS = (0, 1, 479, 480, 481, 959, 960, 961, 1577)


def _hybrid_config(delay_samples: int = 0) -> HybridConfig:
    return HybridConfig(alignment=HybridAlignmentConfig(delay_samples=delay_samples))


def _method_config(
    *,
    dl_weight: float | None = None,
    clipping_threshold: float | None = None,
) -> Method3BandConfig:
    config = METHOD_CONFIG
    if dl_weight is not None:
        config = dataclasses.replace(
            config,
            dl_weights=tuple(dl_weight for _ in config.dl_weights),
        )
    if clipping_threshold is not None:
        config = dataclasses.replace(
            config,
            evaluation=dataclasses.replace(
                config.evaluation,
                clipping_threshold=clipping_threshold,
            ),
        )
    config.validate()
    return config


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
@pytest.mark.parametrize("endpoint_weight, selected", [(0.0, "dsp"), (1.0, "dl")])
@pytest.mark.parametrize("delay", [-7, 0, 7])
@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_every_spectral_mode_endpoint_reconstructs_aligned_boundary_lengths(
    variant_id,
    endpoint_weight,
    selected,
    delay,
    length,
):
    rng = np.random.default_rng(4_500 + length + delay)
    dsp = rng.normal(0.0, 0.15, length).astype(np.float32)
    dl = rng.normal(0.0, 0.15, length).astype(np.float32)
    aligned = FixedDelayAligner(delay).align(dsp, dl)
    expected = aligned.reference if selected == "dsp" else aligned.candidate

    result = FixedFrequencyBandBlender(
        _hybrid_config(delay),
        _method_config(dl_weight=endpoint_weight),
        variant_id=variant_id,
    ).process_array(dsp, dl, sample_rate=48_000)

    assert result.audio.shape == (length,)
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert np.all(np.isfinite(result.audio))
    np.testing.assert_allclose(result.audio, expected, atol=2e-6, rtol=0.0)
    assert result.delay_samples == delay
    assert result.statistics.sample_count == length
    assert result.statistics.minimum_dl_weight == endpoint_weight
    assert result.statistics.maximum_dl_weight == endpoint_weight


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
@pytest.mark.parametrize("delay", [-9, 0, 9])
@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_configured_fixed_band_wrapper_preserves_boundary_lengths(
    variant_id,
    delay,
    length,
):
    rng = np.random.default_rng(5_000 + length + delay)
    dsp = rng.normal(0.0, 0.1, length).astype(np.float64)
    dl = rng.normal(0.0, 0.1, length).astype(np.float64)

    result = FixedFrequencyBandBlender(
        _hybrid_config(delay),
        METHOD_CONFIG,
        variant_id=variant_id,
    ).process_array(dsp, dl, sample_rate=np.int64(48_000))

    assert result.audio.shape == (length,)
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert np.all(np.isfinite(result.audio))


@pytest.mark.parametrize("variant_id", COMPLEX_VARIANT_IDS)
@pytest.mark.parametrize("delay", [-5, 0, 5])
@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_uniform_half_complex_blend_matches_waveform_blend_within_stft_tolerance(
    variant_id,
    delay,
    length,
):
    rng = np.random.default_rng(5_500 + length + delay)
    dsp = rng.normal(0.0, 0.1, length).astype(np.float32)
    dl = rng.normal(0.0, 0.1, length).astype(np.float32)
    hybrid_config = _hybrid_config(delay)

    band = FixedFrequencyBandBlender(
        hybrid_config,
        _method_config(dl_weight=0.5),
        variant_id=variant_id,
    ).process_array(dsp, dl, sample_rate=48_000)
    waveform = FixedWaveformBlender(
        hybrid_config,
        alpha=0.5,
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_allclose(band.audio, waveform.audio, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_silence_is_exact_and_has_zero_diagnostics(variant_id):
    silence = np.zeros(997, dtype=np.float32)

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id=variant_id,
    ).process_array(silence, silence, sample_rate=48_000)

    np.testing.assert_array_equal(result.audio, silence)
    assert result.statistics.peak_abs == 0.0
    assert result.statistics.clipped_sample_count == 0
    assert result.statistics.clipped_sample_fraction == 0.0


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_equal_inputs_reconstruct_for_complex_and_magnitude_phase_modes(variant_id):
    rng = np.random.default_rng(6_001)
    audio = rng.normal(0.0, 0.15, 1_577).astype(np.float32)

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id=variant_id,
    ).process_array(audio, audio, sample_rate=48_000)

    np.testing.assert_allclose(result.audio, audio, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("delay", [-9, 0, 9])
def test_impulses_remain_finite_exact_length_and_causally_aligned(delay):
    dsp = np.zeros(961, dtype=np.float32)
    dl = np.zeros(961, dtype=np.float32)
    dsp[100] = 0.8
    dl[100 + delay] = 0.8

    result = FixedFrequencyBandBlender(
        _hybrid_config(delay),
        METHOD_CONFIG,
        variant_id="complex_step",
    ).process_array(dsp, dl, sample_rate=48_000)

    assert result.audio.shape == dsp.shape
    assert np.all(np.isfinite(result.audio))
    assert int(np.argmax(np.abs(result.audio))) == 100 + max(delay, 0)
    assert result.statistics.peak_abs > 0.0


def test_bin_centered_tone_obeys_uniform_complex_weight():
    time = np.arange(2_400, dtype=np.float64)
    dsp = (0.8 * np.sin(2.0 * np.pi * 1_000.0 * time / 48_000.0)).astype(np.float32)
    dl = np.zeros_like(dsp)

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        _method_config(dl_weight=0.25),
        variant_id="complex_step",
    ).process_array(dsp, dl, sample_rate=48_000)

    np.testing.assert_allclose(result.audio, 0.75 * dsp, atol=2e-6, rtol=0.0)


def test_clipping_diagnostics_do_not_clip_or_normalize_output():
    audio = np.tile(
        np.array([-1.25, -1.2, -0.2, 0.2, 1.2, 1.25], dtype=np.float32),
        200,
    )
    config = _method_config(dl_weight=0.0)

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        config,
        variant_id="complex_step",
    ).process_array(audio, np.zeros_like(audio), sample_rate=48_000)

    np.testing.assert_allclose(result.audio, audio, atol=2e-6, rtol=0.0)
    assert result.statistics.peak_abs == pytest.approx(1.25, abs=2e-6)
    assert result.statistics.clipping_threshold == 1.0
    assert result.statistics.clipped_sample_count == 800
    assert result.statistics.clipped_sample_fraction == pytest.approx(2 / 3)
    assert np.max(result.audio) > 1.0
    assert np.min(result.audio) < -1.0


def test_custom_clipping_threshold_is_diagnostic_only():
    audio = np.tile(np.array([-0.75, -0.2, 0.2, 0.75], dtype=np.float32), 240)

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        _method_config(dl_weight=1.0, clipping_threshold=0.5),
        variant_id="magnitude_dl_phase",
    ).process_array(np.zeros_like(audio), audio, sample_rate=48_000)

    np.testing.assert_allclose(result.audio, audio, atol=2e-6, rtol=0.0)
    assert result.statistics.clipped_sample_count == 480
    assert result.statistics.clipped_sample_fraction == pytest.approx(0.5)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_repeat_reset_and_fresh_instance_are_bit_deterministic(variant_id):
    rng = np.random.default_rng(6_429)
    clip_a = rng.normal(0.0, 0.1, 1_501).astype(np.float32)
    dsp = rng.normal(0.0, 0.1, 997).astype(np.float32)
    dl = rng.normal(0.0, 0.1, 997).astype(np.float32)
    reused = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id=variant_id,
    )

    first = reused.process_array(dsp, dl, sample_rate=48_000)
    repeated = reused.process_array(dsp, dl, sample_rate=48_000)
    reused.process_array(clip_a, -clip_a, sample_rate=48_000)
    after_other_clip = reused.process_array(dsp, dl, sample_rate=48_000)
    reused.reset()
    after_manual_reset = reused.process_array(dsp, dl, sample_rate=48_000)
    fresh = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id=variant_id,
    ).process_array(dsp, dl, sample_rate=48_000)

    for candidate in (repeated, after_other_clip, after_manual_reset, fresh):
        np.testing.assert_array_equal(candidate.audio, first.audio)
        assert candidate.statistics == first.statistics


def test_processing_does_not_mutate_or_alias_caller_arrays():
    dsp = np.linspace(-0.2, 0.2, 777, dtype=np.float64)
    dl = -dsp
    dsp_before = dsp.copy()
    dl_before = dl.copy()

    result = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id="complex_smoothed",
    ).process_array(dsp, dl, sample_rate=48_000)
    result.audio[0] = 123.0

    np.testing.assert_array_equal(dsp, dsp_before)
    np.testing.assert_array_equal(dl, dl_before)


@pytest.mark.parametrize(
    "dsp, dl, exception, message",
    [
        ([0.0], np.zeros(1, dtype=np.float32), TypeError, "numpy.ndarray"),
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
        (
            np.array([np.finfo(np.float64).max], dtype=np.float64),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "finite.*range",
        ),
    ],
)
def test_invalid_arrays_are_rejected(dsp, dl, exception, message):
    blender = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id="complex_step",
    )

    with pytest.raises(exception, match=message):
        blender.process_array(dsp, dl, sample_rate=48_000)


@pytest.mark.parametrize("sample_rate", [16_000, 47_999, 48_001])
def test_wrong_sample_rate_is_rejected(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="must be 48000 Hz"):
        FixedFrequencyBandBlender(
            _hybrid_config(),
            METHOD_CONFIG,
            variant_id="complex_step",
        ).process_array(audio, audio, sample_rate=sample_rate)


@pytest.mark.parametrize("sample_rate", [48_000.0, True, np.bool_(False), "48000"])
def test_noninteger_sample_rate_is_rejected(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(TypeError, match="sample_rate must be an integer"):
        FixedFrequencyBandBlender(
            _hybrid_config(),
            METHOD_CONFIG,
            variant_id="complex_step",
        ).process_array(audio, audio, sample_rate=sample_rate)


@pytest.mark.parametrize(
    "hybrid_config, method_config, variant_id, exception, message",
    [
        (object(), METHOD_CONFIG, "complex_step", TypeError, "hybrid_config"),
        (_hybrid_config(), object(), "complex_step", TypeError, "method_config"),
        (_hybrid_config(), METHOD_CONFIG, None, TypeError, "variant_id"),
        (_hybrid_config(), METHOD_CONFIG, "unknown", ValueError, "unknown"),
        (
            _hybrid_config(),
            dataclasses.replace(METHOD_CONFIG, dl_weights=(0.5,)),
            "complex_step",
            ValueError,
            "one value per frequency band",
        ),
        (
            HybridConfig(sample_rate=16_000),
            METHOD_CONFIG,
            "complex_step",
            ValueError,
            "sample_rate must be 48000",
        ),
    ],
)
def test_constructor_rejects_invalid_types_and_configuration(
    hybrid_config,
    method_config,
    variant_id,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        FixedFrequencyBandBlender(
            hybrid_config,
            method_config,
            variant_id=variant_id,
        )


def test_result_statistics_and_variant_metadata_are_frozen_and_read_only():
    audio = np.linspace(-0.2, 0.2, 481, dtype=np.float32)
    blender = FixedFrequencyBandBlender(
        _hybrid_config(),
        METHOD_CONFIG,
        variant_id="complex_smoothed",
    )

    result = blender.process_array(audio, -audio, sample_rate=48_000)

    assert result.statistics.variant_id == "complex_smoothed"
    assert result.statistics.blend_domain == "complex"
    assert result.statistics.phase_source == "none"
    assert result.statistics.frequency_smoothing_bins == 9
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.delay_samples = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.statistics.sample_count = 0
    with pytest.raises(ValueError, match="read-only"):
        blender.definition.effective_dl_weight_by_bin[0] = 0.0


def test_variant_config_is_frozen():
    variant = METHOD_CONFIG.variant("complex_step")

    assert isinstance(variant, BandBlendVariantConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.blend_domain = "magnitude"
