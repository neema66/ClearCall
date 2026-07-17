"""Method 3 Version 2 fixed-band mapping and spectral-policy tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method3.bands import (
    FixedBandDefinition,
    FixedBandSpectrumProcessor,
    _smooth_weights,
)
from senhance.pipeline.hybrid.method3.config import HybridConfig
from senhance.pipeline.hybrid.method3.method3_band_config import (
    CANONICAL_BIN_WIDTH_HZ,
    CANONICAL_NUM_BINS,
    CANONICAL_NYQUIST_HZ,
    BandBlendVariantConfig,
    Method3BandConfig,
    load_method3_band_config,
)
from senhance.pipeline.hybrid.method3.paired_stft import PairedSpectrumFrame


BASE_CONFIG = load_method3_band_config("config/hybrid_method_3_bands.yaml")
HYBRID_CONFIG = HybridConfig()


def _variant(config: Method3BandConfig, variant_id: str) -> BandBlendVariantConfig:
    return config.variant(variant_id)


def _definition(
    config: Method3BandConfig = BASE_CONFIG,
    variant_id: str = "complex_step",
) -> FixedBandDefinition:
    return FixedBandDefinition.from_config(
        config,
        _variant(config, variant_id),
        HYBRID_CONFIG,
    )


def _config_with_weights(
    weights: tuple[float, ...],
    *,
    edges: tuple[float, ...] | None = None,
    extra_variants: tuple[BandBlendVariantConfig, ...] = (),
) -> Method3BandConfig:
    if edges is None:
        edges = tuple(float(index * 50) for index in range(len(weights))) + (CANONICAL_NYQUIST_HZ,)
    config = dataclasses.replace(
        BASE_CONFIG,
        band_edges_hz=edges,
        dl_weights=weights,
        variants=BASE_CONFIG.variants + extra_variants,
    )
    config.validate()
    return config


def _frame(dsp: np.ndarray, dl: np.ndarray, *, index: int = 0) -> PairedSpectrumFrame:
    return PairedSpectrumFrame(
        index=index,
        reference_spectrum=dsp,
        candidate_spectrum=dl,
    )


def _random_spectra(seed: int = 429) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dsp = rng.normal(size=CANONICAL_NUM_BINS) + 1j * rng.normal(size=CANONICAL_NUM_BINS)
    dl = rng.normal(size=CANONICAL_NUM_BINS) + 1j * rng.normal(size=CANONICAL_NUM_BINS)
    return dsp.astype(np.complex128), dl.astype(np.complex128)


def test_canonical_definition_owns_all_481_bins_exactly_once():
    definition = _definition()

    assert CANONICAL_NUM_BINS == 481
    assert CANONICAL_BIN_WIDTH_HZ == 50.0
    assert CANONICAL_NYQUIST_HZ == 24_000.0
    assert definition.frequencies_hz.shape == (481,)
    assert definition.band_index_by_bin.shape == (481,)
    assert definition.step_dl_weight_by_bin.shape == (481,)
    assert definition.effective_dl_weight_by_bin.shape == (481,)
    assert definition.band_bin_counts == (6, 14, 40, 100, 321)
    assert sum(definition.band_bin_counts) == 481
    np.testing.assert_array_equal(
        np.bincount(definition.band_index_by_bin, minlength=5),
        np.asarray(definition.band_bin_counts),
    )
    np.testing.assert_array_equal(
        np.unique(definition.band_index_by_bin),
        np.arange(5),
    )


@pytest.mark.parametrize(
    "bin_index, expected_frequency, expected_band, expected_weight",
    [
        (0, 0.0, 0, 0.25),
        (5, 250.0, 0, 0.25),
        (6, 300.0, 1, 0.55),
        (19, 950.0, 1, 0.55),
        (20, 1_000.0, 2, 0.85),
        (59, 2_950.0, 2, 0.85),
        (60, 3_000.0, 3, 0.75),
        (159, 7_950.0, 3, 0.75),
        (160, 8_000.0, 4, 0.35),
        (480, 24_000.0, 4, 0.35),
    ],
)
def test_half_open_band_boundaries_and_nyquist_ownership(
    bin_index,
    expected_frequency,
    expected_band,
    expected_weight,
):
    definition = _definition()

    assert definition.frequencies_hz[bin_index] == expected_frequency
    assert definition.band_index_by_bin[bin_index] == expected_band
    assert definition.step_dl_weight_by_bin[bin_index] == expected_weight
    assert definition.effective_dl_weight_by_bin[bin_index] == expected_weight


def test_definition_preserves_policy_metadata_and_exposes_read_only_arrays():
    definition = _definition()

    assert definition.band_edges_hz == BASE_CONFIG.band_edges_hz
    assert definition.band_dl_weights == BASE_CONFIG.dl_weights
    assert definition.variant == _variant(BASE_CONFIG, "complex_step")
    for array in (
        definition.frequencies_hz,
        definition.band_index_by_bin,
        definition.step_dl_weight_by_bin,
        definition.effective_dl_weight_by_bin,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0

    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.band_bin_counts = (481,)


def test_smoothed_definition_changes_only_effective_weights():
    step = _definition(BASE_CONFIG, "complex_step")
    smoothed = _definition(BASE_CONFIG, "complex_smoothed")

    np.testing.assert_array_equal(smoothed.band_index_by_bin, step.band_index_by_bin)
    np.testing.assert_array_equal(smoothed.step_dl_weight_by_bin, step.step_dl_weight_by_bin)
    np.testing.assert_array_equal(
        smoothed.effective_dl_weight_by_bin,
        _smooth_weights(step.step_dl_weight_by_bin, 9),
    )
    assert not np.array_equal(
        smoothed.effective_dl_weight_by_bin,
        smoothed.step_dl_weight_by_bin,
    )
    assert np.all(smoothed.effective_dl_weight_by_bin >= 0.0)
    assert np.all(smoothed.effective_dl_weight_by_bin <= 1.0)


def test_smoothing_width_one_returns_an_independent_exact_copy():
    source = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)

    output = _smooth_weights(source, 1)

    np.testing.assert_array_equal(output, source)
    assert output is not source
    assert not np.shares_memory(output, source)
    output[0] = 0.9
    assert source[0] == 0.0


@pytest.mark.parametrize("constant", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_smoothing_preserves_constants_exactly(constant, width):
    source = np.full(11, constant, dtype=np.float64)

    output = _smooth_weights(source, width)

    np.testing.assert_array_equal(output, source)
    assert output is not source


def test_smoothing_is_edge_padded_and_never_wraps_across_dc_and_nyquist():
    dc_step = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    nyquist_step = dc_step[::-1].copy()

    dc_smoothed = _smooth_weights(dc_step, 3)
    nyquist_smoothed = _smooth_weights(nyquist_step, 3)

    np.testing.assert_allclose(dc_smoothed, [2 / 3, 1 / 3, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(nyquist_smoothed, [0.0, 0.0, 0.0, 1 / 3, 2 / 3])
    assert dc_smoothed[-1] == 0.0
    assert nyquist_smoothed[0] == 0.0


def test_smoothing_transition_is_local_monotone_and_bounded():
    source = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)

    output = _smooth_weights(source, 5)

    np.testing.assert_allclose(
        output,
        [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0],
    )
    assert np.all(np.diff(output) >= 0.0)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0


@pytest.mark.parametrize(
    "samples, width, exception, message",
    [
        (np.zeros(0), 1, ValueError, "non-empty one-dimensional"),
        (np.zeros((1, 3)), 1, ValueError, "non-empty one-dimensional"),
        (np.array([0.0, np.nan]), 1, ValueError, "finite and bounded"),
        (np.array([0.0, np.inf]), 1, ValueError, "finite and bounded"),
        (np.array([-0.01, 0.5]), 1, ValueError, "finite and bounded"),
        (np.array([0.5, 1.01]), 1, ValueError, "finite and bounded"),
        (np.zeros(3), True, TypeError, "integer"),
        (np.zeros(3), 1.0, TypeError, "integer"),
        (np.zeros(3), 0, ValueError, "positive odd"),
        (np.zeros(3), -1, ValueError, "positive odd"),
        (np.zeros(3), 2, ValueError, "positive odd"),
        (np.zeros(3), 5, ValueError, "positive odd"),
    ],
)
def test_smoothing_rejects_invalid_samples_and_widths(samples, width, exception, message):
    with pytest.raises(exception, match=message):
        _smooth_weights(samples, width)


def test_complex_blend_matches_per_bin_formula_and_returns_complex128():
    definition = _definition(BASE_CONFIG, "complex_smoothed")
    processor = FixedBandSpectrumProcessor(definition)
    dsp, dl = _random_spectra()
    weights = definition.effective_dl_weight_by_bin
    expected = weights * dl + (1.0 - weights) * dsp

    output = processor(_frame(dsp, dl))

    np.testing.assert_array_equal(output, expected)
    assert output.shape == (481,)
    assert output.dtype == np.complex128
    assert output.flags.c_contiguous
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("endpoint, selected", [(0.0, "dsp"), (1.0, "dl")])
def test_global_complex_endpoints_are_bit_exact_source_copies(endpoint, selected):
    config = _config_with_weights((endpoint, endpoint, endpoint))
    processor = FixedBandSpectrumProcessor(_definition(config))
    dsp, dl = _random_spectra(430)
    expected = dsp if selected == "dsp" else dl

    output = processor(_frame(dsp, dl))

    np.testing.assert_array_equal(output, expected)
    assert output.tobytes() == expected.tobytes()
    assert output is not expected
    assert not np.shares_memory(output, expected)


def test_mixed_endpoint_bins_are_bit_exact_and_intermediate_bin_uses_formula():
    config = _config_with_weights((0.0, 0.5, 1.0))
    definition = _definition(config)
    processor = FixedBandSpectrumProcessor(definition)
    dsp, dl = _random_spectra(431)

    output = processor(_frame(dsp, dl))

    dsp_bins = definition.effective_dl_weight_by_bin == 0.0
    dl_bins = definition.effective_dl_weight_by_bin == 1.0
    intermediate = ~(dsp_bins | dl_bins)
    assert output[dsp_bins].tobytes() == dsp[dsp_bins].tobytes()
    assert output[dl_bins].tobytes() == dl[dl_bins].tobytes()
    np.testing.assert_array_equal(
        output[intermediate], 0.5 * dl[intermediate] + 0.5 * dsp[intermediate]
    )


@pytest.mark.parametrize(
    "phase_source, variant_id",
    [("dsp", "magnitude_dsp_step"), ("dl", "magnitude_dl_step")],
)
def test_magnitude_blend_uses_weighted_magnitude_and_selected_unit_phase(
    phase_source,
    variant_id,
):
    variant = BandBlendVariantConfig(
        id=variant_id,
        blend_domain="magnitude",
        phase_source=phase_source,
        frequency_smoothing_bins=1,
    )
    config = _config_with_weights((0.0, 0.25, 0.75, 1.0), extra_variants=(variant,))
    definition = _definition(config, variant_id)
    processor = FixedBandSpectrumProcessor(definition)
    dsp, dl = _random_spectra(432)
    weights = definition.effective_dl_weight_by_bin
    selected = dsp if phase_source == "dsp" else dl
    selected_magnitude = np.abs(selected)
    unit_phase = np.ones(481, dtype=np.complex128)
    nonzero = selected_magnitude > 0.0
    unit_phase[nonzero] = selected[nonzero] / selected_magnitude[nonzero]
    expected = (weights * np.abs(dl) + (1.0 - weights) * np.abs(dsp)) * unit_phase
    expected[weights == 0.0] = dsp[weights == 0.0]
    expected[weights == 1.0] = dl[weights == 1.0]

    output = processor(_frame(dsp, dl))

    np.testing.assert_allclose(output, expected, rtol=1e-15, atol=1e-15)
    assert output[weights == 0.0].tobytes() == dsp[weights == 0.0].tobytes()
    assert output[weights == 1.0].tobytes() == dl[weights == 1.0].tobytes()


@pytest.mark.parametrize(
    "phase_source, variant_id, dsp_value, dl_value, expected",
    [
        ("dsp", "zero_dsp_phase", 0.0 + 0.0j, 3.0 + 4.0j, 1.25 + 0.0j),
        ("dl", "zero_dl_phase", 3.0 + 4.0j, 0.0 + 0.0j, 3.75 + 0.0j),
    ],
)
def test_magnitude_blend_uses_positive_real_phase_when_selected_phase_is_zero(
    phase_source,
    variant_id,
    dsp_value,
    dl_value,
    expected,
):
    variant = BandBlendVariantConfig(
        id=variant_id,
        blend_domain="magnitude",
        phase_source=phase_source,
        frequency_smoothing_bins=1,
    )
    config = _config_with_weights((0.0, 0.25, 1.0), extra_variants=(variant,))
    processor = FixedBandSpectrumProcessor(_definition(config, variant_id))
    dsp = np.ones(481, dtype=np.complex128)
    dl = np.ones(481, dtype=np.complex128)
    dsp[1] = dsp_value
    dl[1] = dl_value

    output = processor(_frame(dsp, dl))

    assert output[1] == expected
    assert np.angle(output[1]) == 0.0


def test_processor_accepts_complex64_without_mutating_or_aliasing_inputs():
    processor = FixedBandSpectrumProcessor(_definition())
    dsp128, dl128 = _random_spectra(433)
    dsp = dsp128.astype(np.complex64)
    dl = dl128.astype(np.complex64)
    dsp_before = dsp.copy()
    dl_before = dl.copy()

    output = processor(_frame(dsp, dl))
    output[0] = 123.0 + 456.0j

    np.testing.assert_array_equal(dsp, dsp_before)
    np.testing.assert_array_equal(dl, dl_before)
    assert not np.shares_memory(output, dsp)
    assert not np.shares_memory(output, dl)


def test_processor_is_deterministic_stateless_and_frame_index_independent():
    processor = FixedBandSpectrumProcessor(_definition(BASE_CONFIG, "magnitude_dl_phase"))
    dsp, dl = _random_spectra(434)

    first = processor(_frame(dsp, dl, index=0))
    assert processor.reset() is None
    second = processor(_frame(dsp, dl, index=999))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "method_config, variant, hybrid_config, exception, message",
    [
        (
            object(),
            _variant(BASE_CONFIG, "complex_step"),
            HYBRID_CONFIG,
            TypeError,
            "method_config",
        ),
        (BASE_CONFIG, object(), HYBRID_CONFIG, TypeError, "variant"),
        (BASE_CONFIG, _variant(BASE_CONFIG, "complex_step"), object(), TypeError, "hybrid_config"),
        (
            dataclasses.replace(BASE_CONFIG, dl_weights=(0.1,)),
            _variant(BASE_CONFIG, "complex_step"),
            HYBRID_CONFIG,
            ValueError,
            "one value per frequency band",
        ),
        (
            BASE_CONFIG,
            dataclasses.replace(_variant(BASE_CONFIG, "complex_step"), id="foreign"),
            HYBRID_CONFIG,
            ValueError,
            "belong",
        ),
        (
            BASE_CONFIG,
            _variant(BASE_CONFIG, "complex_step"),
            dataclasses.replace(HYBRID_CONFIG, sample_rate=16_000),
            ValueError,
            "48000",
        ),
    ],
)
def test_definition_factory_rejects_invalid_contract_inputs(
    method_config,
    variant,
    hybrid_config,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        FixedBandDefinition.from_config(method_config, variant, hybrid_config)


def test_processor_constructor_requires_a_fixed_definition():
    with pytest.raises(TypeError, match="FixedBandDefinition"):
        FixedBandSpectrumProcessor(object())


@pytest.mark.parametrize(
    "dsp, dl, exception, message",
    [
        ([0j] * 481, np.zeros(481, dtype=np.complex128), TypeError, "numpy.ndarray"),
        (
            np.zeros(481, dtype=np.complex128),
            [0j] * 481,
            TypeError,
            "numpy.ndarray",
        ),
        (
            np.zeros((1, 481), dtype=np.complex128),
            np.zeros(481, dtype=np.complex128),
            ValueError,
            "one-dimensional",
        ),
        (
            np.zeros(481, dtype=np.complex128),
            np.zeros((481, 1), dtype=np.complex128),
            ValueError,
            "one-dimensional",
        ),
        (
            np.zeros(481, dtype=np.float64),
            np.zeros(481, dtype=np.complex128),
            TypeError,
            "complex floating",
        ),
        (
            np.zeros(481, dtype=np.complex128),
            np.zeros(481, dtype=np.float64),
            TypeError,
            "complex floating",
        ),
        (
            np.full(481, np.nan + 0j, dtype=np.complex128),
            np.zeros(481, dtype=np.complex128),
            ValueError,
            "finite",
        ),
        (
            np.zeros(481, dtype=np.complex128),
            np.full(481, np.inf + 0j, dtype=np.complex128),
            ValueError,
            "finite",
        ),
        (
            np.zeros(480, dtype=np.complex128),
            np.zeros(480, dtype=np.complex128),
            ValueError,
            "size",
        ),
        (
            np.zeros(482, dtype=np.complex128),
            np.zeros(482, dtype=np.complex128),
            ValueError,
            "size",
        ),
    ],
)
def test_processor_rejects_invalid_spectra(dsp, dl, exception, message):
    processor = FixedBandSpectrumProcessor(_definition())

    with pytest.raises(exception, match=message):
        processor(_frame(dsp, dl))


def test_processor_rejects_non_frame_input():
    with pytest.raises(TypeError, match="PairedSpectrumFrame"):
        FixedBandSpectrumProcessor(_definition())(object())
