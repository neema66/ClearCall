"""Strict configuration tests for Method 3 fixed-frequency-band blending."""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Any

import pytest
import yaml

from senhance.pipeline.hybrid.method3.method3_band_config import (
    CANONICAL_BIN_WIDTH_HZ,
    CANONICAL_FRAME_SIZE,
    CANONICAL_NUM_BINS,
    CANONICAL_NYQUIST_HZ,
    CANONICAL_SAMPLE_RATE,
    BandBlendVariantConfig,
    BandEvaluationConfig,
    BandListeningSetConfig,
    Method3BandConfig,
    load_method3_band_config,
)


def _raw_config() -> dict[str, Any]:
    return {
        "method_3_bands": {
            "mode": "fixed_frequency_bands",
            "band_edges_hz": [0.0, 300.0, 1000.0, 3000.0, 8000.0, 24000.0],
            "dl_weights": [0.25, 0.55, 0.85, 0.75, 0.35],
            "variants": [
                {
                    "id": "complex_step",
                    "blend_domain": "complex",
                    "phase_source": "none",
                    "frequency_smoothing_bins": 1,
                },
                {
                    "id": "complex_smoothed",
                    "blend_domain": "complex",
                    "phase_source": "none",
                    "frequency_smoothing_bins": 9,
                },
                {
                    "id": "magnitude_dsp_phase",
                    "blend_domain": "magnitude",
                    "phase_source": "dsp",
                    "frequency_smoothing_bins": 9,
                },
                {
                    "id": "magnitude_dl_phase",
                    "blend_domain": "magnitude",
                    "phase_source": "dl",
                    "frequency_smoothing_bins": 9,
                },
            ],
            "evaluation": {
                "waveform_alpha": 0.7,
                "clipping_threshold": 1.0,
                "listening_set": {
                    "file_names": [
                        "p232_005.wav",
                        "p232_010.wav",
                        "p232_019.wav",
                    ]
                },
            },
        }
    }


def _write_config(tmp_path, raw: Any):
    path = tmp_path / "method3-bands.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _load_raw(tmp_path, raw: Any) -> Method3BandConfig:
    return load_method3_band_config(_write_config(tmp_path, raw))


def _section(raw: Any, path: tuple[str | int, ...]) -> Any:
    section = raw
    for component in path:
        section = section[component]
    return section


def _variants() -> tuple[BandBlendVariantConfig, ...]:
    return (
        BandBlendVariantConfig("complex_step", "complex", "none", 1),
        BandBlendVariantConfig("complex_smoothed", "complex", "none", 9),
        BandBlendVariantConfig("magnitude_dsp", "magnitude", "dsp", 9),
        BandBlendVariantConfig("magnitude_dl", "magnitude", "dl", 9),
    )


def _evaluation(**overrides: Any) -> BandEvaluationConfig:
    values: dict[str, Any] = {
        "waveform_alpha": 0.7,
        "clipping_threshold": 1.0,
        "listening_set": BandListeningSetConfig(("p232_005.wav",)),
    }
    values.update(overrides)
    return BandEvaluationConfig(**values)


def _config(**overrides: Any) -> Method3BandConfig:
    values: dict[str, Any] = {
        "mode": "fixed_frequency_bands",
        "band_edges_hz": (0.0, 300.0, 1000.0, 3000.0, 8000.0, 24000.0),
        "dl_weights": (0.25, 0.55, 0.85, 0.75, 0.35),
        "variants": _variants(),
        "evaluation": _evaluation(),
    }
    values.update(overrides)
    return Method3BandConfig(**values)


def test_project_band_config_loads_exact_experimental_values():
    config = load_method3_band_config("config/hybrid_method_3_bands.yaml")

    assert config.mode == "fixed_frequency_bands"
    assert config.band_edges_hz == (
        0.0,
        300.0,
        1000.0,
        3000.0,
        8000.0,
        24000.0,
    )
    assert config.dl_weights == (0.25, 0.55, 0.85, 0.75, 0.35)
    assert config.variants == (
        BandBlendVariantConfig("complex_step", "complex", "none", 1),
        BandBlendVariantConfig("complex_smoothed", "complex", "none", 9),
        BandBlendVariantConfig("magnitude_dsp_phase", "magnitude", "dsp", 9),
        BandBlendVariantConfig("magnitude_dl_phase", "magnitude", "dl", 9),
    )
    assert config.evaluation == BandEvaluationConfig(
        waveform_alpha=0.7,
        clipping_threshold=1.0,
        listening_set=BandListeningSetConfig(("p232_005.wav", "p232_010.wav", "p232_019.wav")),
    )
    assert config.num_bands == 5
    assert config.variant("complex_smoothed") == config.variants[1]


def test_canonical_fft_constants_are_internally_consistent():
    assert CANONICAL_SAMPLE_RATE == 48_000
    assert CANONICAL_FRAME_SIZE == 960
    assert CANONICAL_NUM_BINS == 481
    assert CANONICAL_NYQUIST_HZ == 24_000.0
    assert CANONICAL_BIN_WIDTH_HZ == 50.0
    assert CANONICAL_NUM_BINS == CANONICAL_FRAME_SIZE // 2 + 1
    assert CANONICAL_BIN_WIDTH_HZ == CANONICAL_SAMPLE_RATE / CANONICAL_FRAME_SIZE


@pytest.mark.parametrize(
    "path,key,operation,message",
    [
        ((), "unexpected", "add", "Unknown root configuration key"),
        ((), "method_3_bands", "remove", "Missing root configuration key"),
        (
            ("method_3_bands",),
            "selected_dsp",
            "add",
            "Unknown method_3_bands configuration key",
        ),
        (
            ("method_3_bands",),
            "mode",
            "remove",
            "Missing method_3_bands configuration key",
        ),
        (
            ("method_3_bands", "variants", 0),
            "window",
            "add",
            "Unknown method_3_bands.variants[0] configuration key",
        ),
        (
            ("method_3_bands", "variants", 0),
            "phase_source",
            "remove",
            "Missing method_3_bands.variants[0] configuration key",
        ),
        (
            ("method_3_bands", "evaluation"),
            "metric",
            "add",
            "Unknown method_3_bands.evaluation configuration key",
        ),
        (
            ("method_3_bands", "evaluation"),
            "waveform_alpha",
            "remove",
            "Missing method_3_bands.evaluation configuration key",
        ),
        (
            ("method_3_bands", "evaluation", "listening_set"),
            "ratings",
            "add",
            "Unknown method_3_bands.evaluation.listening_set configuration key",
        ),
        (
            ("method_3_bands", "evaluation", "listening_set"),
            "file_names",
            "remove",
            "Missing method_3_bands.evaluation.listening_set configuration key",
        ),
    ],
)
def test_loader_rejects_unknown_and_missing_fields(
    tmp_path,
    path,
    key,
    operation,
    message,
):
    raw = _raw_config()
    section = _section(raw, path)
    if operation == "add":
        section[key] = "not-allowed"
    else:
        del section[key]

    with pytest.raises(ValueError, match=re.escape(message)):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "path,replacement,name",
    [
        ((), [], "configuration root"),
        (("method_3_bands",), [], "method_3_bands"),
        (("method_3_bands", "variants", 0), [], "method_3_bands.variants[0]"),
        (("method_3_bands", "evaluation"), [], "method_3_bands.evaluation"),
        (
            ("method_3_bands", "evaluation", "listening_set"),
            [],
            "method_3_bands.evaluation.listening_set",
        ),
    ],
)
def test_loader_requires_mappings_for_every_section(tmp_path, path, replacement, name):
    raw: Any = _raw_config()
    if not path:
        raw = replacement
    else:
        parent = _section(raw, path[:-1])
        parent[path[-1]] = replacement

    with pytest.raises(ValueError, match=rf"{re.escape(name)} must be a YAML mapping"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "path,replacement,name",
    [
        (("method_3_bands", "band_edges_hz"), {}, "band_edges_hz"),
        (("method_3_bands", "dl_weights"), "0.5", "dl_weights"),
        (("method_3_bands", "variants"), {}, "variants"),
        (
            ("method_3_bands", "evaluation", "listening_set", "file_names"),
            "p232_005.wav",
            "file_names",
        ),
    ],
)
def test_loader_requires_yaml_lists(tmp_path, path, replacement, name):
    raw = _raw_config()
    parent = _section(raw, path[:-1])
    parent[path[-1]] = replacement

    with pytest.raises(ValueError, match=rf"{re.escape(name)} must be a YAML list"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "path,replacement,name",
    [
        (("method_3_bands", "mode"), 3, "method_3_bands.mode"),
        (("method_3_bands", "variants", 0, "id"), 1, "variants[0].id"),
        (
            ("method_3_bands", "variants", 0, "blend_domain"),
            True,
            "variants[0].blend_domain",
        ),
        (
            ("method_3_bands", "variants", 0, "phase_source"),
            None,
            "variants[0].phase_source",
        ),
    ],
)
def test_loader_rejects_implicit_string_coercion(tmp_path, path, replacement, name):
    raw = _raw_config()
    parent = _section(raw, path[:-1])
    parent[path[-1]] = replacement

    with pytest.raises(TypeError, match=rf"{re.escape(name)} must be a string"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize("replacement", [True, 1.0, "9", None])
def test_loader_requires_a_real_integer_smoothing_width(tmp_path, replacement):
    raw = _raw_config()
    raw["method_3_bands"]["variants"][0]["frequency_smoothing_bins"] = replacement

    with pytest.raises(TypeError, match="frequency_smoothing_bins must be an integer"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "path,replacement,name",
    [
        (("method_3_bands", "band_edges_hz", 1), 300, "band_edges_hz[1]"),
        (("method_3_bands", "band_edges_hz", 1), True, "band_edges_hz[1]"),
        (("method_3_bands", "dl_weights", 0), 0, "dl_weights[0]"),
        (("method_3_bands", "dl_weights", 0), False, "dl_weights[0]"),
        (
            ("method_3_bands", "evaluation", "waveform_alpha"),
            1,
            "evaluation.waveform_alpha",
        ),
        (
            ("method_3_bands", "evaluation", "waveform_alpha"),
            True,
            "evaluation.waveform_alpha",
        ),
        (
            ("method_3_bands", "evaluation", "clipping_threshold"),
            1,
            "evaluation.clipping_threshold",
        ),
        (
            ("method_3_bands", "evaluation", "clipping_threshold"),
            False,
            "evaluation.clipping_threshold",
        ),
    ],
)
def test_loader_rejects_implicit_numeric_coercion(tmp_path, path, replacement, name):
    raw = _raw_config()
    parent = _section(raw, path[:-1])
    parent[path[-1]] = replacement

    with pytest.raises(TypeError, match=rf"{re.escape(name)} must be a floating-point value"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "path,replacement,name",
    [
        (("method_3_bands", "band_edges_hz", 1), float("nan"), "band_edges_hz[1]"),
        (("method_3_bands", "band_edges_hz", 1), float("inf"), "band_edges_hz[1]"),
        (("method_3_bands", "dl_weights", 0), float("-inf"), "dl_weights[0]"),
        (
            ("method_3_bands", "evaluation", "waveform_alpha"),
            float("nan"),
            "evaluation.waveform_alpha",
        ),
        (
            ("method_3_bands", "evaluation", "clipping_threshold"),
            float("inf"),
            "evaluation.clipping_threshold",
        ),
    ],
)
def test_loader_rejects_nonfinite_numbers(tmp_path, path, replacement, name):
    raw = _raw_config()
    parent = _section(raw, path[:-1])
    parent[path[-1]] = replacement

    with pytest.raises(ValueError, match=rf"{re.escape(name)} must be finite"):
        _load_raw(tmp_path, raw)


@pytest.mark.parametrize(
    "config,message",
    [
        (_config(mode="fixed_waveform"), "fixed_frequency_bands"),
        (_config(band_edges_hz=(0.0,)), "at least 0 Hz and Nyquist"),
        (
            _config(band_edges_hz=(50.0, 300.0, 1000.0, 3000.0, 8000.0, 24000.0)),
            "begin at 0.0",
        ),
        (
            _config(band_edges_hz=(0.0, 300.0, 1000.0, 3000.0, 8000.0, 23950.0)),
            "end at 24000.0",
        ),
        (
            _config(band_edges_hz=(0.0, 300.0, 1000.0, 1000.0, 8000.0, 24000.0)),
            "strictly increasing and unique",
        ),
        (
            _config(band_edges_hz=(0.0, 300.0, 1000.0, 900.0, 8000.0, 24000.0)),
            "strictly increasing and unique",
        ),
        (
            _config(band_edges_hz=(0.0, 300.1, 1000.0, 3000.0, 8000.0, 24000.0)),
            "50 Hz FFT-bin grid",
        ),
        (
            _config(
                band_edges_hz=(
                    0.0,
                    300.0,
                    300.00000000001,
                    3000.0,
                    8000.0,
                    24000.0,
                )
            ),
            "every configured frequency band must contain at least one FFT bin",
        ),
        (
            _config(band_edges_hz=(0.0, 300.0, float("nan"), 3000.0, 8000.0, 24000.0)),
            "must be finite",
        ),
    ],
)
def test_band_edges_must_define_an_ordered_canonical_fft_grid(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    "weights,message",
    [
        ((0.25, 0.55, 0.85, 0.75), "exactly one value per frequency band"),
        ((0.25, 0.55, 0.85, 0.75, 0.35, 0.1), "exactly one value per frequency band"),
        ((-0.01, 0.55, 0.85, 0.75, 0.35), "0.0 <= weight <= 1.0"),
        ((0.25, 0.55, 1.01, 0.75, 0.35), "0.0 <= weight <= 1.0"),
        ((0.25, 0.55, float("nan"), 0.75, 0.35), "must be finite"),
    ],
)
def test_band_weights_require_one_finite_unit_interval_value_per_band(weights, message):
    with pytest.raises(ValueError, match=message):
        _config(dl_weights=weights).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("band_edges_hz", [0.0, 24000.0]),
        ("dl_weights", [0.5]),
        ("band_edges_hz", (0.0, 300, 24000.0)),
        ("dl_weights", (True, 0.5, 0.5, 0.5, 0.5)),
    ],
)
def test_direct_config_validation_rejects_wrong_numeric_container_or_element_types(
    field,
    value,
):
    with pytest.raises((TypeError, ValueError)):
        _config(**{field: value}).validate()


@pytest.mark.parametrize(
    "variant,message",
    [
        (BandBlendVariantConfig("", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("Complex", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("complex-step", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("_complex", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("complex_", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("complex__step", "complex", "none", 1), "variant id"),
        (BandBlendVariantConfig("valid", "power", "none", 1), "blend_domain"),
        (
            BandBlendVariantConfig("valid", "complex", "dsp", 1),
            "complex band blending requires phase_source='none'",
        ),
        (
            BandBlendVariantConfig("valid", "complex", "dl", 1),
            "complex band blending requires phase_source='none'",
        ),
        (
            BandBlendVariantConfig("valid", "magnitude", "none", 1),
            "magnitude band blending requires phase_source='dsp' or 'dl'",
        ),
        (
            BandBlendVariantConfig("valid", "magnitude", "clean", 1),
            "magnitude band blending requires phase_source='dsp' or 'dl'",
        ),
    ],
)
def test_variant_identity_domain_and_phase_rules(variant, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        variant.validate()


@pytest.mark.parametrize("variant_id", ["a", "complex1", "complex_step_9", "9"])
def test_variant_accepts_safe_stable_ids(variant_id):
    BandBlendVariantConfig(variant_id, "complex", "none", 1).validate()


@pytest.mark.parametrize("width", [0, -1, 2, 480, 482, 483])
def test_smoothing_width_must_be_positive_odd_and_within_the_rfft(width):
    variant = BandBlendVariantConfig("valid", "complex", "none", width)

    with pytest.raises(ValueError, match="positive odd integer no larger than 481"):
        variant.validate()


@pytest.mark.parametrize("width", [True, 1.0, "1", None])
def test_direct_variant_validation_rejects_noninteger_smoothing_width(width):
    variant = BandBlendVariantConfig("valid", "complex", "none", width)

    with pytest.raises(TypeError, match="must be an integer"):
        variant.validate()


@pytest.mark.parametrize("width", [1, 3, 481])
def test_smoothing_accepts_valid_odd_widths(width):
    BandBlendVariantConfig("valid", "complex", "none", width).validate()


@pytest.mark.parametrize(
    "variants,exception,message",
    [
        ((), ValueError, "non-empty tuple"),
        (list(_variants()), ValueError, "non-empty tuple"),
        ((_variants()[0], object()), TypeError, "must be BandBlendVariantConfig"),
        (
            (
                _variants()[0],
                dataclasses.replace(_variants()[1], id="complex_step"),
                _variants()[2],
                _variants()[3],
            ),
            ValueError,
            "variant ids must be unique",
        ),
        (
            (
                BandBlendVariantConfig("magnitude_dsp_step", "magnitude", "dsp", 1),
                BandBlendVariantConfig("magnitude_dsp_smooth", "magnitude", "dsp", 9),
                BandBlendVariantConfig("magnitude_dl_smooth", "magnitude", "dl", 9),
            ),
            ValueError,
            "direct complex blend variant is required",
        ),
        (
            (
                _variants()[0],
                _variants()[1],
                BandBlendVariantConfig("magnitude_dl_step", "magnitude", "dl", 1),
            ),
            ValueError,
            "explicitly compare DSP and DL phase",
        ),
        (
            (
                _variants()[0],
                _variants()[1],
                BandBlendVariantConfig("magnitude_dsp_step", "magnitude", "dsp", 1),
            ),
            ValueError,
            "explicitly compare DSP and DL phase",
        ),
        (
            tuple(
                dataclasses.replace(variant, frequency_smoothing_bins=9) for variant in _variants()
            ),
            ValueError,
            "unsmoothed step-weight variant is required",
        ),
        (
            tuple(
                dataclasses.replace(variant, frequency_smoothing_bins=1) for variant in _variants()
            ),
            ValueError,
            "frequency-smoothed variant is required",
        ),
    ],
)
def test_variant_collection_requires_all_planned_ablations(variants, exception, message):
    with pytest.raises(exception, match=message):
        _config(variants=variants).validate()


@pytest.mark.parametrize("variant_id", [None, 3, True])
def test_variant_lookup_requires_string_id(variant_id):
    with pytest.raises(TypeError, match="variant_id must be a string"):
        _config().variant(variant_id)


def test_variant_lookup_rejects_an_unknown_id():
    with pytest.raises(ValueError, match="unknown Method 3 band variant"):
        _config().variant("not_configured")


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.1])
def test_evaluation_waveform_alpha_must_be_strictly_between_endpoints(alpha):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _evaluation(waveform_alpha=alpha).validate()


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf")])
def test_evaluation_waveform_alpha_must_be_finite(alpha):
    with pytest.raises(ValueError, match="must be finite"):
        _evaluation(waveform_alpha=alpha).validate()


@pytest.mark.parametrize("alpha", [True, 1, "0.7", None])
def test_evaluation_waveform_alpha_must_be_a_strict_float(alpha):
    with pytest.raises(TypeError, match="floating-point value"):
        _evaluation(waveform_alpha=alpha).validate()


@pytest.mark.parametrize("threshold", [0.0, -0.1])
def test_evaluation_clipping_threshold_must_be_positive(threshold):
    with pytest.raises(ValueError, match="must be positive"):
        _evaluation(clipping_threshold=threshold).validate()


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_evaluation_clipping_threshold_must_be_finite(threshold):
    with pytest.raises(ValueError, match="must be finite"):
        _evaluation(clipping_threshold=threshold).validate()


@pytest.mark.parametrize("threshold", [True, 1, "1.0", None])
def test_evaluation_clipping_threshold_must_be_a_strict_float(threshold):
    with pytest.raises(TypeError, match="floating-point value"):
        _evaluation(clipping_threshold=threshold).validate()


@pytest.mark.parametrize(
    "listening,exception,message",
    [
        (None, TypeError, "must be BandListeningSetConfig"),
        (BandListeningSetConfig(()), ValueError, "non-empty tuple"),
        (BandListeningSetConfig(["p232_005.wav"]), ValueError, "non-empty tuple"),
        (
            BandListeningSetConfig(("p232_005.wav", "p232_005.wav")),
            ValueError,
            "must be unique",
        ),
        (BandListeningSetConfig(("",)), TypeError, "non-empty strings"),
        (BandListeningSetConfig((3,)), TypeError, "non-empty strings"),
        (BandListeningSetConfig(("../p232_005.wav",)), ValueError, "safe WAV basenames"),
        (BandListeningSetConfig((r"..\p232_005.wav",)), ValueError, "safe WAV basenames"),
        (BandListeningSetConfig(("clips/p232_005.wav",)), ValueError, "safe WAV basenames"),
        (BandListeningSetConfig(("p232_005.flac",)), ValueError, "safe WAV basenames"),
        (BandListeningSetConfig(("p232_005",)), ValueError, "safe WAV basenames"),
    ],
)
def test_listening_set_requires_unique_safe_wav_basenames(listening, exception, message):
    with pytest.raises(exception, match=message):
        _evaluation(listening_set=listening).validate()


def test_listening_set_accepts_case_insensitive_wav_suffix():
    _evaluation(listening_set=BandListeningSetConfig(("P232_005.WAV",))).validate()


def test_method_config_requires_evaluation_config_instance():
    with pytest.raises(TypeError, match="evaluation must be BandEvaluationConfig"):
        _config(evaluation=None).validate()


@pytest.mark.parametrize(
    "instance,field,replacement",
    [
        (_config(), "mode", "other"),
        (_variants()[0], "id", "other"),
        (_evaluation(), "waveform_alpha", 0.5),
        (BandListeningSetConfig(("p232_005.wav",)), "file_names", ("other.wav",)),
    ],
)
def test_all_configuration_dataclasses_are_frozen(instance, field, replacement):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, replacement)


def test_loading_does_not_mutate_caller_owned_raw_values(tmp_path):
    raw = _raw_config()
    original = copy.deepcopy(raw)

    _load_raw(tmp_path, raw)

    assert raw == original


def test_missing_method3_band_config_file_raises():
    with pytest.raises(FileNotFoundError, match="Method 3 band config file not found"):
        load_method3_band_config("config/does-not-exist-method3-bands.yaml")
