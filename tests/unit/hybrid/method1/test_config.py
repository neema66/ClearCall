"""Strict schema, immutable dataclass, and ablation-order tests."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import yaml

from senhance.pipeline.hybrid.method1.config import (
    Method1EvaluationConfig,
    Method1KeepMapConfig,
    Method1ListeningSetConfig,
    Method1SafetyConfig,
    Method1STFTConfig,
    Method1VariantConfig,
    load_method1_config,
)


def _raw_config(method1_config_path):
    return yaml.safe_load(method1_config_path.read_text(encoding="utf-8"))


@pytest.fixture
def method1_config_path():
    from tests.unit.hybrid.method1.conftest import METHOD1_CONFIG_PATH

    return METHOD1_CONFIG_PATH


def _write_yaml(tmp_path, raw):
    path = tmp_path / "method1.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_project_config_loads_exact_geometry_policy_and_variants(method1_config):
    assert method1_config.mode == "dl_keep_map_safety_layer"
    assert method1_config.sample_rate == 48_000
    assert method1_config.alignment.dl_minus_noisy_delay_samples == 0
    assert method1_config.stft == Method1STFTConfig(960, 480, "hann")
    assert method1_config.stft.num_frequency_bins == 481
    assert method1_config.keep_map == Method1KeepMapConfig(1.0e-8, 1.0e-7)
    assert method1_config.safety == Method1SafetyConfig(0.85, 5, 0.10, 0.15, 0.10)
    assert tuple(variant.id for variant in method1_config.variants) == (
        "raw_dl_phase",
        "raw_noisy_phase",
        "temporal_dl_phase",
        "temporal_frequency_dl_phase",
        "temporal_frequency_floor_dl_phase",
        "full_dl_phase",
        "full_noisy_phase",
    )
    assert method1_config.evaluation.clipping_threshold == 1.0
    assert method1_config.evaluation.listening_set.variant_ids == (
        "raw_dl_phase",
        "full_dl_phase",
        "full_noisy_phase",
    )


def test_variant_lookup_is_explicit_and_strict(method1_config):
    assert method1_config.variant("full_dl_phase").rate_limits is True
    with pytest.raises(TypeError, match="variant_id must be a string"):
        method1_config.variant(1)
    with pytest.raises(ValueError, match="unknown Method 1 variant"):
        method1_config.variant("missing")


def test_all_configuration_dataclasses_are_frozen(method1_config):
    values = (
        method1_config,
        method1_config.alignment,
        method1_config.stft,
        method1_config.keep_map,
        method1_config.safety,
        method1_config.variants[0],
        method1_config.evaluation,
        method1_config.evaluation.listening_set,
    )
    for value in values:
        field = dataclasses.fields(value)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, getattr(value, field))


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda raw: raw.update({"extra": True}), "Unknown root"),
        (lambda raw: raw.pop("method_1"), "Missing root"),
        (lambda raw: raw["method_1"].update({"dsp": "improved_dsp"}), "Unknown method_1"),
        (lambda raw: raw["method_1"].pop("keep_map"), "Missing method_1"),
        (
            lambda raw: raw["method_1"]["alignment"].update({"extra": 1}),
            "Unknown method_1.alignment",
        ),
        (
            lambda raw: raw["method_1"]["stft"].pop("window"),
            "Missing method_1.stft",
        ),
        (
            lambda raw: raw["method_1"]["keep_map"].update({"extra": 0.0}),
            "Unknown method_1.keep_map",
        ),
        (
            lambda raw: raw["method_1"]["safety"].pop("gain_floor"),
            "Missing method_1.safety",
        ),
        (
            lambda raw: raw["method_1"]["variants"][0].update({"extra": False}),
            r"Unknown method_1.variants\[0\]",
        ),
        (
            lambda raw: raw["method_1"]["evaluation"].update({"extra": 1}),
            "Unknown method_1.evaluation",
        ),
        (
            lambda raw: raw["method_1"]["evaluation"]["listening_set"].pop("variant_ids"),
            "Missing method_1.evaluation.listening_set",
        ),
    ],
)
def test_loader_rejects_missing_and_unknown_fields(
    tmp_path,
    method1_config_path,
    mutation,
    message,
):
    raw = copy.deepcopy(_raw_config(method1_config_path))
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        load_method1_config(_write_yaml(tmp_path, raw))


@pytest.mark.parametrize(
    "mutation,exception,message",
    [
        (lambda m: m.__setitem__("sample_rate", True), TypeError, "sample_rate.*integer"),
        (
            lambda m: m["alignment"].__setitem__("dl_minus_noisy_delay_samples", False),
            TypeError,
            "delay_samples.*integer",
        ),
        (lambda m: m["stft"].__setitem__("frame_size", 960.0), TypeError, "frame_size.*integer"),
        (lambda m: m["keep_map"].__setitem__("epsilon", 1), TypeError, "epsilon.*floating"),
        (
            lambda m: m["safety"].__setitem__("temporal_smoothing", 1),
            TypeError,
            "temporal_smoothing.*floating",
        ),
        (
            lambda m: m["safety"].__setitem__("frequency_kernel_bins", True),
            TypeError,
            "frequency_kernel_bins.*integer",
        ),
        (
            lambda m: m["variants"][0].__setitem__("rate_limits", 0),
            TypeError,
            "rate_limits.*boolean",
        ),
        (
            lambda m: m["evaluation"].__setitem__("clipping_threshold", 1),
            TypeError,
            "clipping_threshold.*floating",
        ),
    ],
)
def test_loader_rejects_implicit_numeric_and_boolean_coercion(
    tmp_path,
    method1_config_path,
    mutation,
    exception,
    message,
):
    raw = copy.deepcopy(_raw_config(method1_config_path))
    mutation(raw["method_1"])
    with pytest.raises(exception, match=message):
        load_method1_config(_write_yaml(tmp_path, raw))


@pytest.mark.parametrize(
    "replacement,exception,message",
    [
        ({"mode": "other"}, ValueError, "mode"),
        ({"sample_rate": 16_000}, ValueError, "48000"),
        ({"sample_rate": True}, TypeError, "integer"),
        ({"alignment": None}, TypeError, "alignment"),
        ({"stft": Method1STFTConfig(1024, 512, "hann")}, ValueError, "960"),
        ({"stft": Method1STFTConfig(960, 480, "hamming")}, ValueError, "hann"),
        ({"keep_map": Method1KeepMapConfig(0.0, 1.0e-7)}, ValueError, "epsilon"),
        ({"keep_map": Method1KeepMapConfig(1.0e-8, 0.0)}, ValueError, "threshold"),
        ({"safety": Method1SafetyConfig(1.0, 5, 0.1, 0.15, 0.1)}, ValueError, "temporal"),
        ({"safety": Method1SafetyConfig(-0.1, 5, 0.1, 0.15, 0.1)}, ValueError, "temporal"),
        ({"safety": Method1SafetyConfig(0.85, 4, 0.1, 0.15, 0.1)}, ValueError, "odd"),
        ({"safety": Method1SafetyConfig(0.85, 483, 0.1, 0.15, 0.1)}, ValueError, "bin count"),
        ({"safety": Method1SafetyConfig(0.85, 5, -0.1, 0.15, 0.1)}, ValueError, "gain_floor"),
        ({"safety": Method1SafetyConfig(0.85, 5, 0.1, 1.1, 0.1)}, ValueError, "max_gain_drop"),
        ({"evaluation": None}, TypeError, "evaluation"),
    ],
)
def test_complete_config_validation_rejects_invalid_values(
    method1_config,
    replacement,
    exception,
    message,
):
    config = dataclasses.replace(method1_config, **replacement)
    with pytest.raises(exception, match=message):
        config.validate()


@pytest.mark.parametrize(
    "variant,message",
    [
        (Method1VariantConfig("Bad-ID", False, False, False, False, "dl"), "safe lowercase"),
        (Method1VariantConfig("x", False, True, False, False, "dl"), "frequency smoothing"),
        (Method1VariantConfig("x", True, False, True, False, "dl"), "gain floor"),
        (Method1VariantConfig("x", True, True, False, True, "dl"), "rate limits"),
        (Method1VariantConfig("x", False, False, False, False, "other"), "phase"),
    ],
)
def test_variant_validation_enforces_safe_ids_phase_and_cumulative_order(variant, message):
    with pytest.raises(ValueError, match=message):
        variant.validate()


def test_config_requires_complete_ablation_set_and_unique_ids(method1_config):
    missing = dataclasses.replace(method1_config, variants=method1_config.variants[:-1])
    with pytest.raises(ValueError, match="required ablations"):
        missing.validate()

    duplicate = dataclasses.replace(
        method1_config,
        variants=method1_config.variants + (method1_config.variants[0],),
    )
    with pytest.raises(ValueError, match="ids must be unique"):
        duplicate.validate()


@pytest.mark.parametrize("file_name", ["../clip.wav", r"..\clip.wav", "/tmp/clip.wav", "clip.mp3"])
def test_listening_file_names_must_be_safe_wav_basenames(method1_config, file_name):
    listening = dataclasses.replace(
        method1_config.evaluation.listening_set,
        file_names=(file_name,),
    )
    evaluation = dataclasses.replace(method1_config.evaluation, listening_set=listening)
    config = dataclasses.replace(method1_config, evaluation=evaluation)
    with pytest.raises(ValueError, match="safe WAV basenames"):
        config.validate()


def test_listening_set_rejects_duplicates_and_unknown_variants(method1_config):
    known = {variant.id for variant in method1_config.variants}
    with pytest.raises(ValueError, match="file_names.*unique"):
        Method1ListeningSetConfig(("a.wav", "a.wav"), ("raw_dl_phase",)).validate(known)
    with pytest.raises(ValueError, match="variant_ids.*unique"):
        Method1ListeningSetConfig(
            ("a.wav",),
            ("raw_dl_phase", "raw_dl_phase"),
        ).validate(known)
    with pytest.raises(ValueError, match="Unknown Method 1 listening variants"):
        Method1ListeningSetConfig(("a.wav",), ("missing",)).validate(known)


def test_missing_config_file_is_reported():
    with pytest.raises(FileNotFoundError, match="Method 1 config file not found"):
        load_method1_config("config/no-such-method1.yaml")


def test_public_config_types_reject_wrong_nested_objects(method1_config):
    for field in ("alignment", "stft", "keep_map", "safety"):
        config = dataclasses.replace(method1_config, **{field: object()})
        with pytest.raises(TypeError, match=field):
            config.validate()

    evaluation = Method1EvaluationConfig(
        clipping_threshold=1.0,
        listening_set=object(),
    )
    with pytest.raises(TypeError, match="listening_set"):
        evaluation.validate({"raw_dl_phase"})


def test_public_scalar_configs_reject_nonfinite_values():
    for epsilon in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            Method1KeepMapConfig(epsilon, 1.0e-7).validate()
    with pytest.raises(ValueError, match="finite"):
        Method1SafetyConfig(float("nan"), 5, 0.1, 0.15, 0.1).validate()
    with pytest.raises(ValueError, match="finite"):
        Method1EvaluationConfig(
            float("inf"),
            Method1ListeningSetConfig(("a.wav",), ("raw_dl_phase",)),
        ).validate({"raw_dl_phase"})
