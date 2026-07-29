"""Strict configuration tests for Method 3 Version 1 experiments."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from senhance.pipeline.hybrid.method3.method3_config import (
    Method3DSPBackendConfig,
    Method3DSPConfig,
    Method3ListeningSetConfig,
    Method3WaveformConfig,
    load_method3_config,
)


def _config(**overrides):
    values = {
        "mode": "fixed_waveform",
        "dsp": Method3DSPConfig(
            selected_backend="improved_dsp",
            backends=(
                Method3DSPBackendConfig("improved_dsp", 0),
                Method3DSPBackendConfig("legacy_dsp", 0),
            ),
        ),
        "alpha_sweep": (0.0, 0.25, 0.5, 0.7, 1.0),
        "clipping_threshold": 1.0,
        "listening_set": Method3ListeningSetConfig(
            file_names=("p232_005.wav",),
            alpha_values=(0.0, 0.5, 1.0),
        ),
    }
    values.update(overrides)
    return Method3WaveformConfig(**values)


def test_project_method3_config_loads_exact_experimental_values():
    config = load_method3_config("config/hybrid_method_3.yaml")

    assert config.mode == "fixed_waveform"
    assert config.dsp.selected_backend == "improved_dsp"
    assert config.dsp.backend() == Method3DSPBackendConfig("improved_dsp", 0)
    assert config.dsp.backend("legacy_dsp") == Method3DSPBackendConfig(
        "legacy_dsp",
        0,
    )
    assert config.alpha_sweep == (0.0, 0.25, 0.5, 0.7, 1.0)
    assert config.clipping_threshold == 1.0
    assert config.listening_set.file_names == (
        "p232_005.wav",
        "p232_010.wav",
        "p232_019.wav",
    )
    assert config.listening_set.alpha_values == config.alpha_sweep


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        (
            "method_3:\n  mode: fixed_waveform\n  alpha_sweep: [0.0, 0.5, 1.0]\n"
            "  clipping_threshold: 1.0\n  listening_set:\n"
            "    file_names: [p232_005.wav]\n    alpha_values: [0.0, 0.5, 1.0]\n"
            "extra: true\n",
            "Unknown root",
        ),
        (
            "method_3:\n  mode: fixed_waveform\n  alpha_sweep: [0.0, 0.5, 1.0]\n"
            "  clipping_threshold: 1.0\n",
            "Missing method_3",
        ),
        (
            "method_3:\n  mode: fixed_waveform\n  dsp:\n"
            "    selected_backend: improved_dsp\n    backends:\n"
            "      improved_dsp:\n        dl_minus_dsp_delay_samples: 0\n"
            "  alpha_sweep: [0.0, 0.5, 1.0]\n"
            "  clipping_threshold: 1.0\n  listening_set:\n"
            "    file_names: [p232_005.wav]\n    alpha_values: [0.0, 0.5, 1.0]\n"
            "    selected_dsp: improved_dsp\n",
            "selected_dsp",
        ),
        (
            "method_3:\n  mode: fixed_waveform\n  dsp:\n"
            "    selected_backend: improved_dsp\n    backends:\n"
            "      improved_dsp:\n        dl_minus_dsp_delay_samples: 0\n"
            "  alpha_sweep: [0.0, 0.5, 1.0]\n"
            "  clipping_threshold: 1.0\n  listening_set: []\n",
            "must be a YAML mapping",
        ),
    ],
)
def test_strict_loader_rejects_missing_unknown_or_wrong_sections(
    tmp_path,
    yaml_text,
    message,
):
    path = tmp_path / "method3.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_method3_config(path)


@pytest.mark.parametrize(
    "field, value",
    [
        ("alpha_sweep", [0, 0.5, 1.0]),
        ("alpha_sweep", [0.0, True, 1.0]),
        ("clipping_threshold", 1),
        ("listening_alpha_values", [0.0, 0.5, 1]),
    ],
)
def test_loader_rejects_implicit_numeric_coercion(tmp_path, field, value):
    raw = {
        "method_3": {
            "mode": "fixed_waveform",
            "dsp": {
                "selected_backend": "improved_dsp",
                "backends": {
                    "improved_dsp": {"dl_minus_dsp_delay_samples": 0},
                    "legacy_dsp": {"dl_minus_dsp_delay_samples": 0},
                },
            },
            "alpha_sweep": [0.0, 0.5, 1.0],
            "clipping_threshold": 1.0,
            "listening_set": {
                "file_names": ["p232_005.wav"],
                "alpha_values": [0.0, 0.5, 1.0],
            },
        }
    }
    if field == "listening_alpha_values":
        raw["method_3"]["listening_set"]["alpha_values"] = value
    else:
        raw["method_3"][field] = value
    path = tmp_path / "method3.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match="floating-point"):
        load_method3_config(path)


@pytest.mark.parametrize(
    "config, exception, message",
    [
        (_config(mode="waveform"), ValueError, "fixed_waveform"),
        (_config(alpha_sweep=(0.0, 1.0)), ValueError, "at least three"),
        (
            _config(alpha_sweep=(0.0, 0.5, 0.5, 1.0)),
            ValueError,
            "strictly increasing",
        ),
        (
            _config(alpha_sweep=(0.0, 0.7, 0.5, 1.0)),
            ValueError,
            "strictly increasing",
        ),
        (_config(alpha_sweep=(-0.1, 0.0, 0.5, 1.0)), ValueError, "0.0 <= alpha"),
        (_config(alpha_sweep=(0.0, 0.25, 1.0)), ValueError, "must include"),
        (_config(alpha_sweep=(0.0, 0.5, 1.1)), ValueError, "0.0 <= alpha"),
        (_config(alpha_sweep=(0.0, 0.5, float("nan"), 1.0)), ValueError, "finite"),
        (_config(clipping_threshold=0.0), ValueError, "must be positive"),
        (_config(clipping_threshold=float("inf")), ValueError, "must be finite"),
        (_config(listening_set=None), TypeError, "must be Method3ListeningSetConfig"),
        (
            _config(
                listening_set=Method3ListeningSetConfig(
                    file_names=("../p232_005.wav",),
                    alpha_values=(0.0, 0.5, 1.0),
                )
            ),
            ValueError,
            "safe WAV basenames",
        ),
        (
            _config(
                listening_set=Method3ListeningSetConfig(
                    file_names=(r"..\p232_005.wav",),
                    alpha_values=(0.0, 0.5, 1.0),
                )
            ),
            ValueError,
            "safe WAV basenames",
        ),
        (
            _config(
                listening_set=Method3ListeningSetConfig(
                    file_names=("p232_005.wav", "p232_005.wav"),
                    alpha_values=(0.0, 0.5, 1.0),
                )
            ),
            ValueError,
            "must be unique",
        ),
        (
            _config(
                listening_set=Method3ListeningSetConfig(
                    file_names=("p232_005.wav",),
                    alpha_values=(0.0, 0.5, 0.5, 1.0),
                )
            ),
            ValueError,
            "alpha values must be unique",
        ),
        (
            _config(
                listening_set=Method3ListeningSetConfig(
                    file_names=("p232_005.wav",),
                    alpha_values=(0.0, 0.6, 1.0),
                )
            ),
            ValueError,
            "present in alpha_sweep",
        ),
    ],
)
def test_dataclass_validation_rejects_invalid_experiment_settings(
    config,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        config.validate()


def test_method3_config_is_frozen():
    config = _config()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.mode = "other"


def test_replaceable_dsp_selection_changes_only_selected_profile():
    config = _config()

    switched = dataclasses.replace(
        config,
        dsp=dataclasses.replace(config.dsp, selected_backend="legacy_dsp"),
    )
    switched.validate()

    assert switched.dsp.backend().id == "legacy_dsp"
    assert switched.alpha_sweep == config.alpha_sweep


@pytest.mark.parametrize(
    "dsp,exception,message",
    [
        (
            Method3DSPConfig(
                selected_backend="missing_dsp",
                backends=(Method3DSPBackendConfig("improved_dsp", 0),),
            ),
            ValueError,
            "selected.*must exist",
        ),
        (
            Method3DSPConfig(
                selected_backend="improved_dsp",
                backends=(
                    Method3DSPBackendConfig("improved_dsp", 0),
                    Method3DSPBackendConfig("improved_dsp", 1),
                ),
            ),
            ValueError,
            "ids must be unique",
        ),
        (
            Method3DSPConfig(
                selected_backend="improved_dsp",
                backends=(Method3DSPBackendConfig("improved_dsp", True),),
            ),
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_dsp_selection_profiles_are_strict(dsp, exception, message):
    config = _config(dsp=dsp)

    with pytest.raises(exception, match=message):
        config.validate()


def test_missing_method3_config_file_raises():
    with pytest.raises(FileNotFoundError, match="Method 3 config file not found"):
        load_method3_config("config/does-not-exist-method3.yaml")
