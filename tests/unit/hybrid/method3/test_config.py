"""Strict configuration tests for the hybrid-owned signal core."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from senhance.config.settings import AppSettings
from senhance.pipeline.hybrid.method3.config import (
    HybridAlignmentConfig,
    HybridConfig,
    HybridSTFTConfig,
    load_hybrid_config,
)


def test_project_hybrid_config_loads_exact_milestone_2_values():
    config = load_hybrid_config("config/hybrid.yaml")

    assert config.sample_rate == 48_000
    assert config.alignment.delay_samples == 0
    assert config.stft.frame_size == 960
    assert config.stft.hop_size == 480
    assert config.stft.window == "hann"
    assert config.stft.num_frequency_bins == 481


def test_hybrid_geometry_is_not_derived_from_app_dsp_settings():
    app_settings = AppSettings()
    app_settings.dsp.frame_size_ms = 10
    app_settings.dsp.overlap_ratio = 0.25

    hybrid = load_hybrid_config("config/hybrid.yaml")

    assert app_settings.frame_size_samples != hybrid.stft.frame_size
    assert app_settings.hop_size_samples != hybrid.stft.hop_size
    assert (hybrid.stft.frame_size, hybrid.stft.hop_size) == (960, 480)


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        (
            "hybrid:\n  sample_rate: 48000\n  alignment:\n"
            "    delay_samples: 0\n  stft:\n    frame_size: 960\n"
            "    hop_size: 480\n    window: hann\nextra: 1\n",
            "Unknown root",
        ),
        (
            "hybrid:\n  sample_rate: 48000\n  alignment:\n"
            "    delay_samples: 0\n  stft:\n    frame_size: 960\n"
            "    hop_size: 480\n    window: hann\n  selected_dsp: improved_dsp\n",
            "selected_dsp",
        ),
        (
            "hybrid:\n  sample_rate: 48000\n  alignment:\n" "    delay_samples: 0\n",
            "Missing hybrid",
        ),
        (
            "hybrid:\n  sample_rate: 48000\n  alignment: {}\n"
            "  stft:\n    frame_size: 960\n    hop_size: 480\n"
            "    window: hann\n",
            "Missing hybrid.alignment",
        ),
        (
            "hybrid:\n  sample_rate: 48000\n  alignment:\n"
            "    delay_samples: 0\n  stft:\n    frame_size: 960\n"
            "    hop_size: 480\n    window: hann\n    windo: hann\n",
            "windo",
        ),
    ],
)
def test_strict_loader_rejects_unknown_or_missing_keys(tmp_path, yaml_text, message):
    path = tmp_path / "hybrid.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_hybrid_config(path)


@pytest.mark.parametrize(
    "field, value",
    [
        ("sample_rate", True),
        ("sample_rate", 48_000.0),
        ("delay_samples", False),
        ("delay_samples", 0.0),
        ("frame_size", "960"),
        ("hop_size", 480.0),
    ],
)
def test_loader_rejects_implicit_numeric_coercion(tmp_path, field, value):
    values = {
        "sample_rate": 48_000,
        "delay_samples": 0,
        "frame_size": 960,
        "hop_size": 480,
    }
    values[field] = value
    path = tmp_path / "hybrid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "hybrid": {
                    "sample_rate": values["sample_rate"],
                    "alignment": {"delay_samples": values["delay_samples"]},
                    "stft": {
                        "frame_size": values["frame_size"],
                        "hop_size": values["hop_size"],
                        "window": "hann",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must be an integer"):
        load_hybrid_config(path)


@pytest.mark.parametrize(
    "config, message",
    [
        (HybridConfig(sample_rate=0), "sample_rate must be positive"),
        (HybridConfig(sample_rate=16_000), "sample_rate must be 48000 Hz"),
        (
            HybridConfig(stft=HybridSTFTConfig(frame_size=960, hop_size=320)),
            "50% overlap",
        ),
        (
            HybridConfig(stft=HybridSTFTConfig(frame_size=1024, hop_size=512)),
            "frame_size=960 and hop_size=480",
        ),
        (
            HybridConfig(stft=HybridSTFTConfig(window="hamming")),
            "window must be 'hann'",
        ),
        (
            HybridConfig(alignment=HybridAlignmentConfig(delay_samples=True)),
            "delay_samples must be an integer",
        ),
    ],
)
def test_dataclass_validation_rejects_invalid_values(config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        config.validate()


def test_hybrid_config_is_frozen():
    config = HybridConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.sample_rate = 16_000


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Hybrid config file not found"):
        load_hybrid_config("config/does-not-exist-hybrid.yaml")
