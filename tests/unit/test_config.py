"""Unit tests for the configuration loader."""

import pytest

from senhance.config.settings import AppSettings, load_settings


def test_load_default_config():
    settings = load_settings("config/default.yaml")
    assert settings.audio.sample_rate == 48000
    assert settings.dsp.frame_size_ms == 20


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_settings("config/does_not_exist.yaml")


def test_derived_frame_size_samples():
    settings = AppSettings()
    expected = int(settings.audio.sample_rate * settings.dsp.frame_size_ms / 1000)
    assert settings.frame_size_samples == expected


def test_derived_hop_size_samples():
    settings = AppSettings()
    expected = int(settings.frame_size_samples * (1 - settings.dsp.overlap_ratio))
    assert settings.hop_size_samples == expected
