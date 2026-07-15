"""Configuration tests for the independent improved DSP."""

from pathlib import Path

import pytest

from senhance.pipeline.improved_dsp.config import (
    ImprovedDSPConfig,
    MCRAConfig,
    load_improved_dsp_config,
)


def test_project_config_loads() -> None:
    config = load_improved_dsp_config("config/improved_dsp.yaml")

    assert config.mcra.power_calibration == 0.5
    assert config.wiener.fusion_strength == 0.5
    assert config.final_gain_smoothing.enabled is True


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("improved_dsp:\n  mcra:\n    power_smothing: 0.8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="power_smothing"):
        load_improved_dsp_config(path)


def test_invalid_frequency_weights_are_rejected() -> None:
    config = ImprovedDSPConfig(mcra=MCRAConfig(local_frequency_weights=(0.2, 0.2, 0.2)))

    with pytest.raises(ValueError, match="sum to 1"):
        config.validate()
