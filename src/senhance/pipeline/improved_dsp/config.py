"""Configuration for the independent improved DSP pipeline."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclasses.dataclass(frozen=True)
class MCRAConfig:
    """Parameters for minima-controlled recursive noise estimation."""

    local_frequency_weights: tuple[float, float, float] = (0.25, 0.50, 0.25)
    power_smoothing: float = 0.8
    speech_probability_smoothing: float = 0.2
    noise_smoothing: float = 0.95
    speech_ratio_threshold: float = 5.0
    minimum_window_sec: float = 1.0
    power_calibration: float = 0.5


@dataclasses.dataclass(frozen=True)
class SpectralSubtractionConfig:
    """Parameters for the spectral-subtraction keep gain."""

    oversubtraction_factor: float = 0.3
    spectral_floor: float = 0.2


@dataclasses.dataclass(frozen=True)
class WienerConfig:
    """Parameters for decision-directed Wiener gain and soft fusion."""

    smoothing_factor: float = 0.7
    fusion_strength: float = 0.5


@dataclasses.dataclass(frozen=True)
class FinalGainSmoothingConfig:
    """Parameters for optional final gain regularization."""

    enabled: bool = True
    frequency_weights: tuple[float, float, float] = (0.25, 0.50, 0.25)
    temporal_smoothing: float = 0.3


@dataclasses.dataclass(frozen=True)
class ImprovedDSPConfig:
    """Complete tunable configuration for the improved DSP algorithm."""

    window: str = "hann"
    mcra: MCRAConfig = dataclasses.field(default_factory=MCRAConfig)
    spectral_subtraction: SpectralSubtractionConfig = dataclasses.field(
        default_factory=SpectralSubtractionConfig
    )
    wiener: WienerConfig = dataclasses.field(default_factory=WienerConfig)
    final_gain_smoothing: FinalGainSmoothingConfig = dataclasses.field(
        default_factory=FinalGainSmoothingConfig
    )

    def validate(self) -> None:
        """Raise ``ValueError`` when a parameter is outside its valid range."""

        if self.window != "hann":
            raise ValueError("The improved pipeline currently requires a Hann window")

        _validate_weights("mcra.local_frequency_weights", self.mcra.local_frequency_weights)
        _validate_unit_interval("mcra.power_smoothing", self.mcra.power_smoothing)
        _validate_unit_interval(
            "mcra.speech_probability_smoothing",
            self.mcra.speech_probability_smoothing,
        )
        _validate_unit_interval("mcra.noise_smoothing", self.mcra.noise_smoothing)
        if self.mcra.speech_ratio_threshold <= 1.0:
            raise ValueError("mcra.speech_ratio_threshold must be greater than 1")
        if self.mcra.minimum_window_sec <= 0.0:
            raise ValueError("mcra.minimum_window_sec must be positive")
        if self.mcra.power_calibration <= 0.0:
            raise ValueError("mcra.power_calibration must be positive")

        if self.spectral_subtraction.oversubtraction_factor < 0.0:
            raise ValueError("spectral_subtraction.oversubtraction_factor cannot be negative")
        _validate_closed_unit_interval(
            "spectral_subtraction.spectral_floor",
            self.spectral_subtraction.spectral_floor,
        )

        _validate_unit_interval("wiener.smoothing_factor", self.wiener.smoothing_factor)
        _validate_closed_unit_interval("wiener.fusion_strength", self.wiener.fusion_strength)

        _validate_weights(
            "final_gain_smoothing.frequency_weights",
            self.final_gain_smoothing.frequency_weights,
        )
        _validate_unit_interval(
            "final_gain_smoothing.temporal_smoothing",
            self.final_gain_smoothing.temporal_smoothing,
        )


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= value < 1")


def _validate_closed_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= value <= 1")


def _validate_weights(name: str, weights: tuple[float, float, float]) -> None:
    if len(weights) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{name} cannot contain negative values")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _strict_fields(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {section} configuration key(s): {names}")


def _three_weights(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _build_config(data: Mapping[str, Any]) -> ImprovedDSPConfig:
    _strict_fields(
        data,
        {"window", "mcra", "spectral_subtraction", "wiener", "final_gain_smoothing"},
        "improved_dsp",
    )

    mcra_data = _require_mapping(data.get("mcra", {}), "improved_dsp.mcra")
    _strict_fields(
        mcra_data,
        {
            "local_frequency_weights",
            "power_smoothing",
            "speech_probability_smoothing",
            "noise_smoothing",
            "speech_ratio_threshold",
            "minimum_window_sec",
            "power_calibration",
        },
        "improved_dsp.mcra",
    )
    mcra_kwargs = dict(mcra_data)
    if "local_frequency_weights" in mcra_kwargs:
        mcra_kwargs["local_frequency_weights"] = _three_weights(
            mcra_kwargs["local_frequency_weights"],
            "improved_dsp.mcra.local_frequency_weights",
        )

    subtraction_data = _require_mapping(
        data.get("spectral_subtraction", {}),
        "improved_dsp.spectral_subtraction",
    )
    _strict_fields(
        subtraction_data,
        {"oversubtraction_factor", "spectral_floor"},
        "improved_dsp.spectral_subtraction",
    )

    wiener_data = _require_mapping(data.get("wiener", {}), "improved_dsp.wiener")
    _strict_fields(
        wiener_data,
        {"smoothing_factor", "fusion_strength"},
        "improved_dsp.wiener",
    )

    smoothing_data = _require_mapping(
        data.get("final_gain_smoothing", {}),
        "improved_dsp.final_gain_smoothing",
    )
    _strict_fields(
        smoothing_data,
        {"enabled", "frequency_weights", "temporal_smoothing"},
        "improved_dsp.final_gain_smoothing",
    )
    smoothing_kwargs = dict(smoothing_data)
    if "frequency_weights" in smoothing_kwargs:
        smoothing_kwargs["frequency_weights"] = _three_weights(
            smoothing_kwargs["frequency_weights"],
            "improved_dsp.final_gain_smoothing.frequency_weights",
        )

    config = ImprovedDSPConfig(
        window=str(data.get("window", "hann")),
        mcra=MCRAConfig(**mcra_kwargs),
        spectral_subtraction=SpectralSubtractionConfig(**dict(subtraction_data)),
        wiener=WienerConfig(**dict(wiener_data)),
        final_gain_smoothing=FinalGainSmoothingConfig(**smoothing_kwargs),
    )
    config.validate()
    return config


def load_improved_dsp_config(
    config_path: str | Path = "config/improved_dsp.yaml",
) -> ImprovedDSPConfig:
    """Load strict improved-DSP settings from a standalone YAML file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Improved DSP config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    root = _require_mapping(raw, "configuration root")
    _strict_fields(root, {"improved_dsp"}, "root")
    dsp_data = _require_mapping(root.get("improved_dsp", {}), "improved_dsp")
    return _build_config(dsp_data)
