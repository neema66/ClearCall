"""Strict configuration for independent Hybrid Method 2."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


_METHOD_MODE = "dsp_guided_level_calibration"
_SAMPLE_RATE = 48_000


@dataclasses.dataclass(frozen=True)
class Method2LevelConfig:
    """Parameters for DSP-guided whole-clip level calibration."""

    epsilon: float
    minimum_gain: float
    maximum_gain: float
    silence_rms_threshold: float

    def validate(self) -> None:
        _positive_finite("method_2.level.epsilon", self.epsilon)
        _positive_finite("method_2.level.minimum_gain", self.minimum_gain)
        _positive_finite("method_2.level.maximum_gain", self.maximum_gain)
        _nonnegative_finite(
            "method_2.level.silence_rms_threshold",
            self.silence_rms_threshold,
        )
        if self.maximum_gain < self.minimum_gain:
            raise ValueError(
                "method_2.level.maximum_gain must be greater than or equal "
                "to minimum_gain"
            )


@dataclasses.dataclass(frozen=True)
class Method2SafetyConfig:
    """Output peak and clipping-diagnostic thresholds."""

    peak_limit: float
    clipping_threshold: float

    def validate(self) -> None:
        _positive_finite("method_2.safety.peak_limit", self.peak_limit)
        _positive_finite(
            "method_2.safety.clipping_threshold",
            self.clipping_threshold,
        )
        if self.peak_limit > self.clipping_threshold:
            raise ValueError(
                "method_2.safety.peak_limit cannot exceed clipping_threshold"
            )


@dataclasses.dataclass(frozen=True)
class Method2Config:
    """Complete immutable Hybrid Method 2 configuration."""

    mode: str
    sample_rate: int
    level: Method2LevelConfig
    safety: Method2SafetyConfig

    def validate(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("method_2.mode must be a string")
        if self.mode != _METHOD_MODE:
            raise ValueError(f"method_2.mode must be {_METHOD_MODE!r}")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("method_2.sample_rate must be an integer")
        if self.sample_rate != _SAMPLE_RATE:
            raise ValueError("Hybrid Method 2 currently requires sample_rate=48000")
        if not isinstance(self.level, Method2LevelConfig):
            raise TypeError("method_2.level must be Method2LevelConfig")
        if not isinstance(self.safety, Method2SafetyConfig):
            raise TypeError("method_2.safety must be Method2SafetyConfig")
        self.level.validate()
        self.safety.validate()


def _positive_finite(name: str, value: float) -> None:
    _floating(name, value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative_finite(name: str, value: float) -> None:
    _floating(name, value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")


def _floating(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a floating-point value")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a YAML mapping")
    return value


def _exact_fields(
    data: Mapping[str, Any],
    expected: set[str],
    section: str,
) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing:
        raise ValueError(f"Missing {section} configuration key(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown {section} configuration key(s): {', '.join(sorted(unknown))}")


def load_method2_config(
    config_path: str | Path = "config/hybrid_method_2.yaml",
) -> Method2Config:
    """Load and strictly validate the standalone Method 2 YAML file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid Method 2 config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    root = _mapping(raw, "configuration root")
    _exact_fields(root, {"method_2"}, "root")
    method = _mapping(root["method_2"], "method_2")
    _exact_fields(method, {"mode", "sample_rate", "level", "safety"}, "method_2")

    level = _mapping(method["level"], "method_2.level")
    _exact_fields(
        level,
        {"epsilon", "minimum_gain", "maximum_gain", "silence_rms_threshold"},
        "method_2.level",
    )
    safety = _mapping(method["safety"], "method_2.safety")
    _exact_fields(
        safety,
        {"peak_limit", "clipping_threshold"},
        "method_2.safety",
    )

    config = Method2Config(
        mode=method["mode"],
        sample_rate=method["sample_rate"],
        level=Method2LevelConfig(
            epsilon=level["epsilon"],
            minimum_gain=level["minimum_gain"],
            maximum_gain=level["maximum_gain"],
            silence_rms_threshold=level["silence_rms_threshold"],
        ),
        safety=Method2SafetyConfig(
            peak_limit=safety["peak_limit"],
            clipping_threshold=safety["clipping_threshold"],
        ),
    )
    config.validate()
    return config
