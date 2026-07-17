"""Strict, hybrid-owned configuration for alignment and paired STFT."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


@dataclasses.dataclass(frozen=True)
class HybridAlignmentConfig:
    """Fixed offline alignment selected from a separate diagnostic step.

    A positive delay means the candidate signal lags the reference, so the
    faster reference path is delayed. A negative delay means the candidate
    leads, so the faster candidate path is delayed. For Method 3, reference is
    the selected DSP output and candidate is the DL output.
    """

    delay_samples: int = 0

    def validate(self) -> None:
        _require_integer(self.delay_samples, "hybrid.alignment.delay_samples")


@dataclasses.dataclass(frozen=True)
class HybridSTFTConfig:
    """Framing geometry owned by the hybrid rather than a pure-DSP method."""

    frame_size: int = 960
    hop_size: int = 480
    window: str = "hann"

    def validate(self) -> None:
        frame_size = _require_integer(self.frame_size, "hybrid.stft.frame_size")
        hop_size = _require_integer(self.hop_size, "hybrid.stft.hop_size")
        if frame_size <= 0 or hop_size <= 0:
            raise ValueError("hybrid STFT frame_size and hop_size must be positive")
        if frame_size != 2 * hop_size:
            raise ValueError("hybrid STFT requires 50% overlap: frame_size must equal 2 * hop_size")
        if (frame_size, hop_size) != (960, 480):
            raise ValueError("hybrid STFT currently requires frame_size=960 and hop_size=480")
        if not isinstance(self.window, str) or self.window != "hann":
            raise ValueError("hybrid.stft.window must be 'hann'")

    @property
    def num_frequency_bins(self) -> int:
        return self.frame_size // 2 + 1


@dataclasses.dataclass(frozen=True)
class HybridConfig:
    """Complete configuration for the independent hybrid signal core."""

    sample_rate: int = 48_000
    alignment: HybridAlignmentConfig = dataclasses.field(default_factory=HybridAlignmentConfig)
    stft: HybridSTFTConfig = dataclasses.field(default_factory=HybridSTFTConfig)

    def validate(self) -> None:
        sample_rate = _require_integer(self.sample_rate, "hybrid.sample_rate")
        if sample_rate <= 0:
            raise ValueError("hybrid.sample_rate must be positive")
        if sample_rate != 48_000:
            raise ValueError("hybrid.sample_rate must be 48000 Hz")
        if not isinstance(self.alignment, HybridAlignmentConfig):
            raise TypeError("hybrid.alignment must be HybridAlignmentConfig")
        if not isinstance(self.stft, HybridSTFTConfig):
            raise TypeError("hybrid.stft must be HybridSTFTConfig")
        self.alignment.validate()
        self.stft.validate()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: set[str],
    section: str,
) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"Unknown {section} configuration key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing {section} configuration key(s): {', '.join(sorted(missing))}")


def load_hybrid_config(
    config_path: str | Path = "config/hybrid.yaml",
) -> HybridConfig:
    """Load the explicit hybrid schema and reject missing or unknown keys."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _require_mapping(raw, "configuration root")
    _require_exact_fields(root, {"hybrid"}, "root")
    hybrid = _require_mapping(root["hybrid"], "hybrid")
    _require_exact_fields(
        hybrid,
        {"sample_rate", "alignment", "stft"},
        "hybrid",
    )
    alignment = _require_mapping(hybrid["alignment"], "hybrid.alignment")
    _require_exact_fields(
        alignment,
        {"delay_samples"},
        "hybrid.alignment",
    )
    stft = _require_mapping(hybrid["stft"], "hybrid.stft")
    _require_exact_fields(
        stft,
        {"frame_size", "hop_size", "window"},
        "hybrid.stft",
    )

    config = HybridConfig(
        sample_rate=_require_integer(hybrid["sample_rate"], "hybrid.sample_rate"),
        alignment=HybridAlignmentConfig(
            delay_samples=_require_integer(
                alignment["delay_samples"],
                "hybrid.alignment.delay_samples",
            )
        ),
        stft=HybridSTFTConfig(
            frame_size=_require_integer(stft["frame_size"], "hybrid.stft.frame_size"),
            hop_size=_require_integer(stft["hop_size"], "hybrid.stft.hop_size"),
            window=stft["window"],
        ),
    )
    config.validate()
    return config
