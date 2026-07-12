"""
Configuration loader for the speech enhancement system.

Loads settings from a YAML file (default: config/default.yaml) into typed,
dot-accessible dataclasses so the rest of the codebase never hardcodes
parameters like sample rate, frame size, or file paths.

Usage:
    from senhance.config.settings import load_settings

    settings = load_settings("config/default.yaml")
    print(settings.audio.sample_rate)
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclasses.dataclass
class AudioSettings:
    sample_rate: int = 48000
    channels: int = 1
    block_size: int = 480
    input_device: Optional[str] = None
    output_device: Optional[str] = None


@dataclasses.dataclass
class NoiseEstimatorSettings:
    window_sec: float = 1.2
    # Empirically tuned against the team's VoiceBank+DEMAND subset (see
    # docs/evaluation_plan.md) -- lower than the theoretically "unbiased"
    # value because the noise estimate also feeds spectral subtraction and
    # the Wiener filter, both of which apply their own suppression on top;
    # inflating the noise estimate further compounds with them. Retune if
    # either downstream stage's aggressiveness changes materially.
    bias_correction: float = 0.5


@dataclasses.dataclass
class SpectralSubtractionSettings:
    oversubtraction_factor: float = 0.3
    spectral_floor: float = 0.2


@dataclasses.dataclass
class WienerFilterSettings:
    smoothing_factor: float = 0.7


@dataclasses.dataclass
class DSPSettings:
    frame_size_ms: int = 20
    overlap_ratio: float = 0.5
    window: str = "hann"
    noise_estimator: NoiseEstimatorSettings = dataclasses.field(
        default_factory=NoiseEstimatorSettings
    )
    spectral_subtraction: SpectralSubtractionSettings = dataclasses.field(
        default_factory=SpectralSubtractionSettings
    )
    wiener_filter: WienerFilterSettings = dataclasses.field(
        default_factory=WienerFilterSettings
    )


@dataclasses.dataclass
class DeepLearningSettings:
    enabled: bool = False
    model_name: str = "DeepFilterNet3"
    device: str = "cpu"


@dataclasses.dataclass
class EvaluationSettings:
    dataset_dir: str = "data"
    clean_subdir: str = "clean"
    noisy_subdir: str = "noisy"
    sample_rate_for_metrics: int = 16000
    pesq_mode: str = "wb"


@dataclasses.dataclass
class LoggingSettings:
    level: str = "INFO"
    log_dir: str = "logs"
    log_to_console: bool = True
    log_to_file: bool = True


@dataclasses.dataclass
class LatencyBudgetSettings:
    capture: int = 5
    buffering: int = 5
    dsp_processing: int = 10
    dl_processing: int = 20
    output: int = 5
    total_target: int = 40


@dataclasses.dataclass
class AppSettings:
    """Top-level settings object. This is the single source of truth for
    all tunable parameters in the system -- pipeline code should never
    hardcode a sample rate, frame size, or threshold; it should read it
    from an AppSettings instance passed in at construction time."""

    audio: AudioSettings = dataclasses.field(default_factory=AudioSettings)
    dsp: DSPSettings = dataclasses.field(default_factory=DSPSettings)
    deep_learning: DeepLearningSettings = dataclasses.field(
        default_factory=DeepLearningSettings
    )
    evaluation: EvaluationSettings = dataclasses.field(
        default_factory=EvaluationSettings
    )
    logging: LoggingSettings = dataclasses.field(default_factory=LoggingSettings)
    latency_budget_ms: LatencyBudgetSettings = dataclasses.field(
        default_factory=LatencyBudgetSettings
    )

    @property
    def frame_size_samples(self) -> int:
        """STFT frame size in samples, derived from frame_size_ms and sample_rate."""
        return int(self.audio.sample_rate * self.dsp.frame_size_ms / 1000)

    @property
    def hop_size_samples(self) -> int:
        """STFT hop size in samples, derived from frame size and overlap ratio."""
        return int(self.frame_size_samples * (1 - self.dsp.overlap_ratio))


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Recursively build a dataclass instance from a plain dict, only
    overriding fields that are actually present in the YAML file (so
    defaults above still apply for anything the user doesn't specify)."""
    if data is None:
        return cls()

    # NOTE: because this module uses `from __future__ import annotations`,
    # dataclass field.type values are strings, not actual classes -- we
    # must resolve them via typing.get_type_hints() before we can check
    # dataclasses.is_dataclass() on them.
    resolved_types = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        if key not in resolved_types:
            # TODO: consider warning here if the YAML has a typo'd key.
            continue
        field_type = resolved_types[key]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_settings(config_path: str | Path = "config/default.yaml") -> AppSettings:
    """
    Load application settings from a YAML file.

    Args:
        config_path: Path to a YAML config file. Missing keys fall back to
            the defaults defined in the dataclasses above.

    Returns:
        A fully populated AppSettings instance.

    Raises:
        FileNotFoundError: if config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Copy config/default.yaml as a starting point."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return _dict_to_dataclass(AppSettings, raw)
