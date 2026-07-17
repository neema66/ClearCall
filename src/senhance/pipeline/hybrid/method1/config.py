"""Strict configuration for independent offline Hybrid Method 1."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


_SAFE_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_PHASE_SOURCES = {"noisy", "dl"}


@dataclasses.dataclass(frozen=True)
class Method1AlignmentConfig:
    """Reviewed offline delay using ``DL index - noisy index`` convention."""

    dl_minus_noisy_delay_samples: int

    def validate(self) -> None:
        _require_integer(
            self.dl_minus_noisy_delay_samples,
            "method_1.alignment.dl_minus_noisy_delay_samples",
        )


@dataclasses.dataclass(frozen=True)
class Method1STFTConfig:
    """Method-1-owned analysis and reconstruction geometry."""

    frame_size: int
    hop_size: int
    window: str

    def validate(self) -> None:
        frame_size = _require_integer(self.frame_size, "method_1.stft.frame_size")
        hop_size = _require_integer(self.hop_size, "method_1.stft.hop_size")
        if (frame_size, hop_size) != (960, 480):
            raise ValueError("Method 1 currently requires frame_size=960 and hop_size=480")
        if frame_size != 2 * hop_size:
            raise ValueError("Method 1 requires 50% overlap")
        if not isinstance(self.window, str) or self.window != "hann":
            raise ValueError("method_1.stft.window must be 'hann'")

    @property
    def num_frequency_bins(self) -> int:
        return self.frame_size // 2 + 1


@dataclasses.dataclass(frozen=True)
class Method1KeepMapConfig:
    """Numerical policy for estimating the raw DL keep map."""

    epsilon: float
    low_energy_threshold: float

    def validate(self) -> None:
        epsilon = _require_float(self.epsilon, "method_1.keep_map.epsilon")
        threshold = _require_float(
            self.low_energy_threshold,
            "method_1.keep_map.low_energy_threshold",
        )
        if epsilon <= 0.0:
            raise ValueError("method_1.keep_map.epsilon must be positive")
        if threshold <= 0.0:
            raise ValueError("method_1.keep_map.low_energy_threshold must be positive")


@dataclasses.dataclass(frozen=True)
class Method1SafetyConfig:
    """DSP regularization parameters shared by ordered ablations."""

    temporal_smoothing: float
    frequency_kernel_bins: int
    gain_floor: float
    max_gain_drop_per_frame: float
    max_gain_rise_per_frame: float

    def validate(self) -> None:
        temporal = _require_float(
            self.temporal_smoothing,
            "method_1.safety.temporal_smoothing",
        )
        if temporal < 0.0 or temporal >= 1.0:
            raise ValueError("temporal_smoothing must satisfy 0.0 <= value < 1.0")
        kernel = _require_integer(
            self.frequency_kernel_bins,
            "method_1.safety.frequency_kernel_bins",
        )
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("frequency_kernel_bins must be a positive odd integer")
        for name, value in (
            ("gain_floor", self.gain_floor),
            ("max_gain_drop_per_frame", self.max_gain_drop_per_frame),
            ("max_gain_rise_per_frame", self.max_gain_rise_per_frame),
        ):
            normalized = _require_float(value, f"method_1.safety.{name}")
            if normalized < 0.0 or normalized > 1.0:
                raise ValueError(f"{name} must satisfy 0.0 <= value <= 1.0")


@dataclasses.dataclass(frozen=True)
class Method1VariantConfig:
    """One explicit cumulative ablation or reconstruction-phase comparison."""

    id: str
    temporal_smoothing: bool
    frequency_smoothing: bool
    gain_floor: bool
    rate_limits: bool
    reconstruction_phase: str

    def validate(self) -> None:
        if not isinstance(self.id, str) or _SAFE_ID.fullmatch(self.id) is None:
            raise ValueError("Method 1 variant id must be a safe lowercase identifier")
        for name in (
            "temporal_smoothing",
            "frequency_smoothing",
            "gain_floor",
            "rate_limits",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"method_1.variants.{self.id}.{name} must be boolean")
        if self.reconstruction_phase not in _PHASE_SOURCES:
            raise ValueError("Method 1 reconstruction_phase must be 'noisy' or 'dl'")
        if self.frequency_smoothing and not self.temporal_smoothing:
            raise ValueError("frequency smoothing requires the temporal ablation stage")
        if self.gain_floor and not self.frequency_smoothing:
            raise ValueError("gain floor requires temporal and frequency stages")
        if self.rate_limits and not self.gain_floor:
            raise ValueError("rate limits require temporal, frequency, and floor stages")


@dataclasses.dataclass(frozen=True)
class Method1ListeningSetConfig:
    """Development clips and Method 1 variants exported for blind review."""

    file_names: tuple[str, ...]
    variant_ids: tuple[str, ...]

    def validate(self, known_variant_ids: set[str]) -> None:
        if not self.file_names or len(self.file_names) != len(set(self.file_names)):
            raise ValueError("Method 1 listening file_names must be non-empty and unique")
        for file_name in self.file_names:
            if (
                not isinstance(file_name, str)
                or Path(file_name).name != file_name
                or "/" in file_name
                or "\\" in file_name
                or Path(file_name).suffix.lower() != ".wav"
            ):
                raise ValueError("Method 1 listening file names must be safe WAV basenames")
        if not self.variant_ids or len(self.variant_ids) != len(set(self.variant_ids)):
            raise ValueError("Method 1 listening variant_ids must be non-empty and unique")
        if any(
            not isinstance(variant_id, str) or _SAFE_ID.fullmatch(variant_id) is None
            for variant_id in self.variant_ids
        ):
            raise ValueError("Method 1 listening variant_ids must be safe identifiers")
        unknown = set(self.variant_ids) - known_variant_ids
        if unknown:
            raise ValueError(f"Unknown Method 1 listening variants: {sorted(unknown)}")


@dataclasses.dataclass(frozen=True)
class Method1EvaluationConfig:
    """Objective and listening-output safety policy."""

    clipping_threshold: float
    listening_set: Method1ListeningSetConfig

    def validate(self, known_variant_ids: set[str]) -> None:
        threshold = _require_float(
            self.clipping_threshold,
            "method_1.evaluation.clipping_threshold",
        )
        if threshold <= 0.0:
            raise ValueError("Method 1 clipping_threshold must be positive")
        if not isinstance(self.listening_set, Method1ListeningSetConfig):
            raise TypeError("method_1.evaluation.listening_set has invalid type")
        self.listening_set.validate(known_variant_ids)


@dataclasses.dataclass(frozen=True)
class Method1Config:
    """Complete independent Method 1 configuration."""

    mode: str
    sample_rate: int
    alignment: Method1AlignmentConfig
    stft: Method1STFTConfig
    keep_map: Method1KeepMapConfig
    safety: Method1SafetyConfig
    variants: tuple[Method1VariantConfig, ...]
    evaluation: Method1EvaluationConfig

    def validate(self) -> None:
        if self.mode != "dl_keep_map_safety_layer":
            raise ValueError("method_1.mode must be 'dl_keep_map_safety_layer'")
        if _require_integer(self.sample_rate, "method_1.sample_rate") != 48_000:
            raise ValueError("Method 1 currently requires sample_rate=48000")
        for value, expected_type, name in (
            (self.alignment, Method1AlignmentConfig, "alignment"),
            (self.stft, Method1STFTConfig, "stft"),
            (self.keep_map, Method1KeepMapConfig, "keep_map"),
            (self.safety, Method1SafetyConfig, "safety"),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"method_1.{name} has invalid type")
            value.validate()
        if self.safety.frequency_kernel_bins > self.stft.num_frequency_bins:
            raise ValueError("frequency_kernel_bins cannot exceed the STFT bin count")
        if not self.variants or not all(
            isinstance(value, Method1VariantConfig) for value in self.variants
        ):
            raise ValueError("method_1.variants must contain Method1VariantConfig values")
        for variant in self.variants:
            variant.validate()
        variant_ids = [variant.id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Method 1 variant ids must be unique")
        required_semantics = {
            "raw_dl_phase": (False, False, False, False, "dl"),
            "raw_noisy_phase": (False, False, False, False, "noisy"),
            "temporal_dl_phase": (True, False, False, False, "dl"),
            "temporal_frequency_dl_phase": (True, True, False, False, "dl"),
            "temporal_frequency_floor_dl_phase": (True, True, True, False, "dl"),
            "full_dl_phase": (True, True, True, True, "dl"),
            "full_noisy_phase": (True, True, True, True, "noisy"),
        }
        missing = set(required_semantics) - set(variant_ids)
        if missing:
            raise ValueError(f"Method 1 required ablations are missing: {sorted(missing)}")
        variants_by_id = {variant.id: variant for variant in self.variants}
        for variant_id, expected_semantics in required_semantics.items():
            variant = variants_by_id[variant_id]
            actual = (
                variant.temporal_smoothing,
                variant.frequency_smoothing,
                variant.gain_floor,
                variant.rate_limits,
                variant.reconstruction_phase,
            )
            if actual != expected_semantics:
                raise ValueError(
                    f"Method 1 required ablation {variant_id!r} has incorrect semantics"
                )
        if not isinstance(self.evaluation, Method1EvaluationConfig):
            raise TypeError("method_1.evaluation has invalid type")
        self.evaluation.validate(set(variant_ids))

    def variant(self, variant_id: str) -> Method1VariantConfig:
        if not isinstance(variant_id, str):
            raise TypeError("variant_id must be a string")
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ValueError(f"unknown Method 1 variant: {variant_id!r}")


def load_method1_config(
    config_path: str | Path = "config/hybrid_method_1.yaml",
) -> Method1Config:
    """Load Method 1 settings while rejecting missing and unknown keys."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Method 1 config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "configuration root")
    _exact(root, {"method_1"}, "root")
    method = _mapping(root["method_1"], "method_1")
    _exact(
        method,
        {
            "mode",
            "sample_rate",
            "alignment",
            "stft",
            "keep_map",
            "safety",
            "variants",
            "evaluation",
        },
        "method_1",
    )

    alignment = _mapping(method["alignment"], "method_1.alignment")
    _exact(alignment, {"dl_minus_noisy_delay_samples"}, "method_1.alignment")
    stft = _mapping(method["stft"], "method_1.stft")
    _exact(stft, {"frame_size", "hop_size", "window"}, "method_1.stft")
    keep_map = _mapping(method["keep_map"], "method_1.keep_map")
    _exact(keep_map, {"epsilon", "low_energy_threshold"}, "method_1.keep_map")
    safety = _mapping(method["safety"], "method_1.safety")
    _exact(
        safety,
        {
            "temporal_smoothing",
            "frequency_kernel_bins",
            "gain_floor",
            "max_gain_drop_per_frame",
            "max_gain_rise_per_frame",
        },
        "method_1.safety",
    )

    raw_variants = _list(method["variants"], "method_1.variants")
    variants: list[Method1VariantConfig] = []
    variant_fields = {
        "id",
        "temporal_smoothing",
        "frequency_smoothing",
        "gain_floor",
        "rate_limits",
        "reconstruction_phase",
    }
    for index, raw_variant in enumerate(raw_variants):
        variant = _mapping(raw_variant, f"method_1.variants[{index}]")
        _exact(variant, variant_fields, f"method_1.variants[{index}]")
        variants.append(
            Method1VariantConfig(
                id=_string(variant["id"], f"method_1.variants[{index}].id"),
                temporal_smoothing=_boolean(
                    variant["temporal_smoothing"],
                    f"method_1.variants[{index}].temporal_smoothing",
                ),
                frequency_smoothing=_boolean(
                    variant["frequency_smoothing"],
                    f"method_1.variants[{index}].frequency_smoothing",
                ),
                gain_floor=_boolean(
                    variant["gain_floor"],
                    f"method_1.variants[{index}].gain_floor",
                ),
                rate_limits=_boolean(
                    variant["rate_limits"],
                    f"method_1.variants[{index}].rate_limits",
                ),
                reconstruction_phase=_string(
                    variant["reconstruction_phase"],
                    f"method_1.variants[{index}].reconstruction_phase",
                ),
            )
        )

    evaluation = _mapping(method["evaluation"], "method_1.evaluation")
    _exact(evaluation, {"clipping_threshold", "listening_set"}, "method_1.evaluation")
    listening = _mapping(
        evaluation["listening_set"],
        "method_1.evaluation.listening_set",
    )
    _exact(listening, {"file_names", "variant_ids"}, "method_1.evaluation.listening_set")

    config = Method1Config(
        mode=_string(method["mode"], "method_1.mode"),
        sample_rate=_require_integer(method["sample_rate"], "method_1.sample_rate"),
        alignment=Method1AlignmentConfig(
            dl_minus_noisy_delay_samples=_require_integer(
                alignment["dl_minus_noisy_delay_samples"],
                "method_1.alignment.dl_minus_noisy_delay_samples",
            )
        ),
        stft=Method1STFTConfig(
            frame_size=_require_integer(stft["frame_size"], "method_1.stft.frame_size"),
            hop_size=_require_integer(stft["hop_size"], "method_1.stft.hop_size"),
            window=_string(stft["window"], "method_1.stft.window"),
        ),
        keep_map=Method1KeepMapConfig(
            epsilon=_require_float(keep_map["epsilon"], "method_1.keep_map.epsilon"),
            low_energy_threshold=_require_float(
                keep_map["low_energy_threshold"],
                "method_1.keep_map.low_energy_threshold",
            ),
        ),
        safety=Method1SafetyConfig(
            temporal_smoothing=_require_float(
                safety["temporal_smoothing"],
                "method_1.safety.temporal_smoothing",
            ),
            frequency_kernel_bins=_require_integer(
                safety["frequency_kernel_bins"],
                "method_1.safety.frequency_kernel_bins",
            ),
            gain_floor=_require_float(safety["gain_floor"], "method_1.safety.gain_floor"),
            max_gain_drop_per_frame=_require_float(
                safety["max_gain_drop_per_frame"],
                "method_1.safety.max_gain_drop_per_frame",
            ),
            max_gain_rise_per_frame=_require_float(
                safety["max_gain_rise_per_frame"],
                "method_1.safety.max_gain_rise_per_frame",
            ),
        ),
        variants=tuple(variants),
        evaluation=Method1EvaluationConfig(
            clipping_threshold=_require_float(
                evaluation["clipping_threshold"],
                "method_1.evaluation.clipping_threshold",
            ),
            listening_set=Method1ListeningSetConfig(
                file_names=tuple(
                    _string(value, "method_1.evaluation.listening_set.file_names[]")
                    for value in _list(
                        listening["file_names"],
                        "method_1.evaluation.listening_set.file_names",
                    )
                ),
                variant_ids=tuple(
                    _string(value, "method_1.evaluation.listening_set.variant_ids[]")
                    for value in _list(
                        listening["variant_ids"],
                        "method_1.evaluation.listening_set.variant_ids",
                    )
                ),
            ),
        ),
    )
    config.validate()
    return config


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML list")
    return value


def _exact(data: Mapping[str, Any], expected: set[str], section: str) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"Unknown {section} configuration key(s): {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing {section} configuration key(s): {sorted(missing)}")


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a floating-point value")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value
