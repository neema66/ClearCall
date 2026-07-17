"""Strict hybrid-owned configuration for Method 3 fixed-band blending."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


CANONICAL_SAMPLE_RATE = 48_000
CANONICAL_FRAME_SIZE = 960
CANONICAL_NUM_BINS = CANONICAL_FRAME_SIZE // 2 + 1
CANONICAL_NYQUIST_HZ = CANONICAL_SAMPLE_RATE / 2.0
CANONICAL_BIN_WIDTH_HZ = CANONICAL_SAMPLE_RATE / CANONICAL_FRAME_SIZE
_SAFE_VARIANT_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


@dataclasses.dataclass(frozen=True)
class BandBlendVariantConfig:
    """One spectral representation and optional weight-smoothing ablation."""

    id: str
    blend_domain: str
    phase_source: str
    frequency_smoothing_bins: int

    def validate(self) -> None:
        if not isinstance(self.id, str) or _SAFE_VARIANT_ID.fullmatch(self.id) is None:
            raise ValueError(
                "method_3_bands variant id must contain lowercase letters, "
                "digits, and single underscores"
            )
        if self.blend_domain not in {"complex", "magnitude"}:
            raise ValueError("band blend_domain must be 'complex' or 'magnitude'")
        if self.blend_domain == "complex" and self.phase_source != "none":
            raise ValueError("complex band blending requires phase_source='none'")
        if self.blend_domain == "magnitude" and self.phase_source not in {"dsp", "dl"}:
            raise ValueError("magnitude band blending requires phase_source='dsp' or 'dl'")
        width = _require_integer(
            self.frequency_smoothing_bins,
            "method_3_bands.variants.frequency_smoothing_bins",
        )
        if width <= 0 or width > CANONICAL_NUM_BINS or width % 2 == 0:
            raise ValueError(
                "frequency_smoothing_bins must be a positive odd integer no larger than 481"
            )


@dataclasses.dataclass(frozen=True)
class BandListeningSetConfig:
    """Development clips exported for structured human comparison."""

    file_names: tuple[str, ...]

    def validate(self) -> None:
        if not isinstance(self.file_names, tuple) or not self.file_names:
            raise ValueError("band listening file_names must be a non-empty tuple")
        if len(set(self.file_names)) != len(self.file_names):
            raise ValueError("band listening file_names must be unique")
        for file_name in self.file_names:
            if not isinstance(file_name, str) or not file_name:
                raise TypeError("band listening file names must be non-empty strings")
            if (
                Path(file_name).name != file_name
                or "/" in file_name
                or "\\" in file_name
                or Path(file_name).suffix.lower() != ".wav"
            ):
                raise ValueError("band listening file names must be safe WAV basenames")


@dataclasses.dataclass(frozen=True)
class BandEvaluationConfig:
    """Automated and listening settings shared with the selected DSP branch."""

    waveform_alpha: float
    clipping_threshold: float
    listening_set: BandListeningSetConfig

    def validate(self) -> None:
        alpha = _require_finite_float(
            self.waveform_alpha,
            "method_3_bands.evaluation.waveform_alpha",
        )
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("evaluation waveform_alpha must be strictly between 0 and 1")
        threshold = _require_finite_float(
            self.clipping_threshold,
            "method_3_bands.evaluation.clipping_threshold",
        )
        if threshold <= 0.0:
            raise ValueError("evaluation clipping_threshold must be positive")
        if not isinstance(self.listening_set, BandListeningSetConfig):
            raise TypeError("evaluation listening_set must be BandListeningSetConfig")
        self.listening_set.validate()


@dataclasses.dataclass(frozen=True)
class Method3BandConfig:
    """Complete fixed-frequency-band Method 3 Version 2 policy."""

    mode: str
    band_edges_hz: tuple[float, ...]
    dl_weights: tuple[float, ...]
    variants: tuple[BandBlendVariantConfig, ...]
    evaluation: BandEvaluationConfig

    def validate(self) -> None:
        if self.mode != "fixed_frequency_bands":
            raise ValueError("method_3_bands.mode must be 'fixed_frequency_bands'")

        edges = _validate_float_tuple(
            self.band_edges_hz,
            "method_3_bands.band_edges_hz",
        )
        if len(edges) < 2:
            raise ValueError("band_edges_hz must contain at least 0 Hz and Nyquist")
        if edges[0] != 0.0 or edges[-1] != CANONICAL_NYQUIST_HZ:
            raise ValueError("band_edges_hz must begin at 0.0 and end at 24000.0 Hz")
        if any(current <= previous for previous, current in zip(edges, edges[1:])):
            raise ValueError("band_edges_hz must be strictly increasing and unique")

        edge_bins: list[int] = []
        for index, edge in enumerate(edges):
            raw_bin = edge / CANONICAL_BIN_WIDTH_HZ
            nearest_bin = round(raw_bin)
            if not math.isclose(raw_bin, nearest_bin, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"band_edges_hz[{index}] must lie on the 50 Hz FFT-bin grid")
            edge_bins.append(nearest_bin)
        if any(current <= previous for previous, current in zip(edge_bins, edge_bins[1:])):
            raise ValueError("every configured frequency band must contain at least one FFT bin")

        weights = _validate_float_tuple(
            self.dl_weights,
            "method_3_bands.dl_weights",
        )
        if len(weights) != len(edges) - 1:
            raise ValueError("dl_weights must contain exactly one value per frequency band")
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("dl_weights must satisfy 0.0 <= weight <= 1.0")

        if not isinstance(self.variants, tuple) or not self.variants:
            raise ValueError("method_3_bands.variants must be a non-empty tuple")
        if not all(isinstance(variant, BandBlendVariantConfig) for variant in self.variants):
            raise TypeError("every method_3_bands variant must be BandBlendVariantConfig")
        for variant in self.variants:
            variant.validate()
        ids = [variant.id for variant in self.variants]
        if len(set(ids)) != len(ids):
            raise ValueError("method_3_bands variant ids must be unique")
        if not any(variant.blend_domain == "complex" for variant in self.variants):
            raise ValueError("at least one direct complex blend variant is required")
        for phase_source in ("dsp", "dl"):
            if not any(
                variant.blend_domain == "magnitude" and variant.phase_source == phase_source
                for variant in self.variants
            ):
                raise ValueError("magnitude variants must explicitly compare DSP and DL phase")
        if not any(variant.frequency_smoothing_bins == 1 for variant in self.variants):
            raise ValueError("at least one unsmoothed step-weight variant is required")
        if not any(variant.frequency_smoothing_bins > 1 for variant in self.variants):
            raise ValueError("at least one frequency-smoothed variant is required")

        if not isinstance(self.evaluation, BandEvaluationConfig):
            raise TypeError("method_3_bands.evaluation must be BandEvaluationConfig")
        self.evaluation.validate()

    @property
    def num_bands(self) -> int:
        return len(self.dl_weights)

    def variant(self, variant_id: str) -> BandBlendVariantConfig:
        if not isinstance(variant_id, str):
            raise TypeError("variant_id must be a string")
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ValueError(f"unknown Method 3 band variant: {variant_id!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: set[str],
    section: str,
) -> None:
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"Unknown {section} configuration key(s): {rendered}")
    if missing:
        rendered = ", ".join(sorted(str(key) for key in missing))
        raise ValueError(f"Missing {section} configuration key(s): {rendered}")


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML list")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a floating-point value")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_float_tuple(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    return tuple(
        _require_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)
    )


def load_method3_band_config(
    config_path: str | Path = "config/hybrid_method_3_bands.yaml",
) -> Method3BandConfig:
    """Load the exact Method 3 band schema and reject configuration drift."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Method 3 band config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _require_mapping(raw, "configuration root")
    _require_exact_fields(root, {"method_3_bands"}, "root")
    method = _require_mapping(root["method_3_bands"], "method_3_bands")
    _require_exact_fields(
        method,
        {"mode", "band_edges_hz", "dl_weights", "variants", "evaluation"},
        "method_3_bands",
    )

    raw_variants = _require_list(method["variants"], "method_3_bands.variants")
    variants: list[BandBlendVariantConfig] = []
    for index, raw_variant in enumerate(raw_variants):
        variant = _require_mapping(raw_variant, f"method_3_bands.variants[{index}]")
        _require_exact_fields(
            variant,
            {"id", "blend_domain", "phase_source", "frequency_smoothing_bins"},
            f"method_3_bands.variants[{index}]",
        )
        variants.append(
            BandBlendVariantConfig(
                id=_require_string(variant["id"], f"variants[{index}].id"),
                blend_domain=_require_string(
                    variant["blend_domain"],
                    f"variants[{index}].blend_domain",
                ),
                phase_source=_require_string(
                    variant["phase_source"],
                    f"variants[{index}].phase_source",
                ),
                frequency_smoothing_bins=_require_integer(
                    variant["frequency_smoothing_bins"],
                    f"variants[{index}].frequency_smoothing_bins",
                ),
            )
        )

    evaluation = _require_mapping(method["evaluation"], "method_3_bands.evaluation")
    _require_exact_fields(
        evaluation,
        {"waveform_alpha", "clipping_threshold", "listening_set"},
        "method_3_bands.evaluation",
    )
    listening = _require_mapping(
        evaluation["listening_set"],
        "method_3_bands.evaluation.listening_set",
    )
    _require_exact_fields(
        listening,
        {"file_names"},
        "method_3_bands.evaluation.listening_set",
    )

    config = Method3BandConfig(
        mode=_require_string(method["mode"], "method_3_bands.mode"),
        band_edges_hz=tuple(_require_list(method["band_edges_hz"], "method_3_bands.band_edges_hz")),
        dl_weights=tuple(_require_list(method["dl_weights"], "method_3_bands.dl_weights")),
        variants=tuple(variants),
        evaluation=BandEvaluationConfig(
            waveform_alpha=_require_finite_float(
                evaluation["waveform_alpha"],
                "method_3_bands.evaluation.waveform_alpha",
            ),
            clipping_threshold=_require_finite_float(
                evaluation["clipping_threshold"],
                "method_3_bands.evaluation.clipping_threshold",
            ),
            listening_set=BandListeningSetConfig(
                file_names=tuple(
                    _require_list(
                        listening["file_names"],
                        "method_3_bands.evaluation.listening_set.file_names",
                    )
                )
            ),
        ),
    )
    config.validate()
    return config
