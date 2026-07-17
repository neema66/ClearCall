"""Strict, Method-3-owned configuration for fixed waveform experiments."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


_SAFE_BACKEND_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


@dataclasses.dataclass(frozen=True)
class Method3DSPBackendConfig:
    """One replaceable DSP branch and its reviewed residual DL delay."""

    id: str
    dl_minus_dsp_delay_samples: int

    def validate(self) -> None:
        if not isinstance(self.id, str) or _SAFE_BACKEND_ID.fullmatch(self.id) is None:
            raise ValueError(
                "method_3 DSP backend id must contain lowercase letters, digits, "
                "and single underscores"
            )
        _require_integer(
            self.dl_minus_dsp_delay_samples,
            f"method_3.dsp.backends.{self.id}.dl_minus_dsp_delay_samples",
        )


@dataclasses.dataclass(frozen=True)
class Method3DSPConfig:
    """Concrete DSP selection owned by orchestration, not blend mathematics."""

    selected_backend: str
    backends: tuple[Method3DSPBackendConfig, ...]

    def validate(self) -> None:
        if (
            not isinstance(self.selected_backend, str)
            or _SAFE_BACKEND_ID.fullmatch(self.selected_backend) is None
        ):
            raise ValueError("method_3.dsp.selected_backend must be a safe backend id")
        if not isinstance(self.backends, tuple) or not self.backends:
            raise ValueError("method_3.dsp.backends must contain at least one backend")
        if not all(isinstance(value, Method3DSPBackendConfig) for value in self.backends):
            raise TypeError("every method_3.dsp.backends value must be Method3DSPBackendConfig")
        for backend in self.backends:
            backend.validate()
        ids = [backend.id for backend in self.backends]
        if len(ids) != len(set(ids)):
            raise ValueError("method_3 DSP backend ids must be unique")
        if self.selected_backend not in ids:
            raise ValueError("selected Method 3 DSP backend must exist in dsp.backends")

    def backend(self, backend_id: str | None = None) -> Method3DSPBackendConfig:
        """Return the selected or explicitly requested reviewed backend."""

        requested = self.selected_backend if backend_id is None else backend_id
        if not isinstance(requested, str):
            raise TypeError("backend_id must be a string")
        for backend in self.backends:
            if backend.id == requested:
                return backend
        raise ValueError(f"unknown Method 3 DSP backend: {requested!r}")


@dataclasses.dataclass(frozen=True)
class Method3ListeningSetConfig:
    """Development clips and alpha values exported for human review."""

    file_names: tuple[str, ...]
    alpha_values: tuple[float, ...]

    def validate(self, alpha_sweep: tuple[float, ...]) -> None:
        if not isinstance(self.file_names, tuple) or not self.file_names:
            raise ValueError("method_3.listening_set.file_names must be a non-empty tuple")
        if len(set(self.file_names)) != len(self.file_names):
            raise ValueError("method_3.listening_set.file_names must be unique")
        for file_name in self.file_names:
            if not isinstance(file_name, str) or not file_name:
                raise TypeError("method_3 listening file names must be non-empty strings")
            if (
                Path(file_name).name != file_name
                or "/" in file_name
                or "\\" in file_name
                or Path(file_name).suffix.lower() != ".wav"
            ):
                raise ValueError("method_3 listening file names must be safe WAV basenames")

        alpha_values = _validate_alpha_sequence(
            self.alpha_values,
            "method_3.listening_set.alpha_values",
        )
        if len(set(alpha_values)) != len(alpha_values):
            raise ValueError("method_3 listening alpha values must be unique")
        missing = [alpha for alpha in alpha_values if alpha not in alpha_sweep]
        if missing:
            raise ValueError(
                "method_3 listening alpha values must be present in alpha_sweep: " f"{missing}"
            )


@dataclasses.dataclass(frozen=True)
class Method3WaveformConfig:
    """Complete configuration for independent Method 3 Version 1."""

    mode: str
    dsp: Method3DSPConfig
    alpha_sweep: tuple[float, ...]
    clipping_threshold: float
    listening_set: Method3ListeningSetConfig

    def validate(self) -> None:
        if not isinstance(self.mode, str) or self.mode != "fixed_waveform":
            raise ValueError("method_3.mode must be 'fixed_waveform'")
        if not isinstance(self.dsp, Method3DSPConfig):
            raise TypeError("method_3.dsp must be Method3DSPConfig")
        self.dsp.validate()
        alphas = _validate_alpha_sequence(self.alpha_sweep, "method_3.alpha_sweep")
        if len(alphas) < 3:
            raise ValueError("method_3.alpha_sweep must contain at least three values")
        if any(current <= previous for previous, current in zip(alphas, alphas[1:])):
            raise ValueError("method_3.alpha_sweep must be strictly increasing and unique")
        for required in (0.0, 0.5, 1.0):
            if required not in alphas:
                raise ValueError(
                    "method_3.alpha_sweep must include endpoint and midpoint values "
                    "0.0, 0.5, and 1.0"
                )
        threshold = _require_finite_float(
            self.clipping_threshold,
            "method_3.clipping_threshold",
        )
        if threshold <= 0.0:
            raise ValueError("method_3.clipping_threshold must be positive")
        if not isinstance(self.listening_set, Method3ListeningSetConfig):
            raise TypeError("method_3.listening_set must be Method3ListeningSetConfig")
        self.listening_set.validate(alphas)


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


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{name} must be a floating-point value")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _validate_alpha_sequence(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    validated = tuple(
        _require_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)
    )
    if any(value < 0.0 or value > 1.0 for value in validated):
        raise ValueError(f"{name} values must satisfy 0.0 <= alpha <= 1.0")
    return validated


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a YAML list")
    return value


def load_method3_config(
    config_path: str | Path = "config/hybrid_method_3.yaml",
) -> Method3WaveformConfig:
    """Load Method 3 Version 1 settings and reject schema drift."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Method 3 config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _require_mapping(raw, "configuration root")
    _require_exact_fields(root, {"method_3"}, "root")
    method = _require_mapping(root["method_3"], "method_3")
    _require_exact_fields(
        method,
        {"mode", "dsp", "alpha_sweep", "clipping_threshold", "listening_set"},
        "method_3",
    )
    dsp = _require_mapping(method["dsp"], "method_3.dsp")
    _require_exact_fields(
        dsp,
        {"selected_backend", "backends"},
        "method_3.dsp",
    )
    raw_backends = _require_mapping(dsp["backends"], "method_3.dsp.backends")
    backends: list[Method3DSPBackendConfig] = []
    for backend_id, raw_backend in raw_backends.items():
        backend_name = _require_string(backend_id, "method_3.dsp backend id")
        backend = _require_mapping(
            raw_backend,
            f"method_3.dsp.backends.{backend_name}",
        )
        _require_exact_fields(
            backend,
            {"dl_minus_dsp_delay_samples"},
            f"method_3.dsp.backends.{backend_name}",
        )
        backends.append(
            Method3DSPBackendConfig(
                id=backend_name,
                dl_minus_dsp_delay_samples=_require_integer(
                    backend["dl_minus_dsp_delay_samples"],
                    f"method_3.dsp.backends.{backend_name}.dl_minus_dsp_delay_samples",
                ),
            )
        )
    listening = _require_mapping(method["listening_set"], "method_3.listening_set")
    _require_exact_fields(
        listening,
        {"file_names", "alpha_values"},
        "method_3.listening_set",
    )

    file_names = _require_list(
        listening["file_names"],
        "method_3.listening_set.file_names",
    )
    alpha_values = _require_list(
        listening["alpha_values"],
        "method_3.listening_set.alpha_values",
    )
    alpha_sweep = _require_list(method["alpha_sweep"], "method_3.alpha_sweep")
    config = Method3WaveformConfig(
        mode=method["mode"],
        dsp=Method3DSPConfig(
            selected_backend=_require_string(
                dsp["selected_backend"],
                "method_3.dsp.selected_backend",
            ),
            backends=tuple(backends),
        ),
        alpha_sweep=tuple(alpha_sweep),
        clipping_threshold=_require_finite_float(
            method["clipping_threshold"],
            "method_3.clipping_threshold",
        ),
        listening_set=Method3ListeningSetConfig(
            file_names=tuple(file_names),
            alpha_values=tuple(alpha_values),
        ),
    )
    config.validate()
    return config
