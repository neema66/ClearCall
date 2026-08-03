"""Bounded DSP-guided output-level controller for Hybrid Method 2."""

from __future__ import annotations

import dataclasses

import numpy as np

from senhance.pipeline.hybrid.method2.config import Method2Config


@dataclasses.dataclass(frozen=True)
class Method2Statistics:
    """Level, safety, and output diagnostics for one complete clip."""

    sample_count: int
    dsp_rms: float
    dl_rms: float
    raw_ratio: float
    level_gain: float
    peak_safe_gain: float
    applied_gain: float
    silence_bypassed: bool
    minimum_gain_limited: bool
    maximum_gain_limited: bool
    peak_limited: bool
    dl_peak_abs: float
    output_peak_abs: float
    clipping_threshold: float
    clipped_sample_count: int
    clipped_sample_fraction: float


@dataclasses.dataclass(frozen=True)
class Method2Result:
    """Exact-length Method 2 waveform and its diagnostics."""

    audio: np.ndarray
    statistics: Method2Statistics


class DSPGuidedLevelController:
    """Calibrate a DL waveform using only the Improved DSP output RMS."""

    def __init__(self, config: Method2Config) -> None:
        if not isinstance(config, Method2Config):
            raise TypeError("config must be Method2Config")
        config.validate()
        self.config = config

    def apply(self, dsp_audio: np.ndarray, dl_audio: np.ndarray) -> Method2Result:
        """Return bounded level-calibrated DL audio without waveform mixing."""

        dsp = _audio_array(dsp_audio, "Improved DSP")
        dl = _audio_array(dl_audio, "DeepFilterNet")
        if dsp.shape != dl.shape:
            raise ValueError(
                "Hybrid Method 2 branch outputs must have equal shapes: "
                f"DSP={dsp.shape}, DL={dl.shape}"
            )

        sample_count = int(dl.size)
        dsp_power = _mean_square(dsp)
        dl_power = _mean_square(dl)
        epsilon = self.config.level.epsilon
        dsp_rms = float(np.sqrt(dsp_power + epsilon))
        dl_rms = float(np.sqrt(dl_power + epsilon))
        raw_ratio = dsp_rms / dl_rms

        silence_threshold = self.config.level.silence_rms_threshold
        silence_bypassed = bool(
            np.sqrt(dsp_power) <= silence_threshold
            and np.sqrt(dl_power) <= silence_threshold
        )
        minimum_gain = self.config.level.minimum_gain
        maximum_gain = self.config.level.maximum_gain
        if silence_bypassed:
            level_gain = 1.0
        else:
            level_gain = float(np.clip(raw_ratio, minimum_gain, maximum_gain))

        dl_peak = float(np.max(np.abs(dl.astype(np.float64)))) if sample_count else 0.0
        peak_safe_gain = (
            self.config.safety.peak_limit / dl_peak if dl_peak > 0.0 else float("inf")
        )
        applied_gain = min(level_gain, peak_safe_gain)

        output64 = dl.astype(np.float64) * applied_gain
        if not np.all(np.isfinite(output64)):
            raise ValueError("Hybrid Method 2 produced non-finite output")
        output = np.ascontiguousarray(output64, dtype=np.float32)
        if not np.all(np.isfinite(output)):
            raise ValueError("Hybrid Method 2 output exceeds the float32 finite range")

        output_absolute = np.abs(output.astype(np.float64))
        output_peak = float(np.max(output_absolute)) if sample_count else 0.0
        clipping_threshold = self.config.safety.clipping_threshold
        clipped_count = int(np.count_nonzero(output_absolute >= clipping_threshold))

        statistics = Method2Statistics(
            sample_count=sample_count,
            dsp_rms=dsp_rms,
            dl_rms=dl_rms,
            raw_ratio=raw_ratio,
            level_gain=level_gain,
            peak_safe_gain=peak_safe_gain,
            applied_gain=applied_gain,
            silence_bypassed=silence_bypassed,
            minimum_gain_limited=(not silence_bypassed and raw_ratio <= minimum_gain),
            maximum_gain_limited=(not silence_bypassed and raw_ratio >= maximum_gain),
            peak_limited=peak_safe_gain < level_gain,
            dl_peak_abs=dl_peak,
            output_peak_abs=output_peak,
            clipping_threshold=clipping_threshold,
            clipped_sample_count=clipped_count,
            clipped_sample_fraction=(clipped_count / sample_count if sample_count else 0.0),
        )
        return Method2Result(audio=output, statistics=statistics)


def _audio_array(value: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} output must be a numpy.ndarray")
    if value.ndim != 1:
        raise ValueError(f"{label} output must be one-dimensional mono audio")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{label} output must have a real floating dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} output must contain only finite samples")
    with np.errstate(over="ignore", invalid="ignore"):
        audio = np.ascontiguousarray(value, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} output exceeds the float32 finite range")
    return audio


def _mean_square(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    values = audio.astype(np.float64)
    return float(np.mean(values * values))
