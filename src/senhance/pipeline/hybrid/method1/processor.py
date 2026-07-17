"""Independent offline Hybrid Method 1 keep-map safety layer."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from senhance.pipeline.hybrid.method1.config import Method1Config, Method1VariantConfig
from senhance.pipeline.hybrid.method1.keep_map import estimate_raw_keep_map
from senhance.pipeline.hybrid.method1.paired_stft import (
    Method1PairedSTFTCore,
    NoisyDLSpectrumFrame,
)
from senhance.pipeline.hybrid.method1.safety_controller import SafetyGainController


@dataclasses.dataclass(frozen=True)
class Method1Statistics:
    """Waveform and gain-map diagnostics for one complete Method 1 call."""

    variant_id: str
    reconstruction_phase: str
    frame_count: int
    gain_bin_count: int
    raw_gain_min: float
    raw_gain_max: float
    raw_gain_mean: float
    final_gain_min: float
    final_gain_max: float
    final_gain_mean: float
    low_energy_neutral_bin_count: int
    clipped_above_one_bin_count: int
    raw_zero_gain_bin_count: int
    temporal_changed_bin_count: int
    frequency_changed_bin_count: int
    floor_raised_bin_count: int
    drop_limited_bin_count: int
    rise_limited_bin_count: int
    raw_temporal_variation_mean: float
    final_temporal_variation_mean: float
    raw_frequency_roughness_mean: float
    final_frequency_roughness_mean: float
    final_gain_trace_sha256: str
    sample_count: int
    peak_abs: float
    clipping_threshold: float
    clipped_sample_count: int
    clipped_sample_fraction: float


@dataclasses.dataclass(frozen=True)
class Method1Result:
    """One exact-length Method 1 waveform and its diagnostics."""

    audio: np.ndarray
    statistics: Method1Statistics
    dl_minus_noisy_delay_samples: int


class _Method1SpectrumSafetyProcessor:
    def __init__(self, config: Method1Config, variant: Method1VariantConfig) -> None:
        self.config = config
        self.variant = variant
        self._controller = SafetyGainController(config.safety, variant)
        self.reset()

    def reset(self) -> None:
        self._controller.reset()
        self._frame_count = 0
        self._gain_bin_count = 0
        self._raw_min = np.inf
        self._raw_max = -np.inf
        self._raw_sum = 0.0
        self._final_min = np.inf
        self._final_max = -np.inf
        self._final_sum = 0.0
        self._low_energy_neutral = 0
        self._clipped_above_one = 0
        self._raw_zero = 0
        self._temporal_changed = 0
        self._frequency_changed = 0
        self._floor_raised = 0
        self._drop_limited = 0
        self._rise_limited = 0
        self._raw_temporal_variation = 0.0
        self._final_temporal_variation = 0.0
        self._temporal_transition_count = 0
        self._raw_frequency_roughness = 0.0
        self._final_frequency_roughness = 0.0
        self._previous_raw: np.ndarray | None = None
        self._previous_final: np.ndarray | None = None
        self._gain_hash = hashlib.sha256()

    def __call__(self, frame: NoisyDLSpectrumFrame) -> np.ndarray:
        raw = estimate_raw_keep_map(
            frame.noisy_spectrum,
            frame.dl_spectrum,
            self.config.keep_map,
        )
        stages = self._controller.apply(raw.gain)
        final_gain = stages.final_gain
        output = reconstruct_with_gain(
            frame.noisy_spectrum,
            frame.dl_spectrum,
            final_gain,
            phase_source=self.variant.reconstruction_phase,
            low_energy_threshold=self.config.keep_map.low_energy_threshold,
        )

        self._frame_count += 1
        self._gain_bin_count += raw.bin_count
        self._raw_min = min(self._raw_min, float(np.min(raw.gain)))
        self._raw_max = max(self._raw_max, float(np.max(raw.gain)))
        self._raw_sum += float(np.sum(raw.gain))
        self._final_min = min(self._final_min, float(np.min(final_gain)))
        self._final_max = max(self._final_max, float(np.max(final_gain)))
        self._final_sum += float(np.sum(final_gain))
        self._low_energy_neutral += raw.low_energy_neutral_bin_count
        self._clipped_above_one += raw.clipped_above_one_bin_count
        self._raw_zero += raw.zero_gain_bin_count
        self._temporal_changed += stages.temporal_changed_bin_count
        self._frequency_changed += stages.frequency_changed_bin_count
        self._floor_raised += stages.floor_raised_bin_count
        self._drop_limited += stages.drop_limited_bin_count
        self._rise_limited += stages.rise_limited_bin_count
        if self._previous_raw is not None and self._previous_final is not None:
            self._raw_temporal_variation += float(np.mean(np.abs(raw.gain - self._previous_raw)))
            self._final_temporal_variation += float(
                np.mean(np.abs(final_gain - self._previous_final))
            )
            self._temporal_transition_count += 1
        self._previous_raw = raw.gain.copy()
        self._previous_final = final_gain.copy()
        if raw.gain.size > 1:
            self._raw_frequency_roughness += float(np.mean(np.abs(np.diff(raw.gain))))
            self._final_frequency_roughness += float(np.mean(np.abs(np.diff(final_gain))))
        self._gain_hash.update(np.ascontiguousarray(final_gain, dtype="<f8").tobytes())
        return output

    def statistics(
        self,
        audio: np.ndarray,
        *,
        clipping_threshold: float,
    ) -> Method1Statistics:
        sample_count = int(audio.size)
        absolute = np.abs(audio.astype(np.float64))
        clipped = int(np.count_nonzero(absolute >= clipping_threshold))
        gain_count = self._gain_bin_count
        frame_count = self._frame_count
        transitions = self._temporal_transition_count
        return Method1Statistics(
            variant_id=self.variant.id,
            reconstruction_phase=self.variant.reconstruction_phase,
            frame_count=frame_count,
            gain_bin_count=gain_count,
            raw_gain_min=float(self._raw_min) if gain_count else 1.0,
            raw_gain_max=float(self._raw_max) if gain_count else 1.0,
            raw_gain_mean=self._raw_sum / gain_count if gain_count else 1.0,
            final_gain_min=float(self._final_min) if gain_count else 1.0,
            final_gain_max=float(self._final_max) if gain_count else 1.0,
            final_gain_mean=self._final_sum / gain_count if gain_count else 1.0,
            low_energy_neutral_bin_count=self._low_energy_neutral,
            clipped_above_one_bin_count=self._clipped_above_one,
            raw_zero_gain_bin_count=self._raw_zero,
            temporal_changed_bin_count=self._temporal_changed,
            frequency_changed_bin_count=self._frequency_changed,
            floor_raised_bin_count=self._floor_raised,
            drop_limited_bin_count=self._drop_limited,
            rise_limited_bin_count=self._rise_limited,
            raw_temporal_variation_mean=(
                self._raw_temporal_variation / transitions if transitions else 0.0
            ),
            final_temporal_variation_mean=(
                self._final_temporal_variation / transitions if transitions else 0.0
            ),
            raw_frequency_roughness_mean=(
                self._raw_frequency_roughness / frame_count if frame_count else 0.0
            ),
            final_frequency_roughness_mean=(
                self._final_frequency_roughness / frame_count if frame_count else 0.0
            ),
            final_gain_trace_sha256=self._gain_hash.hexdigest(),
            sample_count=sample_count,
            peak_abs=float(np.max(absolute)) if sample_count else 0.0,
            clipping_threshold=clipping_threshold,
            clipped_sample_count=clipped,
            clipped_sample_fraction=clipped / sample_count if sample_count else 0.0,
        )


class Method1SafetyLayer:
    """Whole-array Method 1 API accepting only noisy and cached-DL arrays."""

    def __init__(self, config: Method1Config, *, variant_id: str) -> None:
        if not isinstance(config, Method1Config):
            raise TypeError("config must be Method1Config")
        config.validate()
        self.config = config
        self.variant = config.variant(variant_id)
        self._processor = _Method1SpectrumSafetyProcessor(config, self.variant)
        self._core = Method1PairedSTFTCore(config)

    def reset(self) -> None:
        self._core.reset()
        self._processor.reset()

    def process_array(
        self,
        noisy: np.ndarray,
        dl: np.ndarray,
        *,
        sample_rate: int,
    ) -> Method1Result:
        """Apply Method 1 without enhancer calls, clipping, or normalization."""

        output = self._core.process_array(
            noisy,
            dl,
            self._processor,
            sample_rate=sample_rate,
        )
        output = np.ascontiguousarray(output, dtype=np.float32)
        if not np.all(np.isfinite(output)):
            raise ValueError("Method 1 output must contain finite samples")
        statistics = self._processor.statistics(
            output,
            clipping_threshold=self.config.evaluation.clipping_threshold,
        )
        return Method1Result(
            audio=output,
            statistics=statistics,
            dl_minus_noisy_delay_samples=(self.config.alignment.dl_minus_noisy_delay_samples),
        )


def reconstruct_with_gain(
    noisy_spectrum: np.ndarray,
    dl_spectrum: np.ndarray,
    gain: np.ndarray,
    *,
    phase_source: str,
    low_energy_threshold: float,
) -> np.ndarray:
    """Apply safe noisy magnitude with explicit noisy/DL phase selection."""

    noisy = _complex_spectrum(noisy_spectrum, "noisy")
    dl = _complex_spectrum(dl_spectrum, "DL")
    if noisy.shape != dl.shape:
        raise ValueError("Method 1 reconstruction spectra must have equal shapes")
    if not isinstance(gain, np.ndarray) or gain.ndim != 1:
        raise TypeError("Method 1 reconstruction gain must be a one-dimensional array")
    if gain.shape != noisy.shape or not np.issubdtype(gain.dtype, np.floating):
        raise ValueError("Method 1 reconstruction gain shape/dtype is invalid")
    if not np.all(np.isfinite(gain)) or np.any(gain < 0.0) or np.any(gain > 1.0):
        raise ValueError("Method 1 reconstruction gains must be finite and bounded")
    if phase_source not in {"noisy", "dl"}:
        raise ValueError("phase_source must be 'noisy' or 'dl'")
    if (
        isinstance(low_energy_threshold, bool)
        or not isinstance(low_energy_threshold, float)
        or not np.isfinite(low_energy_threshold)
        or low_energy_threshold < 0.0
    ):
        raise ValueError("low_energy_threshold must be a non-negative float")

    noisy_magnitude = np.abs(noisy)
    dl_magnitude = np.abs(dl)
    noisy_phase = _unit_phase(noisy, noisy_magnitude, low_energy_threshold)
    if phase_source == "noisy":
        selected_phase = noisy_phase
    else:
        selected_phase = noisy_phase.copy()
        valid_dl_phase = dl_magnitude > low_energy_threshold
        selected_phase[valid_dl_phase] = dl[valid_dl_phase] / dl_magnitude[valid_dl_phase]
    output = gain.astype(np.float64) * noisy_magnitude * selected_phase
    if not np.all(np.isfinite(output)):
        raise ValueError("Method 1 reconstruction produced non-finite values")
    return np.ascontiguousarray(output, dtype=np.complex128)


def _unit_phase(
    spectrum: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
) -> np.ndarray:
    phase = np.ones(spectrum.shape, dtype=np.complex128)
    valid = magnitude > threshold
    phase[valid] = spectrum[valid] / magnitude[valid]
    return phase


def _complex_spectrum(value: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Method 1 {label} spectrum must be a numpy.ndarray")
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.complexfloating):
        raise ValueError(f"Method 1 {label} spectrum must be one-dimensional complex")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Method 1 {label} spectrum must contain finite values")
    return np.asarray(value, dtype=np.complex128)
