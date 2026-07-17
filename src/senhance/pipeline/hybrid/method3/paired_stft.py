"""Hybrid-owned synchronized paired STFT and single-path WOLA synthesis."""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
from scipy.signal import get_window  # type: ignore[import-untyped]

from senhance.pipeline.hybrid.method3.alignment import FixedDelayAligner
from senhance.pipeline.hybrid.method3.config import HybridConfig


@dataclasses.dataclass(frozen=True)
class PairedSpectrumFrame:
    """Synchronized reference/candidate spectra for a hybrid processor."""

    index: int
    reference_spectrum: np.ndarray
    candidate_spectrum: np.ndarray


class SpectrumProcessor(Protocol):
    """Neutral callback contract shared by future independent hybrid methods."""

    def __call__(self, frame: PairedSpectrumFrame) -> np.ndarray:
        """Return one finite complex spectrum for the shared synthesis path."""


class PairedSTFTCore:
    """Align, frame, analyze, and reconstruct paired whole-array audio.

    The core owns no denoising or blending policy. A callback receives each
    paired spectrum and independently selects or creates the one spectrum sent
    through the sole inverse-STFT/weighted-overlap-add path.
    """

    def __init__(self, config: HybridConfig) -> None:
        if not isinstance(config, HybridConfig):
            raise TypeError("config must be HybridConfig")
        config.validate()
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_size = config.stft.frame_size
        self.hop_size = config.stft.hop_size
        self.num_frequency_bins = config.stft.num_frequency_bins
        self.framing_delay_samples = self.hop_size
        self.residual_alignment_delay_samples = config.alignment.delay_samples

        self._analysis_window = get_window(
            config.stft.window,
            self.frame_size,
            fftbins=True,
        ).astype(np.float64)
        squared = self._analysis_window * self._analysis_window
        denominator = squared + np.roll(squared, self.hop_size)
        if np.any(denominator <= np.finfo(np.float64).eps):
            raise ValueError("hybrid STFT window does not support stable 50% WOLA")
        self._synthesis_window = self._analysis_window / denominator
        self._aligner = FixedDelayAligner(config.alignment.delay_samples)

        self._previous_reference = np.zeros(self.hop_size, dtype=np.float64)
        self._previous_candidate = np.zeros(self.hop_size, dtype=np.float64)
        self._synthesis_overlap = np.zeros(self.hop_size, dtype=np.float64)
        self._frame_index = 0

    def reset(self) -> None:
        """Clear paired input histories, synthesis overlap, and frame index."""

        self._previous_reference.fill(0.0)
        self._previous_candidate.fill(0.0)
        self._synthesis_overlap.fill(0.0)
        self._frame_index = 0

    def process_aligned_hop(
        self,
        reference_hop: np.ndarray,
        candidate_hop: np.ndarray,
        processor: SpectrumProcessor,
    ) -> np.ndarray:
        """Process one already-aligned hop and return one delayed output hop."""

        if not callable(processor):
            raise TypeError("processor must be callable")
        current_reference = self._validate_hop(reference_hop, "reference")
        current_candidate = self._validate_hop(candidate_hop, "candidate")

        reference_frame = np.concatenate((self._previous_reference, current_reference))
        candidate_frame = np.concatenate((self._previous_candidate, current_candidate))
        self._previous_reference = current_reference.copy()
        self._previous_candidate = current_candidate.copy()

        reference_spectrum = np.fft.rfft(reference_frame * self._analysis_window)
        candidate_spectrum = np.fft.rfft(candidate_frame * self._analysis_window)
        reference_spectrum.setflags(write=False)
        candidate_spectrum.setflags(write=False)
        frame = PairedSpectrumFrame(
            index=self._frame_index,
            reference_spectrum=reference_spectrum,
            candidate_spectrum=candidate_spectrum,
        )
        output_spectrum = self._validate_spectrum(processor(frame))
        self._frame_index += 1

        time_frame = np.fft.irfft(output_spectrum, n=self.frame_size)
        time_frame *= self._synthesis_window
        output = time_frame[: self.hop_size] + self._synthesis_overlap
        self._synthesis_overlap = time_frame[self.hop_size :].copy()
        if not np.all(np.isfinite(output)):
            raise ValueError("hybrid synthesis produced non-finite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            output_float32 = np.asarray(output, dtype=np.float32)
        if not np.all(np.isfinite(output_float32)):
            raise ValueError("hybrid synthesis output exceeds the float32 finite range")
        return np.ascontiguousarray(output_float32)

    def process_array(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        processor: SpectrumProcessor,
        *,
        sample_rate: int,
    ) -> np.ndarray:
        """Process one pair with padding, one flush, delay removal, and trim."""

        self._validate_sample_rate(sample_rate)
        if not callable(processor):
            raise TypeError("processor must be callable")
        aligned = self._aligner.align(reference, candidate)
        self.reset()
        processor_reset = getattr(processor, "reset", None)
        if processor_reset is not None:
            if not callable(processor_reset):
                raise TypeError("processor.reset must be callable")
            processor_reset()

        sample_count = aligned.reference.size
        if sample_count == 0:
            return np.zeros(0, dtype=np.float32)

        padded_length = ((sample_count + self.hop_size - 1) // self.hop_size) * self.hop_size
        padded_reference = np.zeros(padded_length, dtype=np.float32)
        padded_candidate = np.zeros(padded_length, dtype=np.float32)
        padded_reference[:sample_count] = aligned.reference
        padded_candidate[:sample_count] = aligned.candidate

        chunks = []
        for start in range(0, padded_length, self.hop_size):
            chunks.append(
                self.process_aligned_hop(
                    padded_reference[start : start + self.hop_size],
                    padded_candidate[start : start + self.hop_size],
                    processor,
                )
            )
        zero_hop = np.zeros(self.hop_size, dtype=np.float32)
        chunks.append(self.process_aligned_hop(zero_hop, zero_hop, processor))

        delayed = np.concatenate(chunks)
        start = self.framing_delay_samples
        output = delayed[start : start + sample_count]
        if output.shape != (sample_count,):
            raise AssertionError(
                "hybrid framing failed exact-length reconstruction: "
                f"expected={sample_count}, actual={output.size}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("hybrid array output contains non-finite samples")
        return np.ascontiguousarray(output, dtype=np.float32)

    def _validate_hop(self, audio: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"{label} hop must be a numpy.ndarray")
        if audio.ndim != 1 or audio.shape != (self.hop_size,):
            raise ValueError(f"{label} hop must have shape ({self.hop_size},), got {audio.shape}")
        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError(f"{label} hop must have a real floating dtype")
        if not np.all(np.isfinite(audio)):
            raise ValueError(f"{label} hop must contain only finite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            samples = np.array(audio, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"{label} hop is outside the float64 finite range")
        return samples

    def _validate_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("processor output must be a numpy.ndarray")
        if spectrum.ndim != 1 or spectrum.shape != (self.num_frequency_bins,):
            raise ValueError(
                "processor output must have spectrum shape "
                f"({self.num_frequency_bins},), got {spectrum.shape}"
            )
        if not np.issubdtype(spectrum.dtype, np.complexfloating):
            raise TypeError("processor output must have a complex floating dtype")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("processor output spectrum must contain only finite values")
        return np.asarray(spectrum, dtype=np.complex128)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate,
            (int, np.integer),
        ):
            raise TypeError("sample_rate must be an integer")
        if int(sample_rate) != self.sample_rate:
            raise ValueError(f"hybrid sample rate must be {self.sample_rate} Hz, got {sample_rate}")


def select_reference_spectrum(frame: PairedSpectrumFrame) -> np.ndarray:
    """Identity processor selecting the aligned reference spectrum."""

    return frame.reference_spectrum


def select_candidate_spectrum(frame: PairedSpectrumFrame) -> np.ndarray:
    """Identity processor selecting the aligned candidate spectrum."""

    return frame.candidate_spectrum
