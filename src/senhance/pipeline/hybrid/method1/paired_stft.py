"""Method-1-owned synchronized noisy/DL STFT and exact-length WOLA."""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
from scipy.signal import get_window  # type: ignore[import-untyped]

from senhance.pipeline.hybrid.method1.alignment import FixedNoisyDLAligner
from senhance.pipeline.hybrid.method1.config import Method1Config


@dataclasses.dataclass(frozen=True)
class NoisyDLSpectrumFrame:
    """One synchronized read-only noisy/DL spectrum pair."""

    index: int
    noisy_spectrum: np.ndarray
    dl_spectrum: np.ndarray


class Method1SpectrumProcessor(Protocol):
    def __call__(self, frame: NoisyDLSpectrumFrame) -> np.ndarray:
        """Return the one complex spectrum sent to Method 1 synthesis."""


class Method1PairedSTFTCore:
    """Independent paired analysis and single-path Method 1 synthesis."""

    def __init__(self, config: Method1Config) -> None:
        if not isinstance(config, Method1Config):
            raise TypeError("config must be Method1Config")
        config.validate()
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_size = config.stft.frame_size
        self.hop_size = config.stft.hop_size
        self.num_frequency_bins = config.stft.num_frequency_bins
        self.framing_delay_samples = self.hop_size
        self.residual_alignment_delay_samples = config.alignment.dl_minus_noisy_delay_samples
        self._analysis_window = get_window(
            config.stft.window,
            self.frame_size,
            fftbins=True,
        ).astype(np.float64)
        squared = self._analysis_window * self._analysis_window
        denominator = squared + np.roll(squared, self.hop_size)
        if np.any(denominator <= np.finfo(np.float64).eps):
            raise ValueError("Method 1 window does not support stable 50% WOLA")
        self._synthesis_window = self._analysis_window / denominator
        self._aligner = FixedNoisyDLAligner(config.alignment.dl_minus_noisy_delay_samples)
        self._previous_noisy = np.zeros(self.hop_size, dtype=np.float64)
        self._previous_dl = np.zeros(self.hop_size, dtype=np.float64)
        self._synthesis_overlap = np.zeros(self.hop_size, dtype=np.float64)
        self._frame_index = 0

    def reset(self) -> None:
        self._previous_noisy.fill(0.0)
        self._previous_dl.fill(0.0)
        self._synthesis_overlap.fill(0.0)
        self._frame_index = 0

    def process_aligned_hop(
        self,
        noisy_hop: np.ndarray,
        dl_hop: np.ndarray,
        processor: Method1SpectrumProcessor,
    ) -> np.ndarray:
        if not callable(processor):
            raise TypeError("Method 1 spectrum processor must be callable")
        current_noisy = self._hop(noisy_hop, "noisy")
        current_dl = self._hop(dl_hop, "DL")
        noisy_frame = np.concatenate((self._previous_noisy, current_noisy))
        dl_frame = np.concatenate((self._previous_dl, current_dl))
        self._previous_noisy = current_noisy.copy()
        self._previous_dl = current_dl.copy()

        noisy_spectrum = np.fft.rfft(noisy_frame * self._analysis_window)
        dl_spectrum = np.fft.rfft(dl_frame * self._analysis_window)
        noisy_spectrum.setflags(write=False)
        dl_spectrum.setflags(write=False)
        spectrum_frame = NoisyDLSpectrumFrame(
            index=self._frame_index,
            noisy_spectrum=noisy_spectrum,
            dl_spectrum=dl_spectrum,
        )
        output_spectrum = self._spectrum(processor(spectrum_frame))
        self._frame_index += 1

        time_frame = np.fft.irfft(output_spectrum, n=self.frame_size)
        time_frame *= self._synthesis_window
        output = time_frame[: self.hop_size] + self._synthesis_overlap
        self._synthesis_overlap = time_frame[self.hop_size :].copy()
        if not np.all(np.isfinite(output)):
            raise ValueError("Method 1 synthesis produced non-finite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(output, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("Method 1 synthesis exceeds the float32 finite range")
        return np.ascontiguousarray(result)

    def process_array(
        self,
        noisy: np.ndarray,
        dl: np.ndarray,
        processor: Method1SpectrumProcessor,
        *,
        sample_rate: int,
    ) -> np.ndarray:
        self._sample_rate(sample_rate)
        if not callable(processor):
            raise TypeError("Method 1 spectrum processor must be callable")
        aligned = self._aligner.align(noisy, dl)
        self.reset()
        processor_reset = getattr(processor, "reset", None)
        if processor_reset is not None:
            if not callable(processor_reset):
                raise TypeError("Method 1 processor.reset must be callable")
            processor_reset()
        sample_count = aligned.noisy.size
        if sample_count == 0:
            return np.zeros(0, dtype=np.float32)

        padded_length = ((sample_count + self.hop_size - 1) // self.hop_size) * self.hop_size
        padded_noisy = np.zeros(padded_length, dtype=np.float32)
        padded_dl = np.zeros(padded_length, dtype=np.float32)
        padded_noisy[:sample_count] = aligned.noisy
        padded_dl[:sample_count] = aligned.dl
        chunks: list[np.ndarray] = []
        for start in range(0, padded_length, self.hop_size):
            chunks.append(
                self.process_aligned_hop(
                    padded_noisy[start : start + self.hop_size],
                    padded_dl[start : start + self.hop_size],
                    processor,
                )
            )
        zero = np.zeros(self.hop_size, dtype=np.float32)
        chunks.append(self.process_aligned_hop(zero, zero, processor))
        delayed = np.concatenate(chunks)
        output = delayed[self.framing_delay_samples : self.framing_delay_samples + sample_count]
        if output.shape != (sample_count,):
            raise AssertionError(
                "Method 1 failed exact-length reconstruction: "
                f"expected={sample_count}, actual={output.size}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("Method 1 output contains non-finite samples")
        return np.ascontiguousarray(output, dtype=np.float32)

    def _hop(self, audio: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Method 1 {label} hop must be a numpy.ndarray")
        if audio.ndim != 1 or audio.shape != (self.hop_size,):
            raise ValueError(
                f"Method 1 {label} hop must have shape ({self.hop_size},), " f"got {audio.shape}"
            )
        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError(f"Method 1 {label} hop must have a floating dtype")
        if not np.all(np.isfinite(audio)):
            raise ValueError(f"Method 1 {label} hop must contain finite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(audio, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Method 1 {label} hop exceeds the finite range")
        return result

    def _spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("Method 1 processor output must be a numpy.ndarray")
        if spectrum.ndim != 1 or spectrum.shape != (self.num_frequency_bins,):
            raise ValueError(
                "Method 1 processor output must have shape "
                f"({self.num_frequency_bins},), got {spectrum.shape}"
            )
        if not np.issubdtype(spectrum.dtype, np.complexfloating):
            raise TypeError("Method 1 processor output must be complex floating")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("Method 1 processor output must contain finite values")
        return np.asarray(spectrum, dtype=np.complex128)

    def _sample_rate(self, sample_rate: int) -> None:
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate,
            (int, np.integer),
        ):
            raise TypeError("sample_rate must be an integer")
        if int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"Method 1 sample rate must be {self.sample_rate} Hz, got {sample_rate}"
            )


def select_noisy_spectrum(frame: NoisyDLSpectrumFrame) -> np.ndarray:
    return frame.noisy_spectrum


def select_dl_spectrum(frame: NoisyDLSpectrumFrame) -> np.ndarray:
    return frame.dl_spectrum
