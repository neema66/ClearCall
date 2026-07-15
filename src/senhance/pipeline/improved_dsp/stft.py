"""Independent Hann STFT and overlap-add reconstruction."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


class OverlapAddSTFT:
    """Streaming STFT specialized for a periodic Hann window at 50% overlap."""

    def __init__(self, frame_size: int, hop_size: int, window: str = "hann") -> None:
        if frame_size <= 0 or hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive")
        if frame_size != 2 * hop_size:
            raise ValueError("Improved DSP requires 50% overlap (frame_size == 2 * hop_size)")
        if window != "hann":
            raise ValueError("Improved DSP currently supports only a Hann window")

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.num_freq_bins = frame_size // 2 + 1
        self.window = get_window(window, frame_size, fftbins=True).astype(np.float32)
        self._overlap = np.zeros(frame_size - hop_size, dtype=np.float32)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Window and transform one complete analysis frame."""

        samples = np.asarray(frame, dtype=np.float32)
        if samples.ndim != 1 or samples.shape[0] != self.frame_size:
            raise ValueError(f"Expected frame shape ({self.frame_size},), got {samples.shape}")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Audio frame must contain only finite samples")
        return np.fft.rfft(samples * self.window)

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        """Inverse-transform one spectrum and return the next output hop."""

        bins = np.asarray(spectrum)
        if bins.ndim != 1 or bins.shape[0] != self.num_freq_bins:
            raise ValueError(f"Expected spectrum shape ({self.num_freq_bins},), got {bins.shape}")
        if not np.all(np.isfinite(bins)):
            raise ValueError("Spectrum must contain only finite values")

        time_domain = np.fft.irfft(bins, n=self.frame_size).astype(np.float32)
        output = time_domain[: self.hop_size].copy()
        output += self._overlap[: self.hop_size]
        self._overlap = time_domain[self.hop_size :].copy()
        return output

    def reset(self) -> None:
        """Clear overlap-add state."""

        self._overlap.fill(0.0)
