"""
STFT / inverse-STFT utilities with overlap-add reconstruction.

This wraps scipy's STFT machinery with the framing conventions used
throughout the DSP pipeline (frame size, hop size, and window all come
from AppSettings so they're consistent everywhere -- see
senhance.config.settings).

ENSC 429 connection: this is the direct implementation of the DFT/FFT
and windowing concepts from the course, applied to short-time analysis
of a streaming signal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


class StreamingSTFT:
    """
    Maintains the overlap-add state needed to convert a stream of
    fixed-size input frames into a stream of enhanced output frames,
    processing in the frequency domain in between.

    Typical usage per audio frame:
        spectrum = stft.forward(frame)
        # ... modify `spectrum` (e.g., apply a gain mask) ...
        enhanced_frame = stft.inverse(spectrum)
    """

    def __init__(self, frame_size: int, hop_size: int, window: str = "hann"):
        """
        Args:
            frame_size: STFT analysis window length, in samples.
            hop_size: Number of samples between successive frames
                (frame_size - hop_size is the overlap).
            window: Window function name, passed to scipy.signal.get_window.
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = get_window(window, frame_size, fftbins=True).astype(np.float32)

        # Overlap-add buffer holds the "tail" of previous inverse-STFT
        # frames that overlaps with the next output frame.
        self._ola_buffer = np.zeros(frame_size, dtype=np.float32)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the analysis window and compute the FFT of one frame.

        Args:
            frame: 1-D array of length `frame_size`.

        Returns:
            Complex-valued FFT spectrum of the windowed frame.
        """
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Expected frame of length {self.frame_size}, got {len(frame)}. "
                f"TODO: add zero-padding here if variable-length frames are needed."
            )
        windowed = frame * self.window
        return np.fft.rfft(windowed)

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Convert a (possibly modified) spectrum back to the time domain
        and overlap-add it with the buffered tail from the previous call.

        Note: we deliberately do NOT re-apply the window here. A single
        Hann analysis window at 50% overlap satisfies the
        constant-overlap-add (COLA) condition on its own (the shifted
        copies sum to exactly 1.0). Applying Hann on *both* the analysis
        and synthesis sides (a squared window) breaks COLA at 50% overlap
        -- it would require 75% overlap instead. Since we window once in
        `forward()`, we skip windowing here to keep the OLA gain flat.
        See tests/unit/test_stft.py for a regression test on this.

        Args:
            spectrum: Complex-valued FFT spectrum (same shape as returned
                by `forward`, potentially after gain has been applied).

        Returns:
            1-D array of length `hop_size` -- the next chunk of the
            reconstructed output signal, ready to be streamed out.
        """
        time_domain = np.fft.irfft(spectrum, n=self.frame_size).astype(np.float32)

        # Overlap-add: combine with the buffered tail, then shift the buffer.
        output = self._ola_buffer[: self.hop_size].copy()
        output += time_domain[: self.hop_size]

        new_tail = np.zeros(self.frame_size, dtype=np.float32)
        new_tail[: self.frame_size - self.hop_size] = (
            self._ola_buffer[self.hop_size :] + time_domain[self.hop_size :]
        )
        self._ola_buffer = new_tail

        return output

    def reset(self) -> None:
        """Clear the overlap-add buffer (call when starting a new stream)."""
        self._ola_buffer = np.zeros(self.frame_size, dtype=np.float32)
