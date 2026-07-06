"""
DSP enhancement pipeline: wires together STFT, noise estimation,
spectral subtraction, and the Wiener filter into a single
EnhancementStrategy implementation.

This is the classical-DSP counterpart to
senhance.pipeline.dl.deepfilternet_wrapper.DeepFilterNetPipeline -- both
implement EnhancementStrategy so they can be swapped via config.
"""

from __future__ import annotations

import numpy as np

from senhance.config.settings import AppSettings
from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.dsp.noise_estimator import NoiseEstimator
from senhance.pipeline.dsp.spectral_subtraction import spectral_subtract
from senhance.pipeline.dsp.stft import StreamingSTFT
from senhance.pipeline.dsp.wiener_filter import WienerFilter

logger = get_logger(__name__)


class DSPPipeline(EnhancementStrategy):
    """
    Classical DSP speech enhancement:
        noisy frame -> STFT -> noise estimate update
                    -> spectral subtraction -> Wiener filter -> ISTFT
                    -> enhanced frame (overlap-added)

    Both spectral subtraction and the Wiener filter are applied in
    sequence (subtraction first, as a coarse pass, then Wiener filtering
    for a smoother final gain) -- this ordering is a design choice worth
    revisiting/tuning as a team based on listening tests.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        frame_size = settings.frame_size_samples
        hop_size = settings.hop_size_samples
        num_freq_bins = frame_size // 2 + 1

        self._stft = StreamingSTFT(
            frame_size=frame_size, hop_size=hop_size, window=settings.dsp.window
        )
        self._noise_estimator = NoiseEstimator(
            num_freq_bins=num_freq_bins,
            calibration_frames=settings.dsp.noise_estimation_frames,
        )
        self._wiener = WienerFilter(
            num_freq_bins=num_freq_bins,
            smoothing_factor=settings.dsp.wiener_filter.smoothing_factor,
        )

        # Input frames arrive in blocks of `block_size` (from config,
        # e.g. 480 samples), but the STFT operates on `frame_size` (e.g.
        # 20ms). This internal buffer accumulates input until a full STFT
        # frame is available. TODO: if block_size == frame_size exactly
        # in your config, this is a pass-through; otherwise this handles
        # the mismatch.
        self._input_accum = np.zeros(0, dtype=np.float32)

        logger.info(
            "DSPPipeline initialized: frame_size=%d, hop_size=%d, num_freq_bins=%d",
            frame_size,
            hop_size,
            num_freq_bins,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one block of audio. See class docstring for the pipeline
        stages. Returns a block of the same length as the input.

        TODO: this simplified version assumes `frame` is already exactly
        `frame_size` samples (i.e. block_size == frame_size in config).
        If you configure a smaller block_size for lower latency, extend
        this method to accumulate/split via `self._input_accum`.
        """
        cfg = self.settings.dsp

        spectrum = self._stft.forward(frame)
        magnitude = np.abs(spectrum)

        noise_estimate = self._noise_estimator.update(magnitude)

        spectrum = spectral_subtract(
            spectrum,
            noise_estimate,
            oversubtraction_factor=cfg.spectral_subtraction.oversubtraction_factor,
            spectral_floor=cfg.spectral_subtraction.spectral_floor,
        )

        spectrum = self._wiener.apply(spectrum, noise_estimate)

        return self._stft.inverse(spectrum)

    def reset(self) -> None:
        """Reset all stateful sub-components (call between streams)."""
        self._stft.reset()
        self._noise_estimator.reset()
        self._wiener.reset()
        self._input_accum = np.zeros(0, dtype=np.float32)
