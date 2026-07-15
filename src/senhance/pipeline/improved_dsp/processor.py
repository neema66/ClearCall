"""Independent MCRA and soft-gain-fusion DSP enhancement pipeline."""

from __future__ import annotations

import numpy as np

from senhance.config.settings import AppSettings
from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.improved_dsp.config import ImprovedDSPConfig
from senhance.pipeline.improved_dsp.frame_adapter import StreamingFrameAdapter
from senhance.pipeline.improved_dsp.gains import (
    DecisionDirectedWienerGain,
    soft_fuse_gains,
    spectral_subtraction_gain,
)
from senhance.pipeline.improved_dsp.mcra import MCRANoiseEstimator
from senhance.pipeline.improved_dsp.smoothing import FinalGainSmoother
from senhance.pipeline.improved_dsp.stft import OverlapAddSTFT

logger = get_logger(__name__)


class ImprovedDSPPipeline(EnhancementStrategy):
    """Explainable DSP enhancer built independently from the legacy path."""

    def __init__(
        self,
        settings: AppSettings,
        config: ImprovedDSPConfig | None = None,
    ) -> None:
        if settings.audio.channels != 1:
            raise ValueError("ImprovedDSPPipeline supports mono audio only")

        self.settings = settings
        self.config = config or ImprovedDSPConfig()
        self.config.validate()

        self.frame_size = settings.frame_size_samples
        self.hop_size = settings.hop_size_samples
        self.num_freq_bins = self.frame_size // 2 + 1
        if self.frame_size != 2 * self.hop_size:
            raise ValueError("ImprovedDSPPipeline requires exactly 50% STFT overlap")

        hop_period_sec = self.hop_size / settings.audio.sample_rate
        minimum_window_frames = max(
            1,
            round(self.config.mcra.minimum_window_sec / hop_period_sec),
        )

        self._stft = OverlapAddSTFT(
            self.frame_size,
            self.hop_size,
            self.config.window,
        )
        self._frame_adapter = StreamingFrameAdapter(self.frame_size, self.hop_size)
        self._noise_estimator = MCRANoiseEstimator(
            self.num_freq_bins,
            minimum_window_frames,
            self.config.mcra,
        )
        self._wiener = DecisionDirectedWienerGain(
            self.num_freq_bins,
            self.config.wiener.smoothing_factor,
        )
        self._gain_smoother = FinalGainSmoother(
            self.num_freq_bins,
            self.config.spectral_subtraction.spectral_floor,
            self.config.final_gain_smoothing,
        )
        self._processing_mode: str | None = None

        logger.info(
            "ImprovedDSPPipeline initialized: frame_size=%d, hop_size=%d, "
            "minimum_window_frames=%d, smoothing=%s",
            self.frame_size,
            self.hop_size,
            minimum_window_frames,
            self.config.final_gain_smoothing.enabled,
        )

    @property
    def algorithmic_delay_samples(self) -> int:
        """One-hop delay introduced by causal previous/current-hop framing."""

        return self.hop_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process one complete overlapping analysis frame for offline comparison."""

        self._select_mode("frame")
        samples = self._validate_audio(frame, self.frame_size, "frame")
        return self._enhance_frame(samples)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Process one live input hop and return one delayed output hop."""

        self._select_mode("block")
        samples = self._validate_audio(block, self.hop_size, "block")
        frame = self._frame_adapter.push(samples)
        return self._enhance_frame(frame)

    def process_array(self, audio: np.ndarray) -> np.ndarray:
        """Enhance a complete mono signal through the real-time block path.

        The method pads the final partial hop, flushes one delayed hop, removes
        the known startup delay, and returns exactly the original sample count.
        """

        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("process_array expects one-dimensional mono audio")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Audio must contain only finite samples")
        self.reset()
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        padded_length = ((samples.size + self.hop_size - 1) // self.hop_size) * self.hop_size
        padded = np.zeros(padded_length, dtype=np.float32)
        padded[: samples.size] = samples

        chunks = []
        for start in range(0, padded_length, self.hop_size):
            chunks.append(self.process_block(padded[start : start + self.hop_size]))
        chunks.append(self.process_block(np.zeros(self.hop_size, dtype=np.float32)))

        delayed = np.concatenate(chunks)
        aligned = delayed[self.algorithmic_delay_samples :]
        return aligned[: samples.size].astype(np.float32, copy=False)

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        spectrum = self._stft.forward(frame)
        noisy_magnitude = np.abs(spectrum).astype(np.float32)
        noisy_power = (noisy_magnitude * noisy_magnitude).astype(np.float32)

        estimate = self._noise_estimator.update(noisy_power)
        noise_power = estimate.noise_power
        noise_magnitude = np.sqrt(np.maximum(noise_power, 0.0)).astype(np.float32)

        subtraction = spectral_subtraction_gain(
            noisy_magnitude,
            noise_magnitude,
            self.config.spectral_subtraction.oversubtraction_factor,
            self.config.spectral_subtraction.spectral_floor,
        )
        wiener = self._wiener.compute(noisy_power, noise_power)
        fused = soft_fuse_gains(
            subtraction,
            wiener,
            self.config.wiener.fusion_strength,
        )
        final_gain = self._gain_smoother.apply(fused)
        enhanced_spectrum = final_gain * spectrum
        return self._stft.inverse(enhanced_spectrum).astype(np.float32, copy=False)

    def _select_mode(self, requested: str) -> None:
        if self._processing_mode is None:
            self._processing_mode = requested
        elif self._processing_mode != requested:
            raise RuntimeError(
                "Cannot mix frame and block processing state; call reset() before switching modes"
            )

    @staticmethod
    def _validate_audio(audio: np.ndarray, expected_length: int, label: str) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or samples.shape[0] != expected_length:
            raise ValueError(f"Expected {label} shape ({expected_length},), got {samples.shape}")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Audio {label} must contain only finite samples")
        return samples

    def reset(self) -> None:
        """Reset every stateful component and permit selecting either API mode."""

        self._stft.reset()
        self._frame_adapter.reset()
        self._noise_estimator.reset()
        self._wiener.reset()
        self._gain_smoother.reset()
        self._processing_mode = None
