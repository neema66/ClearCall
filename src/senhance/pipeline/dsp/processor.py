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
from senhance.pipeline.dsp.frame_adapter import DSPFrameAdapter
from senhance.pipeline.dsp.noise_estimator import NoiseEstimator
from senhance.pipeline.dsp.spectral_subtraction import spectral_subtract
from senhance.pipeline.dsp.stft import StreamingSTFT
from senhance.pipeline.dsp.wiener_filter import WienerFilter

logger = get_logger(__name__)


class DSPPipeline(EnhancementStrategy):
    """
    Classical DSP speech enhancement:
        noisy frame -> STFT -> noise estimate update
                    -> spectral subtraction gain, Wiener gain (independent)
                    -> combined gain applied once -> ISTFT
                    -> enhanced frame (overlap-added)

    Spectral subtraction and the Wiener filter each estimate their own
    suppression gain from the *same* true noisy spectrum and noise
    estimate, and those two gains are then multiplied together and
    applied once. This is deliberate: both stages derive their gain from
    an SNR estimate against the noise floor, so feeding one stage's
    output into the other as if it were "the noisy signal" would silently
    compound two suppression decisions into one. An earlier version did
    exactly that (subtraction's output fed into the Wiener filter) and
    was measured to make quality *worse* than doing no processing at all
    on the team's VoiceBank+DEMAND subset (PESQ 1.28 vs. a 2.28 do-nothing
    baseline). The default `oversubtraction_factor`, `spectral_floor`,
    `wiener_filter.smoothing_factor`, and `noise_estimator.bias_correction`
    values in config/default.yaml were all retuned together against that
    same subset after this fix -- see docs/evaluation_plan.md for the
    before/after numbers and search methodology.

    `process()` takes a complete `frame_size` frame (offline/frame-wise
    use, e.g. `senhance.evaluation.evaluate.enhance_offline`).
    `process_block()` takes a `hop_size` live callback block instead,
    building each analysis frame from the previous + current hop -- see
    `senhance.pipeline.dsp.live_strategy.DSPBlockStrategy`, the adapter
    that exposes this pipeline to `AudioStreamManager`'s live loop.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.frame_size = frame_size = settings.frame_size_samples
        self.hop_size = hop_size = settings.hop_size_samples
        num_freq_bins = frame_size // 2 + 1

        self._stft = StreamingSTFT(
            frame_size=frame_size, hop_size=hop_size, window=settings.dsp.window
        )
        frame_period_sec = hop_size / settings.audio.sample_rate
        window_frames = max(
            1, round(settings.dsp.noise_estimator.window_sec / frame_period_sec)
        )
        self._noise_estimator = NoiseEstimator(
            num_freq_bins=num_freq_bins,
            window_frames=window_frames,
            bias_correction=settings.dsp.noise_estimator.bias_correction,
        )
        self._wiener = WienerFilter(
            num_freq_bins=num_freq_bins,
            smoothing_factor=settings.dsp.wiener_filter.smoothing_factor,
        )

        # Input arrives as either a complete `frame_size` frame (process(),
        # offline/frame-wise use) or a `hop_size` live callback block
        # (process_block(); the adapter builds a full frame from the
        # previous + current hop). _select_mode() prevents mixing the two
        # without an intervening reset(), matching
        # senhance.pipeline.improved_dsp.processor.ImprovedDSPPipeline.
        self._frame_adapter = DSPFrameAdapter(frame_size, hop_size)
        self._processing_mode: str | None = None

        logger.info(
            "DSPPipeline initialized: frame_size=%d, hop_size=%d, num_freq_bins=%d",
            frame_size,
            hop_size,
            num_freq_bins,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one complete STFT analysis frame (`frame_size` samples,
        e.g. 960) and return one `hop_size`-length enhanced chunk. See
        class docstring for the pipeline stages. Used for offline/
        frame-wise processing; see `process_block()` for live callback
        blocks of `hop_size` samples.
        """
        self._select_mode("frame")
        return self._enhance_frame(frame)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """
        Process one live audio-callback block (`hop_size` samples, e.g.
        480 -- matches `AudioStreamManager`'s `audio.block_size`) and
        return one equally sized enhanced block.

        `AudioStreamManager` always calls `process()` with `block_size`
        chunks, never full `frame_size` frames, so this is the entry
        point a live `EnhancementStrategy` adapter must use (see
        `senhance.pipeline.dsp.live_strategy.DSPBlockStrategy`). Builds
        each full analysis frame from the previous + current hop via
        `DSPFrameAdapter`, so output is delayed by one hop relative to
        input (the same causal delay `ImprovedDSPPipeline` has).
        """
        self._select_mode("block")
        frame = self._frame_adapter.push(block)
        return self._enhance_frame(frame)

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.settings.dsp

        spectrum = self._stft.forward(frame)
        magnitude = np.abs(spectrum)

        noise_estimate = self._noise_estimator.update(magnitude)

        # Each stage computes its gain from the TRUE noisy spectrum (not
        # from the other stage's output -- see class docstring for why),
        # then the two gains are combined multiplicatively and applied
        # once to the original spectrum.
        subtracted_spectrum = spectral_subtract(
            spectrum,
            noise_estimate,
            oversubtraction_factor=cfg.spectral_subtraction.oversubtraction_factor,
            spectral_floor=cfg.spectral_subtraction.spectral_floor,
        )
        wiener_spectrum = self._wiener.apply(spectrum, noise_estimate)

        subtraction_gain = np.abs(subtracted_spectrum) / (magnitude + 1e-12)
        wiener_gain = np.abs(wiener_spectrum) / (magnitude + 1e-12)
        enhanced_spectrum = (subtraction_gain * wiener_gain) * spectrum

        return self._stft.inverse(enhanced_spectrum)

    def _select_mode(self, requested: str) -> None:
        if self._processing_mode is None:
            self._processing_mode = requested
        elif self._processing_mode != requested:
            raise RuntimeError(
                "Cannot mix frame and block processing state; call reset() before switching modes"
            )

    def reset(self) -> None:
        """Reset all stateful sub-components (call between streams)."""
        self._stft.reset()
        self._frame_adapter.reset()
        self._noise_estimator.reset()
        self._wiener.reset()
        self._processing_mode = None
