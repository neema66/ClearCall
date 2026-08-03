"""Independent whole-array Hybrid Method 2 pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.hybrid.method2.config import Method2Config
from senhance.pipeline.hybrid.method2.controller import (
    DSPGuidedLevelController,
    Method2Result,
    Method2Statistics,
)

logger = get_logger(__name__)


class DSPArrayEnhancer(Protocol):
    """Structural interface required from an offline DSP branch."""

    def process_array(self, audio: np.ndarray) -> np.ndarray: ...


class DLArrayEnhancer(Protocol):
    """Structural interface required from an offline DL branch."""

    def enhance_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class HybridMethod2Pipeline(EnhancementStrategy):
    """Compose injected DSP/DL branches through the Method 2 controller.

    The class imports neither branch implementation. Application and evaluation
    orchestration inject compatible branch objects, preserving method ownership.
    """

    def __init__(
        self,
        config: Method2Config,
        dsp_enhancer: DSPArrayEnhancer,
        dl_enhancer: DLArrayEnhancer,
    ) -> None:
        if not isinstance(config, Method2Config):
            raise TypeError("config must be Method2Config")
        config.validate()
        if not callable(getattr(dsp_enhancer, "process_array", None)):
            raise TypeError("dsp_enhancer must provide process_array(audio)")
        if not callable(getattr(dl_enhancer, "enhance_array", None)):
            raise TypeError(
                "dl_enhancer must provide enhance_array(audio, sample_rate)"
            )

        self.config = config
        self._dsp_enhancer = dsp_enhancer
        self._dl_enhancer = dl_enhancer
        self._controller = DSPGuidedLevelController(config)
        self.last_statistics: Method2Statistics | None = None

    def enhance_with_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Method2Result:
        """Run both injected branches once and return audio plus diagnostics."""

        model_input = self._validated_input(audio, sample_rate)

        # Each branch receives its own copy so one branch cannot alter the
        # samples observed by the other branch.
        dsp_output = self._dsp_enhancer.process_array(model_input.copy())
        dl_output = self._dl_enhancer.enhance_array(model_input.copy(), sample_rate)
        result = self._controller.apply(dsp_output, dl_output)
        if result.audio.shape != model_input.shape:
            raise ValueError(
                "Hybrid Method 2 output length must equal input length: "
                f"input={model_input.size}, output={result.audio.size}"
            )

        self.last_statistics = result.statistics
        logger.info(
            "Hybrid Method 2: gain=%.6f, DSP RMS=%.6f, DL RMS=%.6f, "
            "peak=%.6f, clipped_samples=%d",
            result.statistics.applied_gain,
            result.statistics.dsp_rms,
            result.statistics.dl_rms,
            result.statistics.output_peak_abs,
            result.statistics.clipped_sample_count,
        )
        return result

    def enhance_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return only the exact-length float32 Method 2 waveform."""

        return self.enhance_with_result(audio, sample_rate).audio

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Method 2 currently requires complete offline branch outputs."""

        raise NotImplementedError(
            "HybridMethod2Pipeline is whole-array/offline only; use "
            "enhance_array(audio, sample_rate). A causal level estimator and "
            "stateful streaming DeepFilterNet API are required for live use."
        )

    def reset(self) -> None:
        """Reset injected branch state when supported and clear diagnostics."""

        for branch in (self._dsp_enhancer, self._dl_enhancer):
            reset = getattr(branch, "reset", None)
            if callable(reset):
                reset()
        self.last_statistics = None

    def _validated_input(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError("audio must be a numpy.ndarray")
        if audio.ndim != 1:
            raise ValueError("audio must be one-dimensional mono audio")
        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError("audio must have a real floating dtype")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must contain only finite samples")
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate,
            (int, np.integer),
        ):
            raise TypeError("sample_rate must be an integer")
        if int(sample_rate) != self.config.sample_rate:
            raise ValueError(
                f"Hybrid Method 2 requires {self.config.sample_rate} Hz audio, "
                f"got {sample_rate}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            model_input = np.array(audio, dtype=np.float32, order="C", copy=True)
        if not np.all(np.isfinite(model_input)):
            raise ValueError("audio exceeds the float32 finite range")
        return model_input
