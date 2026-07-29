"""Adapter from the shared enhancement interface to improved-DSP block mode."""

from __future__ import annotations

import numpy as np

from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.improved_dsp.processor import ImprovedDSPPipeline


class ImprovedDSPBlockStrategy(EnhancementStrategy):
    """Expose ``process_block`` as ``process`` for ``AudioStreamManager``."""

    def __init__(self, pipeline: ImprovedDSPPipeline) -> None:
        self.pipeline = pipeline

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process one live callback block and return one equally sized block."""

        return self.pipeline.process_block(frame)

    def reset(self) -> None:
        """Reset the delegated improved pipeline."""

        self.pipeline.reset()
