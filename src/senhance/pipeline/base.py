"""
Common interface for all enhancement pipelines.

Both the classical DSP pipeline and the deep learning pipeline implement
this interface, which is what lets them be swapped in and out via config
without touching the audio I/O or evaluation code. Keep this interface
minimal and stable -- it's the contract the whole team codes against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EnhancementStrategy(ABC):
    """
    Abstract base class for a speech enhancement algorithm.

    Implementations receive one frame of noisy audio and return one frame
    of enhanced audio. Framing/overlap-add bookkeeping (if needed) is the
    responsibility of the implementation, not the caller.
    """

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single frame of audio.

        Args:
            frame: 1-D float32 numpy array of audio samples (mono).

        Returns:
            1-D float32 numpy array of the same length, with noise
            suppressed.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any internal state (e.g. noise estimate, overlap-add buffer).
        Override if your implementation is stateful. Default is a no-op
        for stateless strategies.
        """
        pass
