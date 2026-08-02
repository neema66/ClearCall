"""Causal 480-to-960 style framing for live audio blocks.

Self-contained for original_dsp -- not shared with
senhance.pipeline.improved_dsp.frame_adapter even though the mechanics are
identical, per docs/project_structure.md's dependency rules ("do not
create a shared helper merely because two algorithms currently look
similar"): each method owns its framing/state independently.
"""

from __future__ import annotations

import numpy as np


class DSPFrameAdapter:
    """Construct each 50%-overlapped frame from the previous/current hops."""

    def __init__(self, frame_size: int, hop_size: int) -> None:
        if frame_size != 2 * hop_size:
            raise ValueError("DSPFrameAdapter requires frame_size == 2 * hop_size")
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._previous_hop = np.zeros(hop_size, dtype=np.float32)

    def push(self, block: np.ndarray) -> np.ndarray:
        """Return ``previous_hop + current_hop`` and retain the current hop."""

        current = np.asarray(block, dtype=np.float32)
        if current.ndim != 1 or current.shape[0] != self.hop_size:
            raise ValueError(f"Expected block shape ({self.hop_size},), got {current.shape}")
        if not np.all(np.isfinite(current)):
            raise ValueError("Audio block must contain only finite samples")

        frame = np.concatenate((self._previous_hop, current))
        self._previous_hop = current.copy()
        return frame

    def reset(self) -> None:
        """Clear the retained input hop."""

        self._previous_hop.fill(0.0)
