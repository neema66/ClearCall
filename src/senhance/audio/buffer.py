"""
Simple thread-safe audio buffer (Safe Track implementation).

Design note (see docs/architecture.md, "What Changed" section):
This uses Python's standard `queue.Queue` rather than a hand-rolled
lock-free ring buffer. That's a deliberate Safe Track simplification --
it's much faster to get correct, at the cost of being technically
"blocking" under contention. If real-time performance testing in Week 3-4
shows this is a bottleneck, this is the file to replace with a proper
lock-free circular buffer (upgrade path, not a rewrite of the rest of
the system, since callers only depend on put_frame/get_frame).
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np

from senhance.logging_setup.logger import get_logger

logger = get_logger(__name__)


class AudioBuffer:
    """
    Thread-safe FIFO buffer for passing fixed-size audio frames between
    the real-time capture callback and the processing loop.

    TODO (stretch goal): replace with a lock-free ring buffer if profiling
    shows queue contention is adding meaningful latency (see
    docs/architecture.md, Section 2.3, Threading Model).
    """

    def __init__(self, max_frames: int = 50):
        """
        Args:
            max_frames: Maximum number of frames the buffer will hold
                before dropping the oldest one. Sized generously so a
                brief processing hiccup doesn't immediately cause
                audible dropouts, but not so large that latency builds
                up unboundedly.
        """
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_frames)
        self.dropped_frame_count = 0

    def put_frame(self, frame: np.ndarray) -> None:
        """
        Push a new audio frame into the buffer. Called from the real-time
        capture callback -- must be fast and non-raising.

        If the buffer is full (processing is falling behind), the oldest
        frame is dropped to make room, and a counter is incremented so
        this can be surfaced in logs/metrics rather than silently losing
        data.
        """
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
                self.dropped_frame_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # extremely unlikely race; frame is dropped

    def get_frame(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """
        Pop the next available frame. Called from the processing loop
        (not the real-time callback).

        Args:
            timeout: Seconds to wait for a frame before giving up.

        Returns:
            The next frame, or None if the timeout elapsed with nothing
            available.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """Current number of buffered frames (approximate, for monitoring)."""
        return self._queue.qsize()
