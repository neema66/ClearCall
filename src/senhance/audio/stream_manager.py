"""
Real-time audio stream manager.

Wraps `sounddevice` to capture from a physical microphone and play out to
a virtual audio device (e.g., VB-Cable on Windows -- see docs/setup.md),
running the enhancement pipeline in between.

Safe Track scope (Windows-only): this module assumes VB-Cable (or an
equivalent virtual audio cable) is already installed and selected as the
`output_device` in config/default.yaml. See docs/setup.md for setup steps.

Threading model:
- `_audio_callback` runs on sounddevice's real-time thread. It MUST NOT
  block, allocate heavily, or call logging/I/O directly (see
  senhance.logging_setup.logger for details). It only copies data into
  the input AudioBuffer.
- `run()` runs the processing loop on the calling thread (or a worker
  thread you spawn), pulling frames from the input buffer, running the
  enhancement pipeline, and writing to the output stream.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sounddevice as sd

from senhance.audio.buffer import AudioBuffer
from senhance.config.settings import AppSettings
from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy

logger = get_logger(__name__)


class AudioStreamManager:
    """
    Manages the live microphone -> enhancement -> virtual microphone loop.
    """

    def __init__(self, settings: AppSettings, strategy: EnhancementStrategy):
        """
        Args:
            settings: Loaded AppSettings (see senhance.config.settings).
            strategy: An EnhancementStrategy implementation (e.g.
                DSPPipeline) that will process each audio frame.
        """
        self.settings = settings
        self.strategy = strategy
        self._input_buffer = AudioBuffer(max_frames=50)
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._running = False

        # Lightweight counters updated in the real-time callback and read
        # (not logged directly) from the processing loop -- see the
        # real-time-safety note in senhance.logging_setup.logger.
        self.callback_overrun_count = 0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        sounddevice InputStream callback. Runs on the real-time audio
        thread -- keep this function fast and non-blocking.
        """
        if status:
            # `status` flags things like input overflow. We only increment
            # a counter here; never log synchronously from this thread.
            self.callback_overrun_count += 1
        # Copy is required: `indata` is a buffer owned by sounddevice/PortAudio
        # and will be reused after this callback returns.
        self._input_buffer.put_frame(indata.copy())

    def start(self) -> None:
        """Open the input and output audio streams and begin capture."""
        block_size = self.settings.audio.block_size
        sample_rate = self.settings.audio.sample_rate
        channels = self.settings.audio.channels

        logger.info(
            "Starting audio streams: sample_rate=%d, block_size=%d, channels=%d",
            sample_rate,
            block_size,
            channels,
        )

        self._input_stream = sd.InputStream(
            device=self.settings.audio.input_device,
            samplerate=sample_rate,
            channels=channels,
            blocksize=block_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._output_stream = sd.OutputStream(
            device=self.settings.audio.output_device,
            samplerate=sample_rate,
            channels=channels,
            blocksize=block_size,
            dtype="float32",
        )

        self._input_stream.start()
        self._output_stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop and close both audio streams."""
        self._running = False
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
        logger.info(
            "Audio streams stopped. Dropped frames: %d, callback overruns: %d",
            self._input_buffer.dropped_frame_count,
            self.callback_overrun_count,
        )

    def run(self) -> None:
        """
        Processing loop: pulls frames from the input buffer, runs the
        enhancement strategy, and writes the result to the output stream.

        This is intentionally simple for the Safe Track scope -- a
        straightforward blocking loop rather than a fully decoupled
        worker-thread architecture. See docs/architecture.md for the
        upgrade path if stricter real-time guarantees are needed later.
        """
        logger.info("Processing loop started. Press Ctrl+C to stop.")
        try:
            while self._running:
                frame = self._input_buffer.get_frame(timeout=1.0)
                if frame is None:
                    continue  # no data yet; keep waiting
                enhanced = self.strategy.process(frame.flatten())
                if self._output_stream is not None:
                    self._output_stream.write(
                        enhanced.reshape(-1, self.settings.audio.channels).astype("float32")
                    )
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    @staticmethod
    def list_devices() -> None:
        """Print available audio devices -- helpful for setting
        input_device / output_device in config/default.yaml."""
        print(sd.query_devices())
