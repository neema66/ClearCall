"""
Unit tests for StreamingSTFT.

Core correctness check: a signal passed through forward() then inverse()
(with no modification of the spectrum in between) should reconstruct
close to the original signal, once enough frames have been processed to
fill the overlap-add pipeline.
"""

import numpy as np
import pytest

from senhance.pipeline.dsp.stft import StreamingSTFT


def test_forward_inverse_reconstruction():
    frame_size = 480
    hop_size = 240  # 50% overlap
    stft = StreamingSTFT(frame_size=frame_size, hop_size=hop_size, window="hann")

    rng = np.random.default_rng(seed=42)
    signal = rng.normal(size=frame_size * 10).astype(np.float32) * 0.1

    reconstructed_chunks = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start : start + frame_size]
        spectrum = stft.forward(frame)
        reconstructed_chunks.append(stft.inverse(spectrum))

    reconstructed = np.concatenate(reconstructed_chunks)

    # Skip the first couple of hops -- the overlap-add buffer needs to
    # "fill up" before reconstruction is accurate, which is expected
    # behavior, not a bug.
    skip = 2 * hop_size
    original_aligned = signal[: len(reconstructed)][skip:]
    reconstructed_aligned = reconstructed[skip:]

    # TODO: tighten this tolerance once perfect-reconstruction windowing
    # (COLA condition) is verified for the chosen window/hop combination.
    assert np.allclose(original_aligned, reconstructed_aligned, atol=0.05)


def test_forward_rejects_wrong_length():
    stft = StreamingSTFT(frame_size=480, hop_size=240)
    with pytest.raises(ValueError):
        stft.forward(np.zeros(100, dtype=np.float32))


def test_reset_clears_overlap_buffer():
    stft = StreamingSTFT(frame_size=480, hop_size=240)
    frame = np.random.randn(480).astype(np.float32) * 0.1
    stft.forward(frame)
    stft.inverse(stft.forward(frame))
    stft.reset()
    assert np.all(stft._ola_buffer == 0)
