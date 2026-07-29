"""STFT, framing, and assembled improved-pipeline tests."""

import dataclasses

import numpy as np
import pytest

from senhance.config.settings import AppSettings
from senhance.pipeline.improved_dsp.config import (
    FinalGainSmoothingConfig,
    ImprovedDSPConfig,
    SpectralSubtractionConfig,
    WienerConfig,
)
from senhance.pipeline.improved_dsp.frame_adapter import StreamingFrameAdapter
from senhance.pipeline.improved_dsp.live_strategy import ImprovedDSPBlockStrategy
from senhance.pipeline.improved_dsp.processor import ImprovedDSPPipeline
from senhance.pipeline.improved_dsp.stft import OverlapAddSTFT


def _identity_config() -> ImprovedDSPConfig:
    return ImprovedDSPConfig(
        spectral_subtraction=SpectralSubtractionConfig(
            oversubtraction_factor=0.0,
            spectral_floor=0.0,
        ),
        wiener=WienerConfig(smoothing_factor=0.7, fusion_strength=0.0),
        final_gain_smoothing=FinalGainSmoothingConfig(enabled=False),
    )


def test_frame_adapter_uses_previous_and_current_hops() -> None:
    adapter = StreamingFrameAdapter(frame_size=8, hop_size=4)
    first = np.arange(4, dtype=np.float32)
    second = np.arange(4, 8, dtype=np.float32)

    assert np.array_equal(adapter.push(first), np.concatenate((np.zeros(4), first)))
    assert np.array_equal(adapter.push(second), np.concatenate((first, second)))

    adapter.reset()
    assert np.array_equal(adapter.push(second)[:4], np.zeros(4))


def test_overlap_add_stft_reconstructs_after_startup() -> None:
    frame_size = 64
    hop_size = 32
    stft = OverlapAddSTFT(frame_size, hop_size)
    rng = np.random.default_rng(20)
    signal = rng.normal(size=frame_size * 12).astype(np.float32)

    chunks = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start : start + frame_size]
        chunks.append(stft.inverse(stft.forward(frame)))
    reconstructed = np.concatenate(chunks)

    skip = hop_size
    assert np.allclose(reconstructed[skip:], signal[: len(reconstructed)][skip:], atol=1e-6)


def test_process_frame_returns_one_finite_hop() -> None:
    settings = AppSettings()
    pipeline = ImprovedDSPPipeline(settings)
    rng = np.random.default_rng(21)
    frame = (0.05 * rng.normal(size=settings.frame_size_samples)).astype(np.float32)

    output = pipeline.process(frame)

    assert output.shape == (settings.hop_size_samples,)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_block_mode_has_fixed_shape_and_cannot_mix_without_reset() -> None:
    settings = AppSettings()
    pipeline = ImprovedDSPPipeline(settings)
    block = np.zeros(settings.hop_size_samples, dtype=np.float32)

    assert pipeline.process_block(block).shape == block.shape
    with pytest.raises(RuntimeError, match="call reset"):
        pipeline.process(np.zeros(settings.frame_size_samples, dtype=np.float32))

    pipeline.reset()
    assert pipeline.process(np.zeros(settings.frame_size_samples, dtype=np.float32)).shape == (
        settings.hop_size_samples,
    )


def test_live_strategy_maps_shared_process_contract_to_block_mode() -> None:
    settings = AppSettings()
    pipeline = ImprovedDSPPipeline(settings)
    strategy = ImprovedDSPBlockStrategy(pipeline)
    block = np.zeros(settings.audio.block_size, dtype=np.float32)

    output = strategy.process(block)

    assert output.shape == block.shape
    assert pipeline._processing_mode == "block"


def test_process_array_compensates_delay_and_preserves_identity() -> None:
    settings = AppSettings()
    pipeline = ImprovedDSPPipeline(settings, _identity_config())
    rng = np.random.default_rng(22)
    signal = (0.05 * rng.normal(size=3 * settings.hop_size_samples + 137)).astype(np.float32)

    output = pipeline.process_array(signal)

    assert output.shape == signal.shape
    assert np.allclose(output, signal, atol=1e-6)


def test_reset_makes_repeated_array_processing_deterministic() -> None:
    settings = AppSettings()
    config = dataclasses.replace(
        ImprovedDSPConfig(),
        final_gain_smoothing=FinalGainSmoothingConfig(enabled=True),
    )
    pipeline = ImprovedDSPPipeline(settings, config)
    rng = np.random.default_rng(23)
    signal = (0.05 * rng.normal(size=4 * settings.hop_size_samples)).astype(np.float32)

    first = pipeline.process_array(signal)
    second = pipeline.process_array(signal)

    assert np.array_equal(first, second)
