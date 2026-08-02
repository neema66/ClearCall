"""Live block-mode tests for original_dsp (DSPPipeline.process_block).

Mirrors tests/unit/improved_dsp/test_streaming.py's pattern -- these two
methods now have parallel frame/block/live-strategy structure, verified
independently per method ownership (see docs/project_structure.md).
"""

import numpy as np
import pytest

from senhance.config.settings import AppSettings
from senhance.pipeline.dsp.frame_adapter import DSPFrameAdapter
from senhance.pipeline.dsp.live_strategy import DSPBlockStrategy
from senhance.pipeline.dsp.processor import DSPPipeline


def test_frame_adapter_uses_previous_and_current_hops() -> None:
    adapter = DSPFrameAdapter(frame_size=8, hop_size=4)
    first = np.arange(4, dtype=np.float32)
    second = np.arange(4, 8, dtype=np.float32)

    assert np.array_equal(adapter.push(first), np.concatenate((np.zeros(4), first)))
    assert np.array_equal(adapter.push(second), np.concatenate((first, second)))

    adapter.reset()
    assert np.array_equal(adapter.push(second)[:4], np.zeros(4))


def test_frame_adapter_rejects_wrong_hop_size() -> None:
    adapter = DSPFrameAdapter(frame_size=8, hop_size=4)
    with pytest.raises(ValueError):
        adapter.push(np.zeros(3, dtype=np.float32))


def test_frame_adapter_requires_fifty_percent_overlap() -> None:
    with pytest.raises(ValueError):
        DSPFrameAdapter(frame_size=9, hop_size=4)


def test_process_block_returns_one_finite_hop_sized_block() -> None:
    settings = AppSettings()
    pipeline = DSPPipeline(settings)
    rng = np.random.default_rng(30)
    block = (0.05 * rng.normal(size=settings.hop_size_samples)).astype(np.float32)

    output = pipeline.process_block(block)

    assert output.shape == (settings.hop_size_samples,)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_block_mode_matches_config_block_size_and_cannot_mix_without_reset() -> None:
    settings = AppSettings()
    pipeline = DSPPipeline(settings)
    # audio.block_size (480) must equal hop_size_samples for the live loop
    # to work at all -- this is the mismatch that used to crash on the
    # first live frame; confirm it's exactly hop_size now.
    assert settings.audio.block_size == settings.hop_size_samples

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
    pipeline = DSPPipeline(settings)
    strategy = DSPBlockStrategy(pipeline)
    block = np.zeros(settings.audio.block_size, dtype=np.float32)

    output = strategy.process(block)

    assert output.shape == block.shape
    assert pipeline._processing_mode == "block"


def test_live_strategy_reset_delegates_to_pipeline() -> None:
    settings = AppSettings()
    pipeline = DSPPipeline(settings)
    strategy = DSPBlockStrategy(pipeline)
    strategy.process(np.zeros(settings.audio.block_size, dtype=np.float32))

    strategy.reset()

    assert pipeline._processing_mode is None
    assert np.all(pipeline._stft._ola_buffer == 0)
