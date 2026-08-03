"""Hybrid Method 2 branch-composition and boundary tests."""

from __future__ import annotations

import numpy as np
import pytest

from senhance.config.settings import load_settings
from senhance.evaluation.evaluate import _build_pipeline
from senhance.pipeline.hybrid.method2 import HybridMethod2Pipeline


class FakeDSP:
    def __init__(self, scale: float = 1.02) -> None:
        self.scale = scale
        self.calls = 0
        self.reset_calls = 0
        self.inputs: list[np.ndarray] = []

    def process_array(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.inputs.append(audio)
        return np.asarray(audio * self.scale, dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1


class FakeDL:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.inputs: list[np.ndarray] = []
        self.sample_rates: list[int] = []

    def enhance_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        self.inputs.append(audio)
        self.sample_rates.append(sample_rate)
        return audio.copy()

    def reset(self) -> None:
        self.reset_calls += 1


def test_pipeline_calls_each_branch_once_and_preserves_input(method2_config):
    dsp = FakeDSP()
    dl = FakeDL()
    pipeline = HybridMethod2Pipeline(method2_config, dsp, dl)
    noisy = np.linspace(-0.2, 0.2, 480, dtype=np.float32)
    original = noisy.copy()

    result = pipeline.enhance_with_result(noisy, 48_000)

    assert dsp.calls == 1
    assert dl.calls == 1
    assert dl.sample_rates == [48_000]
    np.testing.assert_array_equal(noisy, original)
    assert not np.shares_memory(dsp.inputs[0], noisy)
    assert not np.shares_memory(dl.inputs[0], noisy)
    assert not np.shares_memory(dsp.inputs[0], dl.inputs[0])
    assert result.audio.shape == noisy.shape
    assert result.audio.dtype == np.float32
    assert pipeline.last_statistics is result.statistics


def test_dsp_branch_mutation_cannot_change_dl_input(method2_config):
    class MutatingDSP(FakeDSP):
        def process_array(self, audio: np.ndarray) -> np.ndarray:
            self.calls += 1
            self.inputs.append(audio)
            audio[:] = 0.0
            return audio

    dsp = MutatingDSP()
    dl = FakeDL()
    pipeline = HybridMethod2Pipeline(method2_config, dsp, dl)
    noisy = np.full(32, 0.1, dtype=np.float32)

    output = pipeline.enhance_array(noisy, 48_000)

    np.testing.assert_array_equal(dl.inputs[0], noisy)
    np.testing.assert_array_equal(output, noisy)


def test_reset_resets_injected_branches_and_diagnostics(method2_config):
    dsp = FakeDSP()
    dl = FakeDL()
    pipeline = HybridMethod2Pipeline(method2_config, dsp, dl)
    pipeline.enhance_array(np.full(16, 0.1, dtype=np.float32), 48_000)

    pipeline.reset()

    assert dsp.reset_calls == 1
    assert dl.reset_calls == 1
    assert pipeline.last_statistics is None


@pytest.mark.parametrize("sample_rate", [16_000, 47_999, 48_001])
def test_wrong_sample_rate_is_rejected_before_branch_calls(method2_config, sample_rate):
    dsp = FakeDSP()
    dl = FakeDL()
    pipeline = HybridMethod2Pipeline(method2_config, dsp, dl)

    with pytest.raises(ValueError, match="48000"):
        pipeline.enhance_array(np.zeros(8, dtype=np.float32), sample_rate)

    assert dsp.calls == 0
    assert dl.calls == 0


@pytest.mark.parametrize("sample_rate", [48_000.0, True, "48000"])
def test_noninteger_sample_rate_is_rejected(method2_config, sample_rate):
    pipeline = HybridMethod2Pipeline(method2_config, FakeDSP(), FakeDL())
    with pytest.raises(TypeError, match="integer"):
        pipeline.enhance_array(np.zeros(8, dtype=np.float32), sample_rate)


def test_invalid_input_is_rejected_before_branch_calls(method2_config):
    dsp = FakeDSP()
    dl = FakeDL()
    pipeline = HybridMethod2Pipeline(method2_config, dsp, dl)

    with pytest.raises(ValueError, match="finite"):
        pipeline.enhance_array(np.array([np.inf], dtype=np.float32), 48_000)

    assert dsp.calls == 0
    assert dl.calls == 0


def test_frame_api_explicitly_rejects_streaming(method2_config):
    pipeline = HybridMethod2Pipeline(method2_config, FakeDSP(), FakeDL())
    with pytest.raises(NotImplementedError, match="whole-array/offline"):
        pipeline.process(np.zeros(480, dtype=np.float32))


def test_constructor_requires_array_enhancer_interfaces(method2_config):
    with pytest.raises(TypeError, match="process_array"):
        HybridMethod2Pipeline(method2_config, object(), FakeDL())
    with pytest.raises(TypeError, match="enhance_array"):
        HybridMethod2Pipeline(method2_config, FakeDSP(), object())


def test_official_evaluator_builds_method2_without_eager_model_load():
    settings = load_settings("config/default.yaml")
    pipeline = _build_pipeline("method2", settings)

    assert isinstance(pipeline, HybridMethod2Pipeline)
    assert pipeline.last_statistics is None
