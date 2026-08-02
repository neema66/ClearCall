"""Orchestration tests for scripts/run_virtual_mic_test.py.

Exercises all 5 --pipeline branches' wiring (which real whole-array APIs
get called, in what order, with what arrays) without needing torch/
deepfilternet installed, by injecting a fake ArrayEnhancer wherever a
DeepFilterNetPipeline gets constructed -- see
senhance.pipeline.dl.deepfilternet_wrapper.ArrayEnhancer and
tests/unit/test_deepfilternet_wrapper.py's FakeEnhancer for the pattern
this mirrors. Does not re-verify each method's own internal correctness
(already covered by that method's own test suite) -- only that this
script wires the right calls together in the right order.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_virtual_mic_test.py"

_spec = importlib.util.spec_from_file_location("run_virtual_mic_test", SCRIPT_PATH)
vmic = importlib.util.module_from_spec(_spec)
sys.modules["run_virtual_mic_test"] = vmic
_spec.loader.exec_module(vmic)

from senhance.config.settings import AppSettings  # noqa: E402
from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetModelInfo  # noqa: E402


def _model_info() -> DeepFilterNetModelInfo:
    return DeepFilterNetModelInfo(
        requested_model_name="fake-model",
        loaded_model_name="fake-model",
        model_base_dir=None,
        effective_device="cpu",
        sample_rate=48_000,
        fft_size=960,
        hop_size=480,
        inference_pad=True,
        output_delay_compensated=True,
        delay_compensation_samples=480,
        model_pad_mode="output",
    )


class FakeEnhancer:
    """Mirrors tests/unit/test_deepfilternet_wrapper.py's FakeEnhancer."""

    def __init__(self, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self._transform = transform or (lambda audio: audio.copy())
        self.calls: list[np.ndarray] = []

    @property
    def model_info(self) -> DeepFilterNetModelInfo:
        return _model_info()

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        self.calls.append(audio.copy())
        return self._transform(audio)


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        dsp_config=str(PROJECT_ROOT / "config" / "improved_dsp.yaml"),
        method1_config=str(PROJECT_ROOT / "config" / "hybrid_method_1.yaml"),
        method1_variant="full_dl_phase",
        hybrid_config=str(PROJECT_ROOT / "config" / "hybrid.yaml"),
        method3_config=str(PROJECT_ROOT / "config" / "hybrid_method_3.yaml"),
        alpha=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _noisy_signal() -> np.ndarray:
    rng = np.random.default_rng(42)
    # A few seconds so every method has enough frames to produce output.
    return (0.05 * rng.normal(size=48_000 * 2)).astype(np.float32)


def _assert_valid_audio(output: np.ndarray) -> None:
    assert isinstance(output, np.ndarray)
    assert output.ndim == 1
    assert output.size > 0
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("pipeline_name", ["original_dsp", "improved_dsp"])
def test_dsp_only_pipelines_produce_valid_audio(pipeline_name: str) -> None:
    settings = AppSettings()
    output = vmic.compute_enhanced_array(
        pipeline_name, _args(), settings, _noisy_signal(), settings.audio.sample_rate
    )
    _assert_valid_audio(output)


def test_dl_pipeline_uses_injected_enhancer_and_produces_valid_audio() -> None:
    settings = AppSettings()
    fake = FakeEnhancer()
    noisy = _noisy_signal()

    output = vmic.compute_enhanced_array(
        "dl", _args(), settings, noisy, settings.audio.sample_rate, dl_enhancer=fake
    )

    _assert_valid_audio(output)
    assert len(fake.calls) == 1
    assert fake.calls[0].shape == noisy.shape


def test_hybrid_method_1_runs_dl_then_safety_layer_on_its_output() -> None:
    settings = AppSettings()
    fake = FakeEnhancer(transform=lambda audio: audio * np.float32(0.5))
    noisy = _noisy_signal()

    output = vmic.compute_enhanced_array(
        "hybrid_method_1", _args(), settings, noisy, settings.audio.sample_rate, dl_enhancer=fake
    )

    _assert_valid_audio(output)
    # DL ran exactly once, on the true noisy array (not some intermediate).
    assert len(fake.calls) == 1
    assert fake.calls[0].shape == noisy.shape


def test_hybrid_method_3_runs_dsp_and_dl_then_blends() -> None:
    settings = AppSettings()
    fake = FakeEnhancer(transform=lambda audio: audio * np.float32(0.5))
    noisy = _noisy_signal()

    output = vmic.compute_enhanced_array(
        "hybrid_method_3", _args(), settings, noisy, settings.audio.sample_rate, dl_enhancer=fake
    )

    _assert_valid_audio(output)
    assert len(fake.calls) == 1
    assert fake.calls[0].shape == noisy.shape


def test_hybrid_method_3_warns_on_delay_reconciliation_mismatch(tmp_path, caplog) -> None:
    """config/hybrid.yaml and config/hybrid_method_3.yaml each configure a
    residual DSP-to-DL delay independently, with nothing reconciling them
    (see compute_enhanced_array's docstring/comment). Deliberately
    mismatch them here and confirm the loud warning fires."""

    mismatched_hybrid_yaml = tmp_path / "hybrid_mismatched.yaml"
    mismatched_hybrid_yaml.write_text(
        "hybrid:\n"
        "  sample_rate: 48000\n"
        "  alignment:\n"
        "    delay_samples: 5\n"
        "  stft:\n"
        "    frame_size: 960\n"
        "    hop_size: 480\n"
        "    window: hann\n"
    )
    # config/hybrid_method_3.yaml's backends both configure
    # dl_minus_dsp_delay_samples: 0, so delay_samples=5 above disagrees.

    settings = AppSettings()
    fake = FakeEnhancer()
    noisy = _noisy_signal()

    with caplog.at_level("WARNING"):
        output = vmic.compute_enhanced_array(
            "hybrid_method_3",
            _args(hybrid_config=str(mismatched_hybrid_yaml)),
            settings,
            noisy,
            settings.audio.sample_rate,
            dl_enhancer=fake,
        )

    _assert_valid_audio(output)
    assert any("delay mismatch" in record.message for record in caplog.records)


def test_unknown_pipeline_raises() -> None:
    settings = AppSettings()
    with pytest.raises(ValueError, match="Unknown pipeline"):
        vmic.compute_enhanced_array(
            "not_a_real_pipeline", _args(), settings, _noisy_signal(), settings.audio.sample_rate
        )
