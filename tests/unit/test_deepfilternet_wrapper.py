"""Unit tests for the optional-dependency-free DeepFilterNet boundary."""

from __future__ import annotations

import gc
import sys
from dataclasses import replace
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

from senhance.config.settings import AppSettings
from senhance.pipeline.dl import deepfilternet_wrapper as wrapper
from senhance.pipeline.dl.deepfilternet_wrapper import (
    DeepFilterNetModelInfo,
    DeepFilterNetPipeline,
)


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
    def __init__(
        self,
        transform: Callable[[np.ndarray], object] | None = None,
        model_info: DeepFilterNetModelInfo | object | None = None,
    ) -> None:
        self._transform = transform or (lambda audio: audio * np.float32(0.5))
        self._model_info = _model_info() if model_info is None else model_info
        self.calls: list[np.ndarray] = []

    @property
    def model_info(self):
        return self._model_info

    def enhance(self, audio: np.ndarray):
        self.calls.append(audio.copy())
        return self._transform(audio)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_enhance_array_accepts_real_float_and_normalizes_to_float32(dtype):
    source = np.linspace(-0.5, 0.5, 20, dtype=dtype)[::2]
    original = source.copy()
    fake = FakeEnhancer()
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=fake)

    enhanced = pipeline.enhance_array(source, 48_000)

    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(enhanced, source.astype(np.float32) * 0.5)
    assert enhanced.dtype == np.float32
    assert enhanced.ndim == 1
    assert enhanced.flags.c_contiguous
    assert fake.calls[0].dtype == np.float32
    assert fake.calls[0].flags.c_contiguous
    assert not np.shares_memory(source, fake.calls[0])


def test_one_backend_instance_handles_repeated_arrays_deterministically():
    fake = FakeEnhancer()
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=fake)
    audio = np.linspace(-0.25, 0.25, 101, dtype=np.float32)

    first = pipeline.enhance_array(audio, 48_000)
    second = pipeline.enhance_array(audio, 48_000)

    np.testing.assert_array_equal(first, second)
    assert len(fake.calls) == 2


def test_empty_array_preserves_contract_without_backend_call():
    fake = FakeEnhancer()
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=fake)

    enhanced = pipeline.enhance_array(np.empty(0, dtype=np.float64), 48_000)

    assert enhanced.shape == (0,)
    assert enhanced.dtype == np.float32
    assert fake.calls == []


def test_lazy_production_factory_receives_selected_model_and_device_once(monkeypatch):
    fake = FakeEnhancer()
    calls = []

    def factory(model_name, device):
        calls.append((model_name, device))
        return fake

    monkeypatch.setattr(wrapper, "_create_deepfilternet_enhancer", factory)
    settings = AppSettings()
    settings.deep_learning.model_name = "DeepFilterNet2"
    settings.deep_learning.device = "cpu"
    pipeline = DeepFilterNetPipeline(settings)
    assert calls == []

    assert pipeline.model_info.loaded_model_name == "fake-model"
    pipeline.enhance_array(np.ones(8, dtype=np.float32), 48_000)
    pipeline._load_model()

    assert calls == [("DeepFilterNet2", "cpu")]


def test_enabled_setting_eagerly_loads_once(monkeypatch):
    fake = FakeEnhancer()
    calls = []
    monkeypatch.setattr(
        wrapper,
        "_create_deepfilternet_enhancer",
        lambda model_name, device: calls.append((model_name, device)) or fake,
    )
    settings = AppSettings()
    settings.deep_learning.enabled = True

    pipeline = DeepFilterNetPipeline(settings)
    pipeline.enhance_array(np.ones(8, dtype=np.float32), 48_000)

    assert calls == [("DeepFilterNet3", "cpu")]


def test_production_backend_slot_enforces_one_live_instance(monkeypatch):
    class Backend:
        pass

    monkeypatch.setattr(wrapper, "_ACTIVE_DF_BACKEND", None)
    monkeypatch.setattr(wrapper, "_DF_BACKEND_LOADING", False)
    backend = Backend()
    wrapper._reserve_production_backend()
    wrapper._register_production_backend(backend)

    with pytest.raises(RuntimeError, match="Only one production"):
        wrapper._reserve_production_backend()

    del backend
    gc.collect()
    wrapper._reserve_production_backend()
    wrapper._release_production_backend_reservation()


@pytest.mark.parametrize(
    ("device", "normalized"),
    [
        ("cpu", "cpu"),
        (" CPU ", "cpu"),
        ("cuda", "cuda"),
        ("CUDA:0", "cuda:0"),
        ("cuda:0002", "cuda:2"),
        ("cuda:256", "cuda:256"),
    ],
)
def test_device_spec_is_normalized_without_torch_index_wrapping(device, normalized):
    assert wrapper._normalize_device_spec(device) == normalized


@pytest.mark.parametrize(
    "device",
    ["", "auto", "mps", "cpu:0", "cuda:", "cuda:-1", "cuda:1.5"],
)
def test_invalid_device_spec_is_rejected_before_model_loading(device):
    with pytest.raises(ValueError, match="must be 'cpu', 'cuda'"):
        wrapper._normalize_device_spec(device)


def test_injected_enhancer_bypasses_production_factory_when_enabled(monkeypatch):
    monkeypatch.setattr(
        wrapper,
        "_create_deepfilternet_enhancer",
        lambda *_: pytest.fail("production factory must not be called"),
    )
    settings = AppSettings()
    settings.deep_learning.enabled = True

    pipeline = DeepFilterNetPipeline(settings, enhancer=FakeEnhancer())

    assert pipeline.model_info == _model_info()


def test_injected_enhancer_needs_no_torch_or_deepfilternet_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "df", None)
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    enhanced = pipeline.enhance_array(np.ones(8, dtype=np.float32), 48_000)

    np.testing.assert_array_equal(enhanced, np.full(8, 0.5, dtype=np.float32))


def test_model_info_exposes_model_owned_geometry_not_dsp_geometry():
    settings = AppSettings()
    settings.dsp.frame_size_ms = 10
    fake = FakeEnhancer()

    info = DeepFilterNetPipeline(settings, enhancer=fake).model_info

    assert info.sample_rate == 48_000
    assert info.fft_size == 960
    assert info.hop_size == 480
    assert info.inference_pad is True
    assert info.output_delay_compensated is True
    assert info.delay_compensation_samples == 480
    assert info.model_pad_mode == "output"


@pytest.mark.parametrize(
    "audio",
    [
        np.array(0.0, dtype=np.float32),
        np.zeros((1, 8), dtype=np.float32),
        np.zeros((8, 1), dtype=np.float32),
    ],
)
def test_rejects_non_mono_shape(audio):
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(ValueError, match="one-dimensional mono"):
        pipeline.enhance_array(audio, 48_000)


@pytest.mark.parametrize(
    "audio",
    [
        np.ones(8, dtype=np.int16),
        np.ones(8, dtype=np.complex64),
        np.array([object()], dtype=object),
    ],
)
def test_rejects_non_real_float_input(audio):
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(TypeError, match="real floating dtype"):
        pipeline.enhance_array(audio, 48_000)


def test_rejects_non_array_input():
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(TypeError, match="numpy.ndarray"):
        pipeline.enhance_array([0.0, 1.0], 48_000)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_input(bad_value):
    audio = np.zeros(8, dtype=np.float32)
    audio[3] = bad_value
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(ValueError, match="NaN or infinite"):
        pipeline.enhance_array(audio, 48_000)


def test_rejects_finite_float64_values_that_overflow_float32():
    audio = np.array([np.finfo(np.float64).max], dtype=np.float64)
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(ValueError, match="outside the float32 finite range"):
        pipeline.enhance_array(audio, 48_000)


@pytest.mark.parametrize("sample_rate", [16_000, 44_100, 48_001])
def test_rejects_wrong_sample_rate(sample_rate):
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(ValueError, match="must be 48000 Hz"):
        pipeline.enhance_array(np.ones(8, dtype=np.float32), sample_rate)


@pytest.mark.parametrize("sample_rate", [48_000.0, "48000", True])
def test_rejects_noninteger_sample_rate(sample_rate):
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(TypeError, match="sample_rate must be an integer"):
        pipeline.enhance_array(np.ones(8, dtype=np.float32), sample_rate)


@pytest.mark.parametrize(
    ("transform", "exception", "message"),
    [
        (lambda audio: audio.tolist(), TypeError, "numpy.ndarray"),
        (lambda audio: audio[None, :], ValueError, "one-dimensional mono"),
        (lambda audio: np.ones(audio.size, dtype=np.int16), TypeError, "floating dtype"),
        (lambda audio: audio[:-1], ValueError, "length must equal"),
        (
            lambda audio: np.full(audio.shape, np.nan, dtype=np.float32),
            ValueError,
            "NaN or infinite",
        ),
        (
            lambda audio: np.full(
                audio.shape, np.finfo(np.float64).max, dtype=np.float64
            ),
            ValueError,
            "outside the float32 finite range",
        ),
    ],
)
def test_rejects_backend_contract_violations(transform, exception, message):
    pipeline = DeepFilterNetPipeline(
        AppSettings(), enhancer=FakeEnhancer(transform=transform)
    )

    with pytest.raises(exception, match=message):
        pipeline.enhance_array(np.ones(8, dtype=np.float32), 48_000)


def test_backend_cannot_mutate_callers_input():
    def mutate(audio):
        audio[:] = 0.0
        return audio

    source = np.ones(8, dtype=np.float32)
    pipeline = DeepFilterNetPipeline(
        AppSettings(), enhancer=FakeEnhancer(transform=mutate)
    )

    enhanced = pipeline.enhance_array(source, 48_000)

    np.testing.assert_array_equal(source, np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(enhanced, np.zeros(8, dtype=np.float32))


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (replace(_model_info(), sample_rate=16_000), "must be 48000 Hz"),
        (replace(_model_info(), fft_size=0), "Invalid DeepFilterNet geometry"),
        (replace(_model_info(), inference_pad=False), "must use pad=True"),
        (
            replace(_model_info(), delay_compensation_samples=0),
            "does not match fft_size - hop_size",
        ),
    ],
)
def test_rejects_invalid_backend_metadata(info, message):
    with pytest.raises(ValueError, match=message):
        DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer(model_info=info))


def test_rejects_backend_metadata_of_wrong_type():
    with pytest.raises(TypeError, match="DeepFilterNetModelInfo"):
        DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer(model_info=object()))


def test_rejects_project_model_sample_rate_mismatch():
    settings = AppSettings()
    settings.audio.sample_rate = 16_000

    with pytest.raises(ValueError, match="Project and DeepFilterNet sample rates differ"):
        DeepFilterNetPipeline(settings, enhancer=FakeEnhancer())


def test_process_remains_explicitly_non_streaming():
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=FakeEnhancer())

    with pytest.raises(NotImplementedError, match="whole-array/offline only"):
        pipeline.process(np.zeros(480, dtype=np.float32))


def test_process_file_delegates_to_array_boundary_and_writes_float_wav(tmp_path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "nested" / "output.wav"
    audio = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    sf.write(input_path, audio, 48_000, subtype="FLOAT")
    fake = FakeEnhancer()
    pipeline = DeepFilterNetPipeline(AppSettings(), enhancer=fake)

    pipeline.process_file(input_path, output_path)

    enhanced, sample_rate = sf.read(output_path, dtype="float32")
    np.testing.assert_allclose(enhanced, audio * 0.5)
    assert sample_rate == 48_000
    assert sf.info(output_path).subtype == "FLOAT"
    assert len(fake.calls) == 1
