"""Method-1-owned paired STFT, WOLA, flush, and reset tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method1.alignment import FixedNoisyDLAligner
from senhance.pipeline.hybrid.method1.config import Method1AlignmentConfig
from senhance.pipeline.hybrid.method1.paired_stft import (
    Method1PairedSTFTCore,
    select_dl_spectrum,
    select_noisy_spectrum,
)


IDENTITY_LENGTHS = [0, 1, 479, 480, 481, 959, 960, 961, 1577, 10_000]


def _core(config, delay=0):
    configured = dataclasses.replace(
        config,
        alignment=Method1AlignmentConfig(delay),
    )
    return Method1PairedSTFTCore(configured)


@pytest.mark.parametrize("length", IDENTITY_LENGTHS)
@pytest.mark.parametrize(
    "selector,selected",
    [(select_noisy_spectrum, "noisy"), (select_dl_spectrum, "dl")],
)
def test_endpoint_identity_reconstructs_arbitrary_lengths(
    method1_config,
    length,
    selector,
    selected,
):
    rng = np.random.default_rng(1000 + length)
    noisy = rng.normal(0.0, 0.1, length).astype(np.float32)
    dl = rng.normal(0.0, 0.1, length).astype(np.float32)
    expected = noisy if selected == "noisy" else dl

    output = _core(method1_config).process_array(
        noisy,
        dl,
        selector,
        sample_rate=48_000,
    )

    assert output.shape == (length,)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("length", [1, 479, 480, 481, 959, 960, 961, 1437])
def test_final_sample_impulse_survives_padding_flush_and_trim(method1_config, length):
    signal = np.zeros(length, dtype=np.float32)
    signal[-1] = 0.75

    output = _core(method1_config).process_array(
        signal,
        np.zeros_like(signal),
        select_noisy_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, signal, atol=2e-6, rtol=0.0)
    assert int(np.argmax(np.abs(output))) == length - 1


def test_first_sample_impulse_reconstructs_without_extra_shift(method1_config):
    signal = np.zeros(961, dtype=np.float32)
    signal[0] = -0.8
    output = _core(method1_config).process_array(
        signal,
        signal,
        select_dl_spectrum,
        sample_rate=48_000,
    )
    np.testing.assert_allclose(output, signal, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("length", [0, 480, 481, 1440])
@pytest.mark.parametrize("selector", [select_noisy_spectrum, select_dl_spectrum])
def test_silence_is_exact_zero(method1_config, length, selector):
    silence = np.zeros(length, dtype=np.float32)
    output = _core(method1_config).process_array(
        silence,
        silence,
        selector,
        sample_rate=48_000,
    )
    np.testing.assert_array_equal(output, silence)


@pytest.mark.parametrize("delay", [-5, 0, 5])
@pytest.mark.parametrize(
    "selector,selected",
    [(select_noisy_spectrum, "noisy"), (select_dl_spectrum, "dl")],
)
def test_core_reconstructs_configured_causally_aligned_path(
    method1_config,
    delay,
    selector,
    selected,
):
    noisy = np.zeros(1001, dtype=np.float32)
    dl = np.zeros(1001, dtype=np.float32)
    noisy[100] = 1.0
    dl[100 + delay] = 0.5
    aligned = FixedNoisyDLAligner(delay).align(noisy, dl)
    expected = aligned.noisy if selected == "noisy" else aligned.dl

    output = _core(method1_config, delay).process_array(
        noisy,
        dl,
        selector,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=0.0)


def test_frames_are_synchronized_read_only_and_reset_once(method1_config):
    class Recorder:
        def __init__(self):
            self.frames = []
            self.reset_calls = 0

        def reset(self):
            self.frames.clear()
            self.reset_calls += 1

        def __call__(self, frame):
            self.frames.append(frame)
            return frame.noisy_spectrum

    recorder = Recorder()
    noisy = np.arange(481, dtype=np.float32) / 1000.0
    dl = -noisy
    _core(method1_config).process_array(noisy, dl, recorder, sample_rate=48_000)

    assert recorder.reset_calls == 1
    assert [frame.index for frame in recorder.frames] == [0, 1, 2]
    assert all(frame.noisy_spectrum.shape == (481,) for frame in recorder.frames)
    assert all(frame.dl_spectrum.shape == (481,) for frame in recorder.frames)
    assert all(not frame.noisy_spectrum.flags.writeable for frame in recorder.frames)
    assert all(not frame.dl_spectrum.flags.writeable for frame in recorder.frames)


def test_repeated_processing_is_bit_identical_and_resets_processor(method1_config):
    class StatefulSelector:
        def __init__(self):
            self.calls = 0
            self.reset_calls = 0

        def reset(self):
            self.calls = 0
            self.reset_calls += 1

        def __call__(self, frame):
            self.calls += 1
            return frame.dl_spectrum

    rng = np.random.default_rng(431)
    noisy = rng.normal(size=1703).astype(np.float32)
    dl = rng.normal(size=1703).astype(np.float32)
    selector = StatefulSelector()
    core = _core(method1_config)

    first = core.process_array(noisy, dl, selector, sample_rate=48_000)
    second = core.process_array(noisy, dl, selector, sample_rate=48_000)

    np.testing.assert_array_equal(first, second)
    assert selector.reset_calls == 2


def test_clip_b_after_clip_a_matches_fresh_core(method1_config):
    rng = np.random.default_rng(432)
    clip_a = rng.normal(size=1501).astype(np.float32)
    clip_b = rng.normal(size=997).astype(np.float32)
    reused = _core(method1_config)
    reused.process_array(clip_a, clip_a, select_noisy_spectrum, sample_rate=48_000)
    after_a = reused.process_array(clip_b, clip_b, select_noisy_spectrum, sample_rate=48_000)
    fresh = _core(method1_config).process_array(
        clip_b,
        clip_b,
        select_noisy_spectrum,
        sample_rate=48_000,
    )
    np.testing.assert_array_equal(after_a, fresh)


def test_manual_reset_clears_all_histories_and_overlap(method1_config):
    core = _core(method1_config)
    hop = np.linspace(-0.2, 0.2, 480, dtype=np.float32)
    core.process_aligned_hop(hop, -hop, select_noisy_spectrum)
    core.reset()

    np.testing.assert_array_equal(core._previous_noisy, np.zeros(480))
    np.testing.assert_array_equal(core._previous_dl, np.zeros(480))
    np.testing.assert_array_equal(core._synthesis_overlap, np.zeros(480))
    assert core._frame_index == 0
    reset_output = core.process_aligned_hop(hop, -hop, select_noisy_spectrum)
    fresh_output = _core(method1_config).process_aligned_hop(hop, -hop, select_noisy_spectrum)
    np.testing.assert_array_equal(reset_output, fresh_output)


def test_process_array_never_mutates_callers_inputs(method1_config):
    noisy = np.linspace(-0.2, 0.2, 777, dtype=np.float64)
    dl = -noisy
    noisy_before = noisy.copy()
    dl_before = dl.copy()

    _core(method1_config).process_array(
        noisy,
        dl,
        select_noisy_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_array_equal(noisy, noisy_before)
    np.testing.assert_array_equal(dl, dl_before)


@pytest.mark.parametrize(
    "processor,exception,message",
    [
        (lambda frame: frame.noisy_spectrum.tolist(), TypeError, "numpy.ndarray"),
        (lambda frame: frame.noisy_spectrum[:-1], ValueError, "shape"),
        (lambda frame: np.zeros(481, dtype=np.float32), TypeError, "complex"),
        (
            lambda frame: np.full(481, np.nan + 0j, dtype=np.complex128),
            ValueError,
            "finite",
        ),
    ],
)
def test_rejects_invalid_processor_spectrum(method1_config, processor, exception, message):
    audio = np.zeros(480, dtype=np.float32)
    with pytest.raises(exception, match=message):
        _core(method1_config).process_array(audio, audio, processor, sample_rate=48_000)


@pytest.mark.parametrize(
    "noisy,dl,exception,message",
    [
        (np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32), ValueError, "equal"),
        (np.zeros((1, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), ValueError, "mono"),
        (np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (np.zeros(4, dtype=np.complex64), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (
            np.array([np.inf], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_rejects_invalid_whole_array_audio(
    method1_config,
    noisy,
    dl,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        _core(method1_config).process_array(
            noisy,
            dl,
            select_noisy_spectrum,
            sample_rate=48_000,
        )


@pytest.mark.parametrize(
    "sample_rate,exception", [(16_000, ValueError), (48_000.0, TypeError), (True, TypeError)]
)
def test_rejects_invalid_sample_rate(method1_config, sample_rate, exception):
    audio = np.zeros(4, dtype=np.float32)
    with pytest.raises(exception, match="sample rate|sample_rate"):
        _core(method1_config).process_array(
            audio,
            audio,
            select_noisy_spectrum,
            sample_rate=sample_rate,
        )


def test_rejects_noncallable_processor_and_noncallable_reset(method1_config):
    audio = np.zeros(480, dtype=np.float32)
    core = _core(method1_config)
    with pytest.raises(TypeError, match="callable"):
        core.process_array(audio, audio, object(), sample_rate=48_000)

    class BadReset:
        reset = 1

        def __call__(self, frame):
            return frame.noisy_spectrum

    with pytest.raises(TypeError, match="reset.*callable"):
        core.process_array(audio, audio, BadReset(), sample_rate=48_000)


@pytest.mark.parametrize(
    "hop,exception,message",
    [
        ([0.0] * 480, TypeError, "numpy.ndarray"),
        (np.zeros(479, dtype=np.float32), ValueError, "shape"),
        (np.zeros((1, 480), dtype=np.float32), ValueError, "shape"),
        (np.zeros(480, dtype=np.int16), TypeError, "floating"),
        (np.full(480, np.nan, dtype=np.float32), ValueError, "finite"),
    ],
)
def test_process_aligned_hop_rejects_invalid_hop(method1_config, hop, exception, message):
    valid = np.zeros(480, dtype=np.float32)
    with pytest.raises(exception, match=message):
        _core(method1_config).process_aligned_hop(hop, valid, select_noisy_spectrum)
