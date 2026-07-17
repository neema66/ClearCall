"""Exact-length paired-STFT, WOLA, flush, and reset tests."""

from __future__ import annotations

import numpy as np
import pytest

from senhance.pipeline.hybrid.method3.alignment import FixedDelayAligner
from senhance.pipeline.hybrid.method3.config import (
    HybridAlignmentConfig,
    HybridConfig,
)
from senhance.pipeline.hybrid.method3.paired_stft import (
    PairedSTFTCore,
    select_candidate_spectrum,
    select_reference_spectrum,
)


IDENTITY_LENGTHS = [0, 1, 479, 480, 481, 959, 960, 961, 1577, 10_000]


def _core(delay_samples: int = 0) -> PairedSTFTCore:
    return PairedSTFTCore(
        HybridConfig(alignment=HybridAlignmentConfig(delay_samples=delay_samples))
    )


@pytest.mark.parametrize("length", IDENTITY_LENGTHS)
@pytest.mark.parametrize(
    "selector, selected",
    [
        (select_reference_spectrum, "reference"),
        (select_candidate_spectrum, "candidate"),
    ],
)
def test_endpoint_identity_reconstructs_arbitrary_lengths(length, selector, selected):
    rng = np.random.default_rng(429 + length)
    reference = rng.normal(0.0, 0.1, length).astype(np.float32)
    candidate = rng.normal(0.0, 0.1, length).astype(np.float32)
    expected = reference if selected == "reference" else candidate

    output = _core().process_array(
        reference,
        candidate,
        selector,
        sample_rate=48_000,
    )

    assert output.shape == (length,)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("length", [1, 479, 480, 481, 959, 960, 961, 1437])
def test_final_sample_impulse_survives_padding_flush_and_trim(length):
    signal = np.zeros(length, dtype=np.float32)
    signal[-1] = 0.75

    output = _core().process_array(
        signal,
        np.zeros_like(signal),
        select_reference_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, signal, atol=2e-6, rtol=0.0)
    assert int(np.argmax(np.abs(output))) == length - 1


def test_first_sample_impulse_reconstructs_at_first_sample():
    signal = np.zeros(961, dtype=np.float32)
    signal[0] = -0.8

    output = _core().process_array(
        signal,
        signal,
        select_candidate_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, signal, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("length", [0, 480, 481, 1440])
def test_silence_is_exact_zero(length):
    silence = np.zeros(length, dtype=np.float32)

    output = _core().process_array(
        silence,
        silence,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_array_equal(output, silence)


def test_clipped_input_is_not_normalized_or_clipped_again():
    reference = np.tile(np.array([-1.0, 1.0], dtype=np.float32), 731)
    candidate = np.zeros_like(reference)

    output = _core().process_array(
        reference,
        candidate,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, reference, atol=2e-6, rtol=0.0)
    assert np.max(output) == pytest.approx(1.0, abs=2e-6)
    assert np.min(output) == pytest.approx(-1.0, abs=2e-6)


@pytest.mark.parametrize("delay", [-5, 0, 5])
@pytest.mark.parametrize(
    "selector, selected",
    [
        (select_reference_spectrum, "reference"),
        (select_candidate_spectrum, "candidate"),
    ],
)
def test_core_reconstructs_configured_causally_aligned_endpoint(delay, selector, selected):
    reference = np.zeros(1001, dtype=np.float32)
    candidate = np.zeros(1001, dtype=np.float32)
    reference[100] = 1.0
    candidate[100 + delay] = 0.5
    aligned = FixedDelayAligner(delay).align(reference, candidate)
    expected = aligned.reference if selected == "reference" else aligned.candidate

    output = _core(delay).process_array(
        reference,
        candidate,
        selector,
        sample_rate=48_000,
    )

    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=0.0)


def test_paired_frames_are_synchronized_read_only_and_have_481_bins():
    class Recorder:
        def __init__(self):
            self.frames = []
            self.reset_calls = 0

        def reset(self):
            self.frames.clear()
            self.reset_calls += 1

        def __call__(self, frame):
            self.frames.append(frame)
            return frame.reference_spectrum

    recorder = Recorder()
    reference = np.arange(481, dtype=np.float32) / 1000.0
    candidate = -reference

    _core().process_array(reference, candidate, recorder, sample_rate=48_000)

    assert recorder.reset_calls == 1
    assert [frame.index for frame in recorder.frames] == [0, 1, 2]
    assert all(frame.reference_spectrum.shape == (481,) for frame in recorder.frames)
    assert all(frame.candidate_spectrum.shape == (481,) for frame in recorder.frames)
    assert all(not frame.reference_spectrum.flags.writeable for frame in recorder.frames)
    assert all(not frame.candidate_spectrum.flags.writeable for frame in recorder.frames)


def test_one_startup_hop_is_removed_from_whole_array_output():
    core = _core()
    first_hop = np.zeros(480, dtype=np.float32)
    first_hop[7] = 1.0
    zero = np.zeros(480, dtype=np.float32)

    raw_first = core.process_aligned_hop(
        first_hop,
        first_hop,
        select_reference_spectrum,
    )
    raw_flush = core.process_aligned_hop(zero, zero, select_reference_spectrum)
    raw = np.concatenate((raw_first, raw_flush))
    compensated = _core().process_array(
        first_hop,
        first_hop,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    assert np.argmax(np.abs(raw)) == 480 + 7
    assert np.argmax(np.abs(compensated)) == 7
    assert core.framing_delay_samples == 480


def test_repeated_whole_array_processing_is_bit_identical_and_resets_processor():
    class StatefulSelector:
        def __init__(self):
            self.calls = 0
            self.reset_calls = 0

        def reset(self):
            self.calls = 0
            self.reset_calls += 1

        def __call__(self, frame):
            self.calls += 1
            return frame.candidate_spectrum

    rng = np.random.default_rng(88)
    reference = rng.normal(size=1703).astype(np.float32)
    candidate = rng.normal(size=1703).astype(np.float32)
    processor = StatefulSelector()
    core = _core()

    first = core.process_array(reference, candidate, processor, sample_rate=48_000)
    second = core.process_array(reference, candidate, processor, sample_rate=48_000)

    np.testing.assert_array_equal(first, second)
    assert processor.reset_calls == 2


def test_processing_clip_b_after_clip_a_matches_fresh_core():
    rng = np.random.default_rng(89)
    clip_a = rng.normal(size=1501).astype(np.float32)
    clip_b = rng.normal(size=997).astype(np.float32)
    reused = _core()
    reused.process_array(
        clip_a,
        clip_a,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    after_a = reused.process_array(
        clip_b,
        clip_b,
        select_reference_spectrum,
        sample_rate=48_000,
    )
    fresh = _core().process_array(
        clip_b,
        clip_b,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_array_equal(after_a, fresh)


def test_manual_reset_clears_both_input_histories_and_synthesis_overlap():
    core = _core()
    hop = np.linspace(-0.2, 0.2, 480, dtype=np.float32)
    core.process_aligned_hop(hop, -hop, select_reference_spectrum)

    core.reset()

    np.testing.assert_array_equal(core._previous_reference, np.zeros(480))
    np.testing.assert_array_equal(core._previous_candidate, np.zeros(480))
    np.testing.assert_array_equal(core._synthesis_overlap, np.zeros(480))
    assert core._frame_index == 0
    reset_output = core.process_aligned_hop(hop, -hop, select_reference_spectrum)
    fresh_output = _core().process_aligned_hop(hop, -hop, select_reference_spectrum)
    np.testing.assert_array_equal(reset_output, fresh_output)


def test_process_array_does_not_mutate_callers_inputs():
    reference = np.linspace(-0.2, 0.2, 777, dtype=np.float64)
    candidate = -reference
    reference_before = reference.copy()
    candidate_before = candidate.copy()

    _core().process_array(
        reference,
        candidate,
        select_reference_spectrum,
        sample_rate=48_000,
    )

    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(candidate, candidate_before)


@pytest.mark.parametrize(
    "processor, exception, message",
    [
        (lambda frame: frame.reference_spectrum.tolist(), TypeError, "numpy.ndarray"),
        (lambda frame: frame.reference_spectrum[:-1], ValueError, "spectrum shape"),
        (
            lambda frame: np.zeros(481, dtype=np.float32),
            TypeError,
            "complex floating",
        ),
        (
            lambda frame: np.full(481, np.nan + 0j, dtype=np.complex128),
            ValueError,
            "finite",
        ),
    ],
)
def test_rejects_invalid_processor_spectrum(processor, exception, message):
    audio = np.zeros(480, dtype=np.float32)

    with pytest.raises(exception, match=message):
        _core().process_array(audio, audio, processor, sample_rate=48_000)


@pytest.mark.parametrize(
    "reference, candidate, exception, message",
    [
        (np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32), ValueError, "equal"),
        (np.zeros((1, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), ValueError, "mono"),
        (np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (np.zeros(4, dtype=np.complex64), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (np.array([np.inf], dtype=np.float32), np.zeros(1, dtype=np.float32), ValueError, "finite"),
    ],
)
def test_rejects_invalid_whole_array_audio(reference, candidate, exception, message):
    with pytest.raises(exception, match=message):
        _core().process_array(
            reference,
            candidate,
            select_reference_spectrum,
            sample_rate=48_000,
        )


@pytest.mark.parametrize("sample_rate", [16_000, 47_999, 48_001])
def test_rejects_wrong_sample_rate(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="must be 48000 Hz"):
        _core().process_array(
            audio,
            audio,
            select_reference_spectrum,
            sample_rate=sample_rate,
        )


@pytest.mark.parametrize("sample_rate", [48_000.0, True, "48000"])
def test_rejects_noninteger_sample_rate(sample_rate):
    audio = np.zeros(4, dtype=np.float32)

    with pytest.raises(TypeError, match="sample_rate must be an integer"):
        _core().process_array(
            audio,
            audio,
            select_reference_spectrum,
            sample_rate=sample_rate,
        )


def test_rejects_wrong_hop_shape_and_dtype():
    core = _core()

    with pytest.raises(ValueError, match=r"shape \(480,\)"):
        core.process_aligned_hop(
            np.zeros(479, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
            select_reference_spectrum,
        )
    with pytest.raises(TypeError, match="floating"):
        core.process_aligned_hop(
            np.zeros(480, dtype=np.int16),
            np.zeros(480, dtype=np.float32),
            select_reference_spectrum,
        )


def test_rejects_noncallable_processor_and_reset_attribute():
    audio = np.zeros(480, dtype=np.float32)
    with pytest.raises(TypeError, match="processor must be callable"):
        _core().process_array(audio, audio, None, sample_rate=48_000)

    class BadReset:
        reset = 1

        def __call__(self, frame):
            return frame.reference_spectrum

    with pytest.raises(TypeError, match="processor.reset must be callable"):
        _core().process_array(audio, audio, BadReset(), sample_rate=48_000)
