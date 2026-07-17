"""Signed causal alignment and diagnostic-delay tests."""

from __future__ import annotations

import numpy as np
import pytest

from senhance.pipeline.hybrid.method3 import alignment
from senhance.pipeline.hybrid.method3.alignment import (
    FixedDelayAligner,
    estimate_delay_cross_correlation,
    measure_impulse_delay,
)


def _impulse_pair(delay: int, length: int = 64, reference_index: int = 20):
    reference = np.zeros(length, dtype=np.float32)
    candidate = np.zeros(length, dtype=np.float32)
    reference[reference_index] = 1.0
    candidate[reference_index + delay] = 0.4
    return reference, candidate


@pytest.mark.parametrize("delay", [-9, -1, 0, 1, 9])
def test_impulse_diagnostics_follow_candidate_minus_reference_sign_convention(delay):
    reference, candidate = _impulse_pair(delay)

    impulse_delay = measure_impulse_delay(reference, candidate)
    estimate = estimate_delay_cross_correlation(reference, candidate, max_abs_delay=12)

    assert impulse_delay == delay
    assert estimate.delay_samples == delay
    assert abs(estimate.normalized_peak) > 0.95
    assert estimate.peak_at_search_boundary is False
    assert estimate.ambiguous_peak is False


@pytest.mark.parametrize("delay", [-7, 0, 7])
def test_fixed_delay_aligner_delays_faster_path_and_aligns_impulses(delay):
    reference, candidate = _impulse_pair(delay)
    reference_before = reference.copy()
    candidate_before = candidate.copy()

    aligned = FixedDelayAligner(delay).align(reference, candidate)

    expected_index = 20 + max(delay, 0)
    assert np.argmax(np.abs(aligned.reference)) == expected_index
    assert np.argmax(np.abs(aligned.candidate)) == expected_index
    assert aligned.reference.shape == reference.shape
    assert aligned.candidate.shape == candidate.shape
    assert aligned.reference.dtype == np.float32
    assert aligned.candidate.dtype == np.float32
    assert aligned.reference.flags.c_contiguous
    assert aligned.candidate.flags.c_contiguous
    assert aligned.delay_samples == delay
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(candidate, candidate_before)


def test_positive_delay_prepends_reference_zeros_and_truncates_reference_tail():
    reference = np.arange(6, dtype=np.float32)
    candidate = np.arange(10, 16, dtype=np.float32)

    aligned = FixedDelayAligner(2).align(reference, candidate)

    np.testing.assert_array_equal(aligned.reference, [0, 0, 0, 1, 2, 3])
    np.testing.assert_array_equal(aligned.candidate, candidate)


def test_negative_delay_prepends_candidate_zeros_and_truncates_candidate_tail():
    reference = np.arange(6, dtype=np.float32)
    candidate = np.arange(10, 16, dtype=np.float32)

    aligned = FixedDelayAligner(-2).align(reference, candidate)

    np.testing.assert_array_equal(aligned.reference, reference)
    np.testing.assert_array_equal(aligned.candidate, [0, 0, 10, 11, 12, 13])


@pytest.mark.parametrize("delay", [-7, -6, 6, 7])
def test_delay_at_or_beyond_length_zeroes_only_the_faster_path(delay):
    reference = np.arange(1, 7, dtype=np.float32)
    candidate = np.arange(11, 17, dtype=np.float32)

    aligned = FixedDelayAligner(delay).align(reference, candidate)

    if delay > 0:
        np.testing.assert_array_equal(aligned.reference, np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(aligned.candidate, candidate)
    else:
        np.testing.assert_array_equal(aligned.reference, reference)
        np.testing.assert_array_equal(aligned.candidate, np.zeros(6, dtype=np.float32))


def test_zero_delay_returns_independent_copies():
    reference = np.linspace(-1, 1, 8, dtype=np.float64)
    candidate = np.linspace(1, -1, 8, dtype=np.float64)

    aligned = FixedDelayAligner(0).align(reference, candidate)
    aligned.reference[0] = 0.0
    aligned.candidate[0] = 0.0

    assert reference[0] == -1.0
    assert candidate[0] == 1.0


def test_fixed_production_alignment_never_calls_diagnostic(monkeypatch):
    monkeypatch.setattr(
        alignment,
        "estimate_delay_cross_correlation",
        lambda *_args, **_kwargs: pytest.fail("diagnostic must not run in production"),
    )

    aligned = FixedDelayAligner(0).align(
        np.ones(8, dtype=np.float32),
        np.ones(8, dtype=np.float32),
    )

    np.testing.assert_array_equal(aligned.reference, aligned.candidate)


def test_cross_correlation_search_bound_is_reported():
    reference, candidate = _impulse_pair(5)

    estimate = estimate_delay_cross_correlation(reference, candidate, max_abs_delay=5)

    assert estimate.delay_samples == 5
    assert estimate.peak_at_search_boundary is True


def test_cross_correlation_handles_amplitude_and_polarity_changes():
    rng = np.random.default_rng(429)
    reference = rng.normal(size=256).astype(np.float32)
    candidate = np.zeros_like(reference)
    candidate[11:] = -0.25 * reference[:-11]

    estimate = estimate_delay_cross_correlation(reference, candidate, max_abs_delay=20)

    assert estimate.delay_samples == 11
    assert estimate.normalized_peak < -0.9


@pytest.mark.parametrize(
    "reference, candidate, exception, message",
    [
        (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            ValueError,
            "non-empty",
        ),
        (
            np.zeros(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
            ValueError,
            "non-silent",
        ),
        (
            np.ones(16, dtype=np.float32),
            np.ones(16, dtype=np.float32),
            ValueError,
            "non-constant",
        ),
    ],
)
def test_cross_correlation_rejects_undefined_diagnostics(reference, candidate, exception, message):
    with pytest.raises(exception, match=message):
        estimate_delay_cross_correlation(reference, candidate)


def test_impulse_diagnostic_rejects_ambiguous_peak():
    reference = np.zeros(16, dtype=np.float32)
    candidate = np.zeros(16, dtype=np.float32)
    reference[[3, 4]] = 1.0
    candidate[5] = 1.0

    with pytest.raises(ValueError, match="one unique reference peak"):
        measure_impulse_delay(reference, candidate)


@pytest.mark.parametrize("bad_limit", [-1, 1.5, True])
def test_cross_correlation_rejects_invalid_search_bound(bad_limit):
    reference, candidate = _impulse_pair(0)

    with pytest.raises((TypeError, ValueError), match="max_abs_delay"):
        estimate_delay_cross_correlation(reference, candidate, max_abs_delay=bad_limit)


@pytest.mark.parametrize(
    "reference, candidate, exception, message",
    [
        (np.zeros(4, dtype=np.float32), np.zeros(5, dtype=np.float32), ValueError, "equal"),
        (np.zeros((1, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), ValueError, "mono"),
        (np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (np.zeros(4, dtype=np.complex64), np.zeros(4, dtype=np.float32), TypeError, "floating"),
        (
            np.array([0, np.nan], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_aligner_rejects_invalid_audio(reference, candidate, exception, message):
    with pytest.raises(exception, match=message):
        FixedDelayAligner(0).align(reference, candidate)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_aligner_rejects_noninteger_delay(delay):
    with pytest.raises(TypeError, match="delay_samples must be an integer"):
        FixedDelayAligner(delay)


def test_aligner_rejects_float32_overflow():
    reference = np.array([np.finfo(np.float64).max], dtype=np.float64)
    candidate = np.zeros(1, dtype=np.float64)

    with pytest.raises(ValueError, match="finite.*range"):
        FixedDelayAligner(0).align(reference, candidate)
