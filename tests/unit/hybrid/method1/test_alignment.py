"""Fixed noisy/DL alignment and read-only delay diagnostic tests."""

from __future__ import annotations

import numpy as np
import pytest

from senhance.pipeline.hybrid.method1.alignment import (
    FixedNoisyDLAligner,
    estimate_noisy_dl_delay,
    measure_noisy_dl_impulse_delay,
)


def test_zero_delay_returns_independent_contiguous_float32_copies():
    noisy = np.linspace(-0.5, 0.5, 11, dtype=np.float64)[::2]
    dl = -noisy
    noisy_before = noisy.copy()
    dl_before = dl.copy()

    aligned = FixedNoisyDLAligner(0).align(noisy, dl)

    np.testing.assert_array_equal(aligned.noisy, noisy.astype(np.float32))
    np.testing.assert_array_equal(aligned.dl, dl.astype(np.float32))
    assert aligned.noisy.dtype == np.float32
    assert aligned.dl.dtype == np.float32
    assert aligned.noisy.flags.c_contiguous
    assert aligned.dl.flags.c_contiguous
    assert not np.shares_memory(aligned.noisy, noisy)
    assert not np.shares_memory(aligned.dl, dl)
    np.testing.assert_array_equal(noisy, noisy_before)
    np.testing.assert_array_equal(dl, dl_before)


@pytest.mark.parametrize("delay", [1, 3, 9])
def test_positive_delay_delays_only_faster_noisy_path(delay):
    noisy = np.arange(8, dtype=np.float32) + 1
    dl = -noisy

    aligned = FixedNoisyDLAligner(delay).align(noisy, dl)

    expected_noisy = np.zeros_like(noisy)
    if delay < noisy.size:
        expected_noisy[delay:] = noisy[:-delay]
    np.testing.assert_array_equal(aligned.noisy, expected_noisy)
    np.testing.assert_array_equal(aligned.dl, dl)
    assert aligned.dl_minus_noisy_delay_samples == delay


@pytest.mark.parametrize("delay", [-1, -3, -9])
def test_negative_delay_delays_only_faster_dl_path(delay):
    noisy = np.arange(8, dtype=np.float32) + 1
    dl = -noisy

    aligned = FixedNoisyDLAligner(delay).align(noisy, dl)

    expected_dl = np.zeros_like(dl)
    amount = -delay
    if amount < dl.size:
        expected_dl[amount:] = dl[:-amount]
    np.testing.assert_array_equal(aligned.noisy, noisy)
    np.testing.assert_array_equal(aligned.dl, expected_dl)


@pytest.mark.parametrize("delay", [-7, 0, 7])
def test_impulse_delay_uses_dl_index_minus_noisy_index(delay):
    noisy = np.zeros(64, dtype=np.float32)
    dl = np.zeros(64, dtype=np.float32)
    noisy[20] = 1.0
    dl[20 + delay] = -0.8

    assert measure_noisy_dl_impulse_delay(noisy, dl) == delay


@pytest.mark.parametrize("delay", [-11, -3, 0, 4, 13])
def test_cross_correlation_estimates_known_residual_delay(delay):
    rng = np.random.default_rng(429)
    source = rng.normal(size=256).astype(np.float32)
    noisy = np.zeros(320, dtype=np.float32)
    dl = np.zeros(320, dtype=np.float32)
    noisy_start = 30
    dl_start = noisy_start + delay
    noisy[noisy_start : noisy_start + source.size] = source
    dl[dl_start : dl_start + source.size] = source

    estimate = estimate_noisy_dl_delay(noisy, dl, max_abs_delay=20)

    assert estimate.dl_minus_noisy_delay_samples == delay
    assert estimate.max_abs_delay == 20
    assert estimate.peak_at_search_boundary is False
    assert estimate.ambiguous_peak is False
    assert estimate.normalized_peak > 0.99


def test_cross_correlation_reports_search_boundary():
    rng = np.random.default_rng(430)
    noisy = rng.normal(size=128).astype(np.float32)
    dl = np.concatenate((np.zeros(5, dtype=np.float32), noisy[:-5]))
    estimate = estimate_noisy_dl_delay(noisy, dl, max_abs_delay=5)
    assert estimate.dl_minus_noisy_delay_samples == 5
    assert estimate.peak_at_search_boundary is True


@pytest.mark.parametrize(
    "noisy,dl,exception,message",
    [
        ([0.0], np.zeros(1, dtype=np.float32), TypeError, "numpy.ndarray"),
        (np.zeros((1, 2), dtype=np.float32), np.zeros(2, dtype=np.float32), ValueError, "mono"),
        (np.zeros(2, dtype=np.int16), np.zeros(2, dtype=np.float32), TypeError, "floating"),
        (np.zeros(2, dtype=np.complex64), np.zeros(2, dtype=np.float32), TypeError, "floating"),
        (np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32), ValueError, "equal lengths"),
        (
            np.array([np.nan], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            ValueError,
            "finite",
        ),
    ],
)
def test_aligner_rejects_invalid_array_contract(noisy, dl, exception, message):
    with pytest.raises(exception, match=message):
        FixedNoisyDLAligner(0).align(noisy, dl)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_aligner_rejects_non_integer_delay(delay):
    with pytest.raises(TypeError, match="integer"):
        FixedNoisyDLAligner(delay)


def test_delay_diagnostics_reject_empty_silent_constant_and_ambiguous_inputs():
    empty = np.zeros(0, dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        estimate_noisy_dl_delay(empty, empty)

    silence = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="non-silent"):
        estimate_noisy_dl_delay(silence, silence)

    constant = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="non-constant"):
        estimate_noisy_dl_delay(constant, constant)

    two_peaks = np.zeros(8, dtype=np.float32)
    two_peaks[[2, 5]] = 1.0
    one_peak = np.zeros(8, dtype=np.float32)
    one_peak[3] = 1.0
    with pytest.raises(ValueError, match="one unique noisy peak"):
        measure_noisy_dl_impulse_delay(two_peaks, one_peak)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_cross_correlation_rejects_invalid_search_limit(limit):
    noisy = np.zeros(16, dtype=np.float32)
    noisy[4] = 1.0
    dl = noisy.copy()
    exception = ValueError if limit == -1 else TypeError
    with pytest.raises(exception):
        estimate_noisy_dl_delay(noisy, dl, max_abs_delay=limit)
