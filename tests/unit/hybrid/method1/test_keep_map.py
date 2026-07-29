"""Raw keep-map numerical safety and diagnostics tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from senhance.pipeline.hybrid.method1.config import Method1KeepMapConfig
from senhance.pipeline.hybrid.method1.keep_map import estimate_raw_keep_map


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_identical_spectra_produce_exact_all_one_map(method1_config, dtype):
    spectrum = np.array(
        [0.0, 1.0e-12, 1.0e-7, 0.25 + 0.5j, -3.0j],
        dtype=dtype,
    )

    result = estimate_raw_keep_map(spectrum, spectrum.copy(), method1_config.keep_map)

    np.testing.assert_array_equal(result.gain, np.ones(spectrum.size))
    assert result.gain.dtype == np.float64
    assert result.gain.flags.c_contiguous
    assert result.bin_count == spectrum.size
    assert result.low_energy_neutral_bin_count == int(
        np.count_nonzero(np.abs(spectrum) <= method1_config.keep_map.low_energy_threshold)
    )
    assert result.clipped_above_one_bin_count == 0
    assert result.zero_gain_bin_count == 0


def test_controlled_ratios_are_guarded_clamped_and_counted(method1_config):
    noisy = np.array([2.0, 1.0, 1.0, 1.0e-9, 0.0], dtype=np.complex128)
    dl = np.array([1.0, 2.0, 0.0, 0.0, 1.0], dtype=np.complex128)

    result = estimate_raw_keep_map(noisy, dl, method1_config.keep_map)

    np.testing.assert_array_equal(result.gain, np.array([0.5, 1.0, 0.0, 1.0, 1.0]))
    assert result.low_energy_neutral_bin_count == 1
    assert result.clipped_above_one_bin_count == 2
    assert result.zero_gain_bin_count == 1


def test_division_uses_larger_of_epsilon_and_low_energy_threshold():
    config = Method1KeepMapConfig(epsilon=0.1, low_energy_threshold=0.01)
    noisy = np.array([0.05, 0.2], dtype=np.complex128)
    dl = np.array([0.0, 0.1], dtype=np.complex128)

    result = estimate_raw_keep_map(noisy, dl, config)

    np.testing.assert_array_equal(result.gain, np.array([1.0, 0.5]))
    assert result.low_energy_neutral_bin_count == 0


def test_silent_and_empty_spectra_are_stable(method1_config):
    silence = np.zeros(481, dtype=np.complex128)
    silent_result = estimate_raw_keep_map(silence, silence, method1_config.keep_map)
    np.testing.assert_array_equal(silent_result.gain, np.ones(481))
    assert silent_result.low_energy_neutral_bin_count == 481

    empty = np.zeros(0, dtype=np.complex128)
    empty_result = estimate_raw_keep_map(empty, empty, method1_config.keep_map)
    assert empty_result.gain.shape == (0,)
    assert empty_result.bin_count == 0
    assert empty_result.low_energy_neutral_bin_count == 0


def test_keep_map_does_not_mutate_or_alias_inputs(method1_config):
    noisy = np.array([1.0 + 2.0j, -3.0j], dtype=np.complex64)
    dl = np.array([0.5 - 1.0j, 1.0j], dtype=np.complex64)
    noisy_before = noisy.copy()
    dl_before = dl.copy()

    result = estimate_raw_keep_map(noisy, dl, method1_config.keep_map)

    np.testing.assert_array_equal(noisy, noisy_before)
    np.testing.assert_array_equal(dl, dl_before)
    assert not np.shares_memory(result.gain, noisy)
    assert not np.shares_memory(result.gain, dl)
    assert np.all(np.isfinite(result.gain))
    assert np.all((0.0 <= result.gain) & (result.gain <= 1.0))


def test_phase_never_changes_magnitude_ratio(method1_config):
    noisy = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
    dl = 0.25 * np.array([1.0j, -1.0, -1.0j, 1.0], dtype=np.complex128)

    result = estimate_raw_keep_map(noisy, dl, method1_config.keep_map)

    np.testing.assert_allclose(result.gain, np.full(4, 0.25), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "noisy,dl,exception,message",
    [
        ([1.0 + 0j], np.ones(1, dtype=np.complex128), TypeError, "numpy.ndarray"),
        (
            np.ones((1, 2), dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            ValueError,
            "one-dimensional",
        ),
        (
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.complex128),
            TypeError,
            "complex floating",
        ),
        (
            np.ones(2, dtype=np.complex128),
            np.ones(3, dtype=np.complex128),
            ValueError,
            "equal shapes",
        ),
        (
            np.array([np.nan + 0j], dtype=np.complex128),
            np.ones(1, dtype=np.complex128),
            ValueError,
            "finite",
        ),
        (
            np.ones(1, dtype=np.complex128),
            np.array([np.inf + 0j], dtype=np.complex128),
            ValueError,
            "finite",
        ),
    ],
)
def test_rejects_invalid_spectrum_contract(noisy, dl, exception, message, method1_config):
    with pytest.raises(exception, match=message):
        estimate_raw_keep_map(noisy, dl, method1_config.keep_map)


def test_rejects_wrong_or_invalid_keep_map_config(method1_config):
    spectrum = np.ones(2, dtype=np.complex128)
    with pytest.raises(TypeError, match="Method1KeepMapConfig"):
        estimate_raw_keep_map(spectrum, spectrum, object())

    invalid = dataclasses.replace(method1_config.keep_map, epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        estimate_raw_keep_map(spectrum, spectrum, invalid)
