"""Immutable frequency-band mapping and Method 3 spectral blend policy."""

from __future__ import annotations

import dataclasses

import numpy as np

from senhance.pipeline.hybrid.method3.config import HybridConfig
from senhance.pipeline.hybrid.method3.method3_band_config import (
    CANONICAL_BIN_WIDTH_HZ,
    BandBlendVariantConfig,
    Method3BandConfig,
)
from senhance.pipeline.hybrid.method3.paired_stft import PairedSpectrumFrame


@dataclasses.dataclass(frozen=True)
class FixedBandDefinition:
    """Read-only ownership and effective DL weight for every rFFT bin."""

    band_edges_hz: tuple[float, ...]
    band_dl_weights: tuple[float, ...]
    variant: BandBlendVariantConfig
    frequencies_hz: np.ndarray
    band_index_by_bin: np.ndarray
    step_dl_weight_by_bin: np.ndarray
    effective_dl_weight_by_bin: np.ndarray
    band_bin_counts: tuple[int, ...]

    @classmethod
    def from_config(
        cls,
        method_config: Method3BandConfig,
        variant: BandBlendVariantConfig,
        hybrid_config: HybridConfig,
    ) -> FixedBandDefinition:
        """Expand strict band policy into one bounded weight for all 481 bins."""

        if not isinstance(method_config, Method3BandConfig):
            raise TypeError("method_config must be Method3BandConfig")
        if not isinstance(variant, BandBlendVariantConfig):
            raise TypeError("variant must be BandBlendVariantConfig")
        if not isinstance(hybrid_config, HybridConfig):
            raise TypeError("hybrid_config must be HybridConfig")
        method_config.validate()
        variant.validate()
        hybrid_config.validate()
        if variant not in method_config.variants:
            raise ValueError("variant must belong to method_config.variants")

        num_bins = hybrid_config.stft.num_frequency_bins
        frequencies = np.arange(num_bins, dtype=np.float64) * CANONICAL_BIN_WIDTH_HZ
        edge_bins = np.rint(
            np.asarray(method_config.band_edges_hz, dtype=np.float64) / CANONICAL_BIN_WIDTH_HZ
        ).astype(np.int64)
        internal_edges = edge_bins[1:-1]
        bin_numbers = np.arange(num_bins, dtype=np.int64)
        band_indices = np.searchsorted(
            internal_edges,
            bin_numbers,
            side="right",
        ).astype(np.int16)

        counts_array = np.bincount(
            band_indices.astype(np.int64),
            minlength=method_config.num_bands,
        )
        if counts_array.shape != (method_config.num_bands,):
            raise AssertionError("band mapping produced an unexpected number of bands")
        if np.any(counts_array <= 0) or int(np.sum(counts_array)) != num_bins:
            raise AssertionError("frequency bands must cover all rFFT bins exactly once")

        band_weights = np.asarray(method_config.dl_weights, dtype=np.float64)
        step_weights = band_weights[band_indices]
        effective_weights = _smooth_weights(
            step_weights,
            variant.frequency_smoothing_bins,
        )
        if effective_weights.shape != (num_bins,):
            raise AssertionError("effective band-weight map must cover every rFFT bin")
        if (
            not np.all(np.isfinite(effective_weights))
            or np.any(effective_weights < 0.0)
            or np.any(effective_weights > 1.0)
        ):
            raise AssertionError("effective band weights must remain finite and bounded")

        for array in (frequencies, band_indices, step_weights, effective_weights):
            array.setflags(write=False)
        return cls(
            band_edges_hz=method_config.band_edges_hz,
            band_dl_weights=method_config.dl_weights,
            variant=variant,
            frequencies_hz=frequencies,
            band_index_by_bin=band_indices,
            step_dl_weight_by_bin=step_weights,
            effective_dl_weight_by_bin=effective_weights,
            band_bin_counts=tuple(int(value) for value in counts_array),
        )


def _smooth_weights(step_weights: np.ndarray, width: int) -> np.ndarray:
    """Use an edge-padded, non-circular odd moving average over bin weights."""

    samples = np.asarray(step_weights, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("step_weights must be a non-empty one-dimensional array")
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("smoothing width must be an integer")
    if width <= 0 or width > samples.size or width % 2 == 0:
        raise ValueError("smoothing width must be a positive odd value within the bin count")
    if not np.all(np.isfinite(samples)) or np.any(samples < 0.0) or np.any(samples > 1.0):
        raise ValueError("step weights must be finite and bounded")
    if width == 1:
        return samples.copy()
    if np.all(samples == samples[0]):
        return samples.copy()

    half = width // 2
    padded = np.pad(samples, (half, half), mode="edge")
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed, 0.0, 1.0)


class FixedBandSpectrumProcessor:
    """Blend paired spectra using one immutable per-bin DL weight map."""

    def __init__(self, definition: FixedBandDefinition) -> None:
        if not isinstance(definition, FixedBandDefinition):
            raise TypeError("definition must be FixedBandDefinition")
        self.definition = definition
        self.variant = definition.variant

    def reset(self) -> None:
        """The fixed processor is stateless; provided for the shared contract."""

    def __call__(self, frame: PairedSpectrumFrame) -> np.ndarray:
        if not isinstance(frame, PairedSpectrumFrame):
            raise TypeError("frame must be PairedSpectrumFrame")
        dsp = _validate_spectrum(frame.reference_spectrum, "DSP")
        dl = _validate_spectrum(frame.candidate_spectrum, "DL")
        if dsp.shape != self.definition.effective_dl_weight_by_bin.shape:
            raise ValueError("paired spectrum size does not match the configured band-weight map")
        weights = self.definition.effective_dl_weight_by_bin

        # Global endpoint branches preserve exact complex sources regardless
        # of the configured intermediate magnitude-phase policy.
        if np.all(weights == 0.0):
            return dsp.copy()
        if np.all(weights == 1.0):
            return dl.copy()

        output = np.empty(dsp.shape, dtype=np.complex128)
        dsp_endpoint = weights == 0.0
        dl_endpoint = weights == 1.0
        intermediate = ~(dsp_endpoint | dl_endpoint)
        output[dsp_endpoint] = dsp[dsp_endpoint]
        output[dl_endpoint] = dl[dl_endpoint]

        if self.variant.blend_domain == "complex":
            local_weights = weights[intermediate]
            output[intermediate] = (
                local_weights * dl[intermediate] + (1.0 - local_weights) * dsp[intermediate]
            )
        else:
            local_weights = weights[intermediate]
            magnitude = local_weights * np.abs(dl[intermediate]) + (1.0 - local_weights) * np.abs(
                dsp[intermediate]
            )
            phase_source = dsp if self.variant.phase_source == "dsp" else dl
            source = phase_source[intermediate]
            source_magnitude = np.abs(source)
            unit_phase = np.ones(source.shape, dtype=np.complex128)
            nonzero = source_magnitude > 0.0
            unit_phase[nonzero] = source[nonzero] / source_magnitude[nonzero]
            output[intermediate] = magnitude * unit_phase

        if not np.all(np.isfinite(output)):
            raise ValueError("fixed-band blend produced non-finite spectral values")
        return np.ascontiguousarray(output)


def _validate_spectrum(spectrum: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(f"{label} spectrum must be a numpy.ndarray")
    if spectrum.ndim != 1:
        raise ValueError(f"{label} spectrum must be one-dimensional")
    if not np.issubdtype(spectrum.dtype, np.complexfloating):
        raise TypeError(f"{label} spectrum must have a complex floating dtype")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError(f"{label} spectrum must contain only finite values")
    return np.asarray(spectrum, dtype=np.complex128)
