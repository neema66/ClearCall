"""
Objective speech quality metric wrappers: PESQ, STOI, and SNR improvement.

These wrap the `pesq` and `pystoi` PyPI packages rather than
reimplementing the metrics -- see docs/evaluation_plan.md for why (short
version: these are standardized, validated implementations; a subtly
wrong from-scratch PESQ would invalidate your results without you
necessarily noticing).
"""

from __future__ import annotations

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly


def resample_for_metrics(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to the sample rate expected by the metric functions
    (PESQ/STOI expect 16kHz or 8kHz, while our internal pipeline runs at
    48kHz -- see config/default.yaml, evaluation.sample_rate_for_metrics).
    """
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def compute_snr_improvement(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray
) -> float:
    """
    Compute the improvement in signal-to-noise ratio (dB) achieved by
    enhancement, relative to the original noisy signal.

    Args:
        clean: Reference clean speech signal.
        noisy: Original noisy signal (same length as clean).
        enhanced: Enhanced signal (same length as clean).

    Returns:
        SNR improvement in dB (enhanced SNR - noisy SNR). Positive means
        the enhancement helped.
    """
    def _snr(reference: np.ndarray, degraded: np.ndarray) -> float:
        noise = degraded - reference
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2) + 1e-12
        return 10 * np.log10(signal_power / noise_power)

    noisy_snr = _snr(clean, noisy)
    enhanced_snr = _snr(clean, enhanced)
    return enhanced_snr - noisy_snr


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int, mode: str = "wb") -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        clean: Reference clean speech, at `sample_rate`.
        enhanced: Enhanced speech, same length/sample rate as clean.
        sample_rate: Must be 16000 (wideband, mode="wb") or 8000
            (narrowband, mode="nb") -- PESQ does not support arbitrary
            rates. Use resample_for_metrics() first if needed.
        mode: "wb" or "nb".

    Returns:
        PESQ score (roughly -0.5 to 4.5, higher is better).
    """
    return pesq(sample_rate, clean, enhanced, mode)


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility).

    Args:
        clean: Reference clean speech.
        enhanced: Enhanced speech, same length/sample rate as clean.
        sample_rate: Sample rate of both signals.

    Returns:
        STOI score (0 to 1, higher is better / more intelligible).
    """
    return stoi(clean, enhanced, sample_rate, extended=False)
