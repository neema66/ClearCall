"""Unit tests for DSPPipeline (spectral subtraction + Wiener combination)."""

import numpy as np

from senhance.config.settings import AppSettings
from senhance.pipeline.dsp.processor import DSPPipeline


def test_wiener_receives_true_noisy_spectrum_not_subtracted_output():
    """
    Regression test for the double-suppression cascade bug: an earlier
    version fed spectral subtraction's output into WienerFilter.apply()
    as if it were the noisy spectrum, silently compounding two
    suppression decisions computed against the same noise estimate
    (confirmed to measurably hurt PESQ/STOI/SNR -- see
    docs/evaluation_plan.md). WienerFilter.apply() must always receive
    the true noisy spectrum, not another stage's output.
    """
    settings = AppSettings()
    pipeline = DSPPipeline(settings)

    frame = np.zeros(settings.frame_size_samples, dtype=np.float32)
    frame[:10] = 1.0

    # StreamingSTFT.forward() has no side effects (only inverse() mutates
    # state), so calling it again here for reference is safe and doesn't
    # disturb the pipeline's own state before process() runs below.
    true_spectrum = pipeline._stft.forward(frame)

    captured = {}
    original_apply = pipeline._wiener.apply

    def spy_apply(noisy_spectrum, noise_magnitude):
        captured["noisy_spectrum"] = noisy_spectrum.copy()
        return original_apply(noisy_spectrum, noise_magnitude)

    pipeline._wiener.apply = spy_apply
    pipeline.process(frame)

    assert np.allclose(captured["noisy_spectrum"], true_spectrum)


def test_process_runs_end_to_end_with_config_derived_sizes():
    """
    Integration smoke test: no prior test exercised DSPPipeline through
    settings-derived frame sizes end-to-end. Verifies it runs across
    multiple frames without error and produces finite, bounded output.
    """
    settings = AppSettings()
    pipeline = DSPPipeline(settings)

    rng = np.random.default_rng(5)
    frame_size = settings.frame_size_samples

    for _ in range(20):
        frame = (rng.normal(size=frame_size) * 0.1).astype(np.float32)
        output = pipeline.process(frame)

        assert output.shape == (settings.hop_size_samples,)
        assert np.all(np.isfinite(output))


def test_reset_reinitializes_noise_estimator_state():
    settings = AppSettings()
    pipeline = DSPPipeline(settings)

    rng = np.random.default_rng(6)
    frame_size = settings.frame_size_samples
    for _ in range(5):
        pipeline.process((rng.normal(size=frame_size) * 0.1).astype(np.float32))

    pipeline.reset()

    assert pipeline._noise_estimator._frames_seen == 0
    assert np.all(pipeline._stft._ola_buffer == 0)
