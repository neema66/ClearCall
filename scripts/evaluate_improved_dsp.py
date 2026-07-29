#!/usr/bin/env python
"""Evaluate the independent improved DSP without changing the legacy evaluator.

By default, the script uses the repository's established overlapping-frame
loop so legacy and improved DSP are compared under identical framing. Use
``--framing streaming`` to exercise the production 480-sample block path.

Usage:
    python scripts/evaluate_improved_dsp.py
    python scripts/evaluate_improved_dsp.py --profile conservative
    python scripts/evaluate_improved_dsp.py --framing streaming --no-legacy
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from senhance.config.settings import AppSettings, load_settings  # noqa: E402
from senhance.evaluation.metrics import (  # noqa: E402
    compute_pesq,
    compute_snr_improvement,
    compute_stoi,
    resample_for_metrics,
)
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402
from senhance.pipeline.dsp.processor import DSPPipeline  # noqa: E402
from senhance.pipeline.improved_dsp import (  # noqa: E402
    ImprovedDSPConfig,
    ImprovedDSPPipeline,
    load_improved_dsp_config,
)

logger = get_logger(__name__)


class FramePipeline(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


def _framewise_enhance(
    pipeline: FramePipeline,
    noisy: np.ndarray,
    settings: AppSettings,
) -> np.ndarray:
    """Match the legacy evaluator's exact frame boundaries."""

    pipeline.reset()
    chunks = []
    for start in range(
        0,
        len(noisy) - settings.frame_size_samples,
        settings.hop_size_samples,
    ):
        frame = noisy[start : start + settings.frame_size_samples]
        chunks.append(pipeline.process(frame))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _profile_config(config: ImprovedDSPConfig, profile: str) -> ImprovedDSPConfig:
    if profile == "smoothed":
        return config
    smoothing = dataclasses.replace(config.final_gain_smoothing, enabled=False)
    return dataclasses.replace(config, final_gain_smoothing=smoothing)


def _metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    settings: AppSettings,
) -> tuple[float, float, float]:
    n = min(len(clean), len(noisy), len(enhanced))
    if n == 0:
        raise ValueError("Cannot evaluate an empty enhanced signal")
    clean = clean[:n]
    noisy = noisy[:n]
    enhanced = enhanced[:n]

    metric_rate = settings.evaluation.sample_rate_for_metrics
    clean_metric = resample_for_metrics(clean, sample_rate, metric_rate)
    enhanced_metric = resample_for_metrics(enhanced, sample_rate, metric_rate)
    return (
        compute_snr_improvement(clean, noisy, enhanced),
        compute_pesq(clean_metric, enhanced_metric, metric_rate, settings.evaluation.pesq_mode),
        compute_stoi(clean_metric, enhanced_metric, metric_rate),
    )


def evaluate(args: argparse.Namespace) -> None:
    settings = load_settings(args.app_config)
    config = _profile_config(load_improved_dsp_config(args.dsp_config), args.profile)
    improved = ImprovedDSPPipeline(settings, config)
    legacy = DSPPipeline(settings) if not args.no_legacy and args.framing == "frame" else None

    root = Path(settings.evaluation.dataset_dir)
    clean_dir = root / settings.evaluation.clean_subdir
    noisy_dir = root / settings.evaluation.noisy_subdir
    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        raise FileNotFoundError(f"No evaluation WAV files found in {clean_dir}")

    improved_results: list[tuple[float, float, float]] = []
    legacy_results: list[tuple[float, float, float]] = []
    noisy_results: list[tuple[float, float, float]] = []

    for clean_path in clean_files:
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            logger.warning("Skipping %s: matching noisy file is absent", clean_path.name)
            continue

        clean, clean_rate = sf.read(clean_path, dtype="float32", always_2d=False)
        noisy, noisy_rate = sf.read(noisy_path, dtype="float32", always_2d=False)
        if clean.ndim != 1 or noisy.ndim != 1:
            raise ValueError(f"{clean_path.name}: clean/noisy files must be mono")
        if clean_rate != noisy_rate or noisy_rate != settings.audio.sample_rate:
            raise ValueError(f"{clean_path.name}: unexpected or mismatched sample rate")

        if args.framing == "frame":
            enhanced = _framewise_enhance(improved, noisy, settings)
        else:
            enhanced = improved.process_array(noisy)

        result = _metrics(clean, noisy, enhanced, clean_rate, settings)
        improved_results.append(result)
        noisy_results.append(_metrics(clean, noisy, noisy[: len(enhanced)], clean_rate, settings))

        if legacy is not None:
            legacy_audio = _framewise_enhance(legacy, noisy, settings)
            legacy_results.append(_metrics(clean, noisy, legacy_audio, clean_rate, settings))

        logger.info(
            "%s improved: SNRi=%+.3f dB, PESQ=%.3f, STOI=%.3f",
            clean_path.name,
            result[0],
            result[1],
            result[2],
        )

    _log_average("NOISY", noisy_results)
    if legacy_results:
        _log_average("LEGACY DSP", legacy_results)
    _log_average(f"IMPROVED DSP ({args.profile}, {args.framing})", improved_results)


def _log_average(name: str, results: list[tuple[float, float, float]]) -> None:
    if not results:
        logger.warning("No results available for %s", name)
        return
    values = np.asarray(results, dtype=np.float64)
    logger.info(
        "=== %s average over %d files: SNRi=%+.3f dB, PESQ=%.3f, STOI=%.3f ===",
        name,
        len(results),
        values[:, 0].mean(),
        values[:, 1].mean(),
        values[:, 2].mean(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the independent improved DSP.")
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    parser.add_argument("--profile", choices=["smoothed", "conservative"], default="smoothed")
    parser.add_argument("--framing", choices=["frame", "streaming"], default="frame")
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Skip the legacy comparison (streaming framing always skips it).",
    )
    args = parser.parse_args()

    configure_logging()
    evaluate(args)


if __name__ == "__main__":
    main()
