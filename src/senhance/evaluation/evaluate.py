"""
Batch evaluation script: runs a pipeline against a folder of clean/noisy
pairs and reports PESQ, STOI, and SNR improvement.

Safe Track note: this is a manual script you run by hand at the end of
each week (see docs/architecture.md, "What Changed" section) -- it is
not wired into CI or run automatically. Upgrading to an automated
`benchmark_runner.py` with regression tracking is an Ambitious Track item.

Usage:
    python -m senhance.evaluation.evaluate --pipeline dsp
    python -m senhance.evaluation.evaluate --pipeline dl
"""

from __future__ import annotations
import tempfile
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from senhance.config.settings import AppSettings, load_settings
from senhance.evaluation.metrics import (
    compute_pesq,
    compute_snr_improvement,
    compute_stoi,
    resample_for_metrics,
)
from senhance.logging_setup.logger import configure_logging, get_logger
from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.dsp.processor import DSPPipeline

logger = get_logger(__name__)


def _build_pipeline(pipeline_name: str, settings: AppSettings) -> EnhancementStrategy:
    if pipeline_name == "dsp":
        return DSPPipeline(settings)
    elif pipeline_name == "dl":
        # TODO (Member 3): wire up DeepFilterNetPipeline once
        # implemented -- see senhance.pipeline.dl.deepfilternet_wrapper.
        from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetPipeline

        return DeepFilterNetPipeline(settings)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


def enhance_offline(pipeline: EnhancementStrategy, noisy: np.ndarray, settings: AppSettings) -> np.ndarray:
    """
    Run a whole (offline) noisy clip through a frame-based
    EnhancementStrategy by chunking it into frames and reassembling.

    TODO: for DeepFilterNetPipeline, prefer calling `process_file`
    directly instead of this generic frame-chunking loop, since
    DeepFilterNet has its own internal framing conventions.
    """
    frame_size = settings.frame_size_samples
    hop_size = settings.hop_size_samples

    pipeline.reset()
    output_chunks = []
    for start in range(0, len(noisy) - frame_size, hop_size):
        frame = noisy[start : start + frame_size]
        output_chunks.append(pipeline.process(frame))
    return np.concatenate(output_chunks) if output_chunks else np.zeros(0, dtype=np.float32)


def evaluate_dataset(pipeline_name: str, config_path: str = "config/default.yaml") -> None:
    """
    Evaluate a pipeline against every clean/noisy pair in the dataset
    directories configured in config/default.yaml, and print a summary
    table of PESQ / STOI / SNR-improvement results.
    """
    settings = load_settings(config_path)
    pipeline = _build_pipeline(pipeline_name, settings)

    clean_dir = Path(settings.evaluation.dataset_dir) / settings.evaluation.clean_subdir
    noisy_dir = Path(settings.evaluation.dataset_dir) / settings.evaluation.noisy_subdir

    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        logger.warning(
            "No .wav files found in %s -- see data/README.md for how to "
            "populate sample data before running evaluation.",
            clean_dir,
        )
        return

    results = []
    for clean_path in clean_files:
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            logger.warning("No matching noisy file for %s, skipping.", clean_path.name)
            continue

        clean, sr_clean = sf.read(clean_path, dtype="float32")
        noisy, sr_noisy = sf.read(noisy_path, dtype="float32")
        assert sr_clean == sr_noisy, "Clean/noisy sample rates must match."

        # =========modified
        if pipeline_name == "dl":
            from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetPipeline

            with tempfile.TemporaryDirectory() as tmpdir:
                noisy_tmp = Path(tmpdir) / "noisy.wav"
                enhanced_tmp = Path(tmpdir) / "enhanced.wav"

                sf.write(noisy_tmp, noisy, sr_noisy)

                assert isinstance(pipeline, DeepFilterNetPipeline)
                pipeline.process_file(noisy_tmp, enhanced_tmp)

                enhanced, sr_enhanced = sf.read(enhanced_tmp, dtype="float32")
                assert sr_enhanced == sr_noisy
        else:
            enhanced = enhance_offline(pipeline, noisy, settings)
        # ---------------------
        
        # Align lengths (framing may produce a slightly shorter output).
        n = min(len(clean), len(noisy), len(enhanced))
        clean, noisy, enhanced = clean[:n], noisy[:n], enhanced[:n]

        metric_sr = settings.evaluation.sample_rate_for_metrics
        clean_rs = resample_for_metrics(clean, sr_clean, metric_sr)
        enhanced_rs = resample_for_metrics(enhanced, sr_clean, metric_sr)

        snr_improvement = compute_snr_improvement(clean, noisy, enhanced)
        pesq_score = compute_pesq(clean_rs, enhanced_rs, metric_sr, settings.evaluation.pesq_mode)
        stoi_score = compute_stoi(clean_rs, enhanced_rs, metric_sr)

        results.append((clean_path.name, snr_improvement, pesq_score, stoi_score))
        logger.info(
            "%s: SNR improvement=%.2f dB, PESQ=%.2f, STOI=%.2f",
            clean_path.name,
            snr_improvement,
            pesq_score,
            stoi_score,
        )

    if results:
        avg_snr = np.mean([r[1] for r in results])
        avg_pesq = np.mean([r[2] for r in results])
        avg_stoi = np.mean([r[3] for r in results])
        logger.info(
            "=== %s pipeline averages over %d files: SNR=%.2f dB, PESQ=%.2f, STOI=%.2f ===",
            pipeline_name.upper(),
            len(results),
            avg_snr,
            avg_pesq,
            avg_stoi,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an enhancement pipeline offline.")
    parser.add_argument(
        "--pipeline", choices=["dsp", "dl"], default="dsp", help="Which pipeline to evaluate."
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to a config YAML file."
    )
    args = parser.parse_args()

    configure_logging()
    evaluate_dataset(args.pipeline, args.config)


if __name__ == "__main__":
    main()
