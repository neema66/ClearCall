#!/usr/bin/env python
"""Enhance one WAV file with the independent improved DSP pipeline.

Usage:
    python scripts/run_improved_dsp.py \
        --input data/noisy/p232_001.wav \
        --output outputs/p232_001_improved.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf  # noqa: E402

from senhance.config.settings import load_settings  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402
from senhance.pipeline.improved_dsp import (  # noqa: E402
    ImprovedDSPPipeline,
    load_improved_dsp_config,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance one WAV using independent improved classical DSP."
    )
    parser.add_argument("--input", required=True, help="Input noisy mono WAV.")
    parser.add_argument("--output", required=True, help="Output enhanced WAV.")
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    args = parser.parse_args()

    configure_logging()
    settings = load_settings(args.app_config)
    dsp_config = load_improved_dsp_config(args.dsp_config)

    noisy, sample_rate = sf.read(args.input, dtype="float32", always_2d=False)
    if noisy.ndim != 1:
        raise ValueError("Improved DSP currently accepts mono WAV files only")
    if sample_rate != settings.audio.sample_rate:
        raise ValueError(f"Expected {settings.audio.sample_rate} Hz input, got {sample_rate} Hz")

    pipeline = ImprovedDSPPipeline(settings, dsp_config)
    enhanced = pipeline.process_array(noisy)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, enhanced, sample_rate)
    logger.info(
        "Wrote %d improved-DSP samples to %s (algorithmic delay compensated)",
        len(enhanced),
        output_path,
    )


if __name__ == "__main__":
    main()
