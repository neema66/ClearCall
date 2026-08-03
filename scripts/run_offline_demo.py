#!/usr/bin/env python
"""
Offline demo: run the DSP, whole-array DL, or Hybrid Method 2 pipeline on
a single input WAV file and save the enhanced output, without needing a
live microphone or virtual audio device.

Usage:
    python scripts/run_offline_demo.py --input INPUT.wav --output DSP_OUTPUT.wav
    python scripts/run_offline_demo.py --input INPUT.wav --output DL_OUTPUT.wav --pipeline dl
    python scripts/run_offline_demo.py --input INPUT.wav \
        --output METHOD2_OUTPUT.wav --pipeline method2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf  # noqa: E402

from senhance.config.settings import load_settings  # noqa: E402
from senhance.evaluation.evaluate import _build_pipeline, enhance_offline  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline enhancement on a single file.")
    parser.add_argument("--input", required=True, help="Path to input noisy WAV file.")
    parser.add_argument("--output", required=True, help="Path to write the enhanced WAV file.")
    parser.add_argument("--pipeline", choices=["dsp", "dl", "method2"], default="dsp")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--method2-config", default="config/hybrid_method_2.yaml")
    parser.add_argument("--improved-dsp-config", default="config/improved_dsp.yaml")
    args = parser.parse_args()

    configure_logging()
    settings = load_settings(args.config)

    if args.pipeline in {"dl", "method2"}:
        pipeline = _build_pipeline(
            args.pipeline,
            settings,
            args.method2_config,
            args.improved_dsp_config,
        )
        noisy, sr = sf.read(args.input, dtype="float32", always_2d=False)
        enhanced = pipeline.enhance_array(noisy, sr)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, enhanced, sr, subtype="FLOAT")
        logger.info("Wrote enhanced output to %s", args.output)
        return

    pipeline = _build_pipeline(args.pipeline, settings)

    noisy, sr = sf.read(args.input, dtype="float32")
    logger.info(
        "Processing %s (%d samples, %d Hz) with %s pipeline",
        args.input,
        len(noisy),
        sr,
        args.pipeline,
    )

    enhanced = enhance_offline(pipeline, noisy, settings)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced, sr)
    logger.info("Wrote enhanced output to %s", args.output)


if __name__ == "__main__":
    main()
