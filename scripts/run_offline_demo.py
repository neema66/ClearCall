#!/usr/bin/env python
"""
Offline demo: run the DSP or whole-array DL pipeline on
a single input WAV file and save the enhanced output, without needing a
live microphone or virtual audio device.

Usage:
    python scripts/run_offline_demo.py --input INPUT.wav --output DSP_OUTPUT.wav
    python scripts/run_offline_demo.py --input INPUT.wav --output DL_OUTPUT.wav --pipeline dl
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
    parser.add_argument("--pipeline", choices=["dsp", "dl"], default="dsp")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    configure_logging()
    settings = load_settings(args.config)

    if args.pipeline == "dl":
        from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetPipeline

        pipeline = DeepFilterNetPipeline(settings)
        pipeline.process_file(args.input, args.output)
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
