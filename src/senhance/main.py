"""
Main entry point for the real-time speech enhancement system.

This starts the live microphone -> DSP enhancement -> virtual microphone
loop (Safe Track scope: DSP pipeline only, Windows-only virtual audio
device). For offline evaluation instead, see scripts/run_offline_demo.py
or `python -m senhance.evaluation.evaluate`.

Usage:
    python -m senhance.main
    python -m senhance.main --config config/local.yaml
    python -m senhance.main --list-devices
"""

from __future__ import annotations

import argparse
import sys

from senhance.audio.stream_manager import AudioStreamManager
from senhance.config.settings import load_settings
from senhance.logging_setup.logger import configure_logging, get_logger
from senhance.pipeline.dsp.processor import DSPPipeline

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live speech enhancement demo.")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to a config YAML file."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit (useful for setting "
        "input_device/output_device in your config file).",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioStreamManager.list_devices()
        sys.exit(0)

    settings = load_settings(args.config)
    configure_logging(
        level=settings.logging.level,
        log_dir=settings.logging.log_dir,
        log_to_console=settings.logging.log_to_console,
        log_to_file=settings.logging.log_to_file,
    )

    logger.info("Loaded config from %s", args.config)

    # Safe Track: DSP pipeline only for the live loop. The DL pipeline
    # remains offline-only (see docs/architecture.md and
    # senhance.pipeline.dl.deepfilternet_wrapper). Swapping this line for
    # a DeepFilterNetPipeline is the natural place to upgrade later.
    strategy = DSPPipeline(settings)

    manager = AudioStreamManager(settings, strategy)
    manager.start()
    manager.run()


if __name__ == "__main__":
    main()
