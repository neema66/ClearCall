#!/usr/bin/env python
"""Run the independent improved DSP in the existing live audio manager.

Usage:
    python scripts/run_live_improved_dsp.py
    python scripts/run_live_improved_dsp.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senhance.config.settings import load_settings  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402
from senhance.pipeline.improved_dsp import (  # noqa: E402
    ImprovedDSPBlockStrategy,
    ImprovedDSPPipeline,
    load_improved_dsp_config,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live independent improved DSP.")
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    try:
        from senhance.audio.stream_manager import AudioStreamManager
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Live audio requires the PortAudio system library; see docs/setup.md"
        ) from exc

    if args.list_devices:
        AudioStreamManager.list_devices()
        return

    settings = load_settings(args.app_config)
    config = load_improved_dsp_config(args.dsp_config)
    configure_logging(
        level=settings.logging.level,
        log_dir=settings.logging.log_dir,
        log_to_console=settings.logging.log_to_console,
        log_to_file=settings.logging.log_to_file,
    )

    pipeline = ImprovedDSPPipeline(settings, config)
    if settings.audio.block_size != pipeline.hop_size:
        raise ValueError(
            "Live improved DSP requires audio.block_size to equal the STFT hop "
            f"({pipeline.hop_size} samples), got {settings.audio.block_size}"
        )

    logger.info(
        "Starting independent improved DSP from %s and %s",
        args.app_config,
        args.dsp_config,
    )
    strategy = ImprovedDSPBlockStrategy(pipeline)
    manager = AudioStreamManager(settings, strategy)
    manager.start()
    manager.run()


if __name__ == "__main__":
    main()
