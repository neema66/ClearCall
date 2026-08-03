"""
Main entry point for the ClearCall / senhance speech enhancement system.

This now opens a launcher window rather than running a single headless
CLI loop. From the launcher you can:

- Open the real-time mic -> speaker GUI (original_dsp / improved_dsp,
  plus the classic lowpass/highpass/bandpass filters, two-mic mixing,
  and synthetic noise injection -- see senhance.experiments.gui_demo)
- Run one of the offline-only methods (dl, hybrid_method_1,
  hybrid_method_3) against a WAV file -- these have no streaming
  implementation (DeepFilterNet has no streaming implementation, and
  both hybrids require a complete pre-computed array up front)
- List audio devices

`--pipeline`, `--config`, and `--dsp-config` still work and pre-fill the
launcher / real-time GUI's fields, matching the old CLI's behavior.
`--list-devices` still runs instantly without opening any window, for
scripting/CI use.

Usage:
    python -m senhance.main
    python -m senhance.main --pipeline improved_dsp
    python -m senhance.main --config config/local.yaml
    python -m senhance.main --list-devices
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from senhance.audio.stream_manager import AudioStreamManager
from senhance.launcher import LauncherWindow
from senhance.logging_setup.logger import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the ClearCall / senhance launcher window."
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to a config YAML file."
    )
    parser.add_argument(
        "--pipeline",
        choices=["original_dsp", "improved_dsp", "dl", "hybrid_method_1", "hybrid_method_3"],
        default=None,
        help="Pre-select this pipeline when opening the launcher. Only "
        "original_dsp and improved_dsp are offered in the real-time GUI "
        "(the other three are offline-only); any of the five pre-selects "
        "the matching section.",
    )
    parser.add_argument(
        "--dsp-config",
        default="config/improved_dsp.yaml",
        help="Algorithm config for the improved_dsp pipeline (ignored otherwise).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit, without opening any "
        "window (useful for setting input_device/output_device in your "
        "config file, or from scripts/CI).",
    )
    args = parser.parse_args()

    if args.list_devices:
        AudioStreamManager.list_devices()
        sys.exit(0)

    configure_logging(
        level="INFO",
        log_dir="logs",
        log_to_console=True,
        log_to_file=True,
    )

    logger.info("Opening launcher (config=%s, pipeline=%s)", args.config, args.pipeline)

    app = QApplication(sys.argv)

    initial_realtime_pipeline = (
        args.pipeline if args.pipeline in ("original_dsp", "improved_dsp") else None
    )

    window = LauncherWindow(
        initial_config_path=args.config,
        initial_dsp_config_path=args.dsp_config,
        initial_pipeline=initial_realtime_pipeline,
    )
    window.show()

    if args.pipeline in ("dl", "hybrid_method_1", "hybrid_method_3"):
        window.offline_pipeline_box.setCurrentText(args.pipeline)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()