#!/usr/bin/env python
"""Run the independent improved DSP in the existing live audio manager.

Thin wrapper around `senhance.main` (which now has a `--pipeline` selector
covering both live-capable methods) so there's exactly one live-loop
implementation to maintain. Kept for backward-compatible flag names.

Usage:
    python scripts/run_live_improved_dsp.py
    python scripts/run_live_improved_dsp.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senhance.main import main as run_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live independent improved DSP.")
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    forwarded = [
        "--pipeline",
        "improved_dsp",
        "--config",
        args.app_config,
        "--dsp-config",
        args.dsp_config,
    ]
    if args.list_devices:
        forwarded.append("--list-devices")

    sys.argv = [sys.argv[0], *forwarded]
    run_main()


if __name__ == "__main__":
    main()
