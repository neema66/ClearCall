#!/usr/bin/env python
"""
Convenience wrapper for running the live demo from the repo root without
needing to remember the `python -m senhance.main` module syntax.

Usage:
    python scripts/run_live_demo.py
    python scripts/run_live_demo.py --list-devices
"""

import sys
from pathlib import Path

# Allow running this script directly without installing the package
# (adds src/ to the path). If you `pip install -e .`, this isn't needed,
# but it keeps things simple for teammates who haven't set that up yet.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senhance.main import main  # noqa: E402

if __name__ == "__main__":
    main()
