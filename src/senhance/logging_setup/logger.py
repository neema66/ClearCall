"""
Logging setup for the speech enhancement system.

Provides a single `get_logger()` entry point used throughout the codebase
instead of ad-hoc `print()` statements. Configure once at application
startup (see scripts/run_live_demo.py or scripts/run_offline_demo.py),
then call `get_logger(__name__)` anywhere else.

IMPORTANT (real-time safety):
Do NOT call logger methods synchronously from inside the real-time audio
callback (see senhance.audio.stream_manager). Logging involves I/O and
can block, which can cause audio glitches/dropouts. Instead, accumulate
lightweight counters in the callback and log them periodically from the
processing loop or a separate thread. See docs/architecture.md, Section
on Threading Model, for details.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_filename: str = "senhance.log",
) -> None:
    """
    Configure the root logger for the whole application. Call this once,
    early in your entry-point script (main.py / scripts/*.py).

    Args:
        level: Logging level name, e.g. "DEBUG", "INFO", "WARNING".
        log_dir: Directory where log files are written.
        log_to_console: If True, also stream logs to stdout.
        log_to_file: If True, write logs to a rotating file under log_dir.
        log_filename: Name of the log file within log_dir.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return  # Avoid adding duplicate handlers if called more than once.

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If configure_logging() hasn't been called yet,
    this falls back to a basic default configuration so modules can still
    log sensibly during unit tests / ad-hoc scripts.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
