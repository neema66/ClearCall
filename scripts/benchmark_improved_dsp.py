#!/usr/bin/env python
"""Benchmark independent improved-DSP frame or streaming-block processing.

Usage:
    python scripts/benchmark_improved_dsp.py --mode frame --iterations 1000
    python scripts/benchmark_improved_dsp.py --mode block --iterations 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from senhance.config.settings import load_settings  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402
from senhance.pipeline.improved_dsp import (  # noqa: E402
    ImprovedDSPPipeline,
    load_improved_dsp_config,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark independent improved DSP.")
    parser.add_argument("--mode", choices=["frame", "block"], default="block")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--app-config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    configure_logging()
    settings = load_settings(args.app_config)
    config = load_improved_dsp_config(args.dsp_config)
    pipeline = ImprovedDSPPipeline(settings, config)

    rng = np.random.default_rng(429)
    input_size = pipeline.frame_size if args.mode == "frame" else pipeline.hop_size
    audio = (0.01 * rng.normal(size=input_size)).astype(np.float32)
    process = pipeline.process if args.mode == "frame" else pipeline.process_block

    for _ in range(20):
        process(audio)

    durations_ms = np.empty(args.iterations, dtype=np.float64)
    for index in range(args.iterations):
        start = time.perf_counter_ns()
        process(audio)
        durations_ms[index] = (time.perf_counter_ns() - start) / 1_000_000.0

    hop_budget_ms = 1000.0 * pipeline.hop_size / settings.audio.sample_rate
    logger.info(
        "=== Improved DSP %s benchmark over %d iterations ===",
        args.mode,
        args.iterations,
    )
    logger.info(
        "mean=%.3f ms, median=%.3f ms, p95=%.3f ms, p99=%.3f ms, max=%.3f ms",
        durations_ms.mean(),
        np.median(durations_ms),
        np.percentile(durations_ms, 95),
        np.percentile(durations_ms, 99),
        durations_ms.max(),
    )
    logger.info(
        "10 ms hop budget utilization at p95: %.2f%%",
        100.0 * np.percentile(durations_ms, 95) / hop_budget_ms,
    )


if __name__ == "__main__":
    main()
