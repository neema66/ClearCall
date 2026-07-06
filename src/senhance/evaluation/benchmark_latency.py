"""
Latency benchmarking script.

Measures how long a pipeline takes to process a single frame, run many
times, to check against the latency budget defined in
config/default.yaml (latency_budget_ms). This does NOT measure true
end-to-end audio latency (mic to speaker) -- it isolates the processing
stage specifically, which is the piece most under the team's control.

Usage:
    python -m senhance.evaluation.benchmark_latency --pipeline dsp --iterations 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from senhance.config.settings import load_settings
from senhance.logging_setup.logger import configure_logging, get_logger
from senhance.pipeline.dsp.processor import DSPPipeline

logger = get_logger(__name__)


def benchmark_pipeline(pipeline_name: str, iterations: int, config_path: str) -> None:
    settings = load_settings(config_path)

    if pipeline_name == "dsp":
        pipeline = DSPPipeline(settings)
    elif pipeline_name == "dl":
        # TODO (Member 3): benchmark DeepFilterNetPipeline once its
        # frame-level `process` method is implemented. For now this
        # will raise NotImplementedError, which is expected until that
        # work is done -- see senhance.pipeline.dl.deepfilternet_wrapper.
        from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetPipeline

        pipeline = DeepFilterNetPipeline(settings)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    frame_size = settings.frame_size_samples
    dummy_frame = np.random.randn(frame_size).astype(np.float32) * 0.01

    # Warm-up iterations (not timed) to avoid measuring one-time setup
    # costs (e.g. FFT plan caching, lazy imports).
    for _ in range(10):
        pipeline.process(dummy_frame)

    durations_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        pipeline.process(dummy_frame)
        durations_ms.append((time.perf_counter() - start) * 1000)

    durations_ms = np.array(durations_ms)
    frame_duration_ms = 1000 * frame_size / settings.audio.sample_rate

    logger.info("=== Latency benchmark: %s pipeline, %d iterations ===", pipeline_name, iterations)
    logger.info("Frame duration (real-time budget per frame): %.2f ms", frame_duration_ms)
    logger.info("Mean processing time:   %.3f ms", durations_ms.mean())
    logger.info("Median processing time: %.3f ms", np.median(durations_ms))
    logger.info("95th percentile:        %.3f ms", np.percentile(durations_ms, 95))
    logger.info("Max processing time:    %.3f ms", durations_ms.max())

    if durations_ms.mean() > frame_duration_ms:
        logger.warning(
            "Mean processing time exceeds the real-time frame budget! "
            "This pipeline cannot keep up with live audio at this frame size."
        )
    else:
        headroom_pct = 100 * (1 - durations_ms.mean() / frame_duration_ms)
        logger.info("Real-time headroom: %.1f%%", headroom_pct)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline processing latency.")
    parser.add_argument("--pipeline", choices=["dsp", "dl"], default="dsp")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    configure_logging()
    benchmark_pipeline(args.pipeline, args.iterations, args.config)


if __name__ == "__main__":
    main()
