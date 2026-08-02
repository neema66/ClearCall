"""
Main entry point for the real-time speech enhancement system.

This starts the live microphone -> enhancement -> virtual microphone loop.
Of the project's 5 enhancement methods (see docs/project_structure.md),
only `original_dsp` and `improved_dsp` can genuinely process audio
frame-by-frame in real time and are selectable here. `deepfilternet3`,
`hybrid_method_1`, and `hybrid_method_3` are architecturally offline/
whole-array-only (DeepFilterNet has no streaming implementation, and both
hybrids require a complete pre-computed array up front) -- audition those
through scripts/run_virtual_mic_test.py instead, which feeds a WAV file
through their real APIs and plays the result out the same virtual-mic
device this live loop uses.

For offline single-file processing instead of any live loop, see
scripts/run_offline_demo.py or `python -m senhance.evaluation.evaluate`.

Usage:
    python -m senhance.main
    python -m senhance.main --pipeline improved_dsp
    python -m senhance.main --config config/local.yaml
    python -m senhance.main --list-devices
"""

from __future__ import annotations

import argparse
import sys

from senhance.audio.stream_manager import AudioStreamManager
from senhance.config.settings import AppSettings, load_settings
from senhance.logging_setup.logger import configure_logging, get_logger
from senhance.pipeline.base import EnhancementStrategy
from senhance.pipeline.dsp.live_strategy import DSPBlockStrategy
from senhance.pipeline.dsp.processor import DSPPipeline

logger = get_logger(__name__)

# Each offline-only pipeline mapped to the class that documents why it
# can't stream, so the error message below points somewhere concrete.
_OFFLINE_ONLY_PIPELINES = {
    "dl": "senhance.pipeline.dl.deepfilternet_wrapper.DeepFilterNetPipeline",
    "hybrid_method_1": "senhance.pipeline.hybrid.method1.processor.Method1SafetyLayer",
    "hybrid_method_3": "senhance.pipeline.hybrid.method3.blender.FixedWaveformBlender",
}


def _build_live_strategy(
    pipeline_name: str, settings: AppSettings, dsp_config_path: str
) -> EnhancementStrategy:
    """Construct the live `EnhancementStrategy` for a selectable pipeline."""

    if pipeline_name == "original_dsp":
        return DSPBlockStrategy(DSPPipeline(settings))
    if pipeline_name == "improved_dsp":
        from senhance.pipeline.improved_dsp import (
            ImprovedDSPBlockStrategy,
            ImprovedDSPPipeline,
            load_improved_dsp_config,
        )

        config = load_improved_dsp_config(dsp_config_path)
        return ImprovedDSPBlockStrategy(ImprovedDSPPipeline(settings, config))
    if pipeline_name in _OFFLINE_ONLY_PIPELINES:
        raise ValueError(
            f"'{pipeline_name}' has no live/streaming implementation -- it only "
            f"processes a complete whole-clip array ({_OFFLINE_ONLY_PIPELINES[pipeline_name]}"
            ".process_array(...)). Use scripts/run_virtual_mic_test.py "
            f"--pipeline {pipeline_name} to audition it through the virtual mic "
            "from a WAV file instead of a live microphone."
        )
    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live speech enhancement demo.")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to a config YAML file."
    )
    parser.add_argument(
        "--pipeline",
        choices=["original_dsp", "improved_dsp", "dl", "hybrid_method_1", "hybrid_method_3"],
        default="original_dsp",
        help="Which enhancement method drives the live loop. Only original_dsp "
        "and improved_dsp can actually stream in real time; the other three "
        "are offline-only (see scripts/run_virtual_mic_test.py).",
    )
    parser.add_argument(
        "--dsp-config",
        default="config/improved_dsp.yaml",
        help="Algorithm config for --pipeline improved_dsp (ignored otherwise).",
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

    # Both live-capable pipelines derive frame/hop timing from the same
    # AppSettings, and AudioStreamManager always calls process() with
    # audio.block_size chunks -- so this check applies uniformly and can
    # run before constructing anything.
    hop_size = settings.hop_size_samples
    if settings.audio.block_size != hop_size:
        raise ValueError(
            f"Live '{args.pipeline}' requires audio.block_size to equal the STFT "
            f"hop ({hop_size} samples), got {settings.audio.block_size}"
        )

    strategy = _build_live_strategy(args.pipeline, settings, args.dsp_config)

    logger.info("Starting live audio with pipeline=%s", args.pipeline)
    manager = AudioStreamManager(settings, strategy)
    manager.start()
    manager.run()


if __name__ == "__main__":
    main()
