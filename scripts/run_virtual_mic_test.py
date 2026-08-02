#!/usr/bin/env python
"""Audition any of the 5 enhancement methods through the virtual mic.

Not a live microphone loop: this feeds a noisy WAV file through the
selected method's real, existing whole-array API (running DeepFilterNet
first for methods that need a precomputed DL array, and the configured
DSP backend first for hybrid_method_3), then plays the finished result
out through the same virtual-mic device (`audio.output_device`) Zoom/
Teams would read from -- or writes it to a WAV file with --output-file.

This exists because deepfilternet3, hybrid_method_1, and hybrid_method_3
are architecturally offline-only (see senhance.main's module docstring
for why) and can't be selected for a genuine live microphone loop. Use
`python -m senhance.main --pipeline {original_dsp,improved_dsp}` instead
for those two.

Usage:
    python scripts/run_virtual_mic_test.py --pipeline original_dsp --input data/noisy/p232_001.wav
    python scripts/run_virtual_mic_test.py --pipeline improved_dsp --input data/noisy/p232_001.wav --loop
    python scripts/run_virtual_mic_test.py --pipeline dl --input data/noisy/p232_001.wav --output-file outputs/dl.wav
    python scripts/run_virtual_mic_test.py --pipeline hybrid_method_1 --input data/noisy/p232_001.wav --method1-variant full_dl_phase
    python scripts/run_virtual_mic_test.py --pipeline hybrid_method_3 --input data/noisy/p232_001.wav --alpha 0.5
    python scripts/run_virtual_mic_test.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from senhance.config.settings import AppSettings, load_settings  # noqa: E402
from senhance.evaluation.evaluate import enhance_offline  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402
from senhance.pipeline.dl.deepfilternet_wrapper import DeepFilterNetPipeline  # noqa: E402
from senhance.pipeline.dsp.processor import DSPPipeline  # noqa: E402
from senhance.pipeline.hybrid.method1 import Method1SafetyLayer, load_method1_config  # noqa: E402
from senhance.pipeline.hybrid.method3 import FixedWaveformBlender, load_hybrid_config  # noqa: E402
from senhance.pipeline.hybrid.method3.method3_config import load_method3_config  # noqa: E402
from senhance.pipeline.improved_dsp import ImprovedDSPPipeline, load_improved_dsp_config  # noqa: E402

logger = get_logger(__name__)

PIPELINE_CHOICES = ("original_dsp", "improved_dsp", "dl", "hybrid_method_1", "hybrid_method_3")

# Maps hybrid_method_3's configured DSP backend ids (config/hybrid_method_3.yaml
# -> method_3.dsp.backends) to how to compute that backend's array. Not the
# same id space as PIPELINE_CHOICES ("legacy_dsp" here vs. "original_dsp"
# there) -- kept as an explicit table rather than assumed to line up.
_METHOD3_BACKENDS = ("improved_dsp", "legacy_dsp")


def _compute_dsp_backend_array(
    backend_id: str, args: argparse.Namespace, settings: AppSettings, noisy: np.ndarray
) -> np.ndarray:
    if backend_id == "improved_dsp":
        config = load_improved_dsp_config(args.dsp_config)
        return ImprovedDSPPipeline(settings, config).process_array(noisy)
    if backend_id == "legacy_dsp":
        return enhance_offline(DSPPipeline(settings), noisy, settings)
    raise ValueError(
        f"Unhandled hybrid_method_3 DSP backend {backend_id!r}; known backends: "
        f"{_METHOD3_BACKENDS}"
    )


def compute_enhanced_array(
    pipeline_name: str,
    args: argparse.Namespace,
    settings: AppSettings,
    noisy: np.ndarray,
    sr: int,
    *,
    dl_enhancer: object | None = None,
) -> np.ndarray:
    """Run the selected method's real whole-array API end to end.

    `dl_enhancer` is an optional injected `ArrayEnhancer` (see
    senhance.pipeline.dl.deepfilternet_wrapper), passed straight through to
    every `DeepFilterNetPipeline` constructed here. Production callers
    never set it (real model loading applies); tests inject a fake so the
    hybrid branches are exercisable without torch/deepfilternet installed.
    """

    if pipeline_name == "original_dsp":
        return enhance_offline(DSPPipeline(settings), noisy, settings)

    if pipeline_name == "improved_dsp":
        config = load_improved_dsp_config(args.dsp_config)
        return ImprovedDSPPipeline(settings, config).process_array(noisy)

    if pipeline_name == "dl":
        return DeepFilterNetPipeline(settings, dl_enhancer).enhance_array(noisy, sr)

    if pipeline_name == "hybrid_method_1":
        dl_array = DeepFilterNetPipeline(settings, dl_enhancer).enhance_array(noisy, sr)
        config = load_method1_config(args.method1_config)
        layer = Method1SafetyLayer(config, variant_id=args.method1_variant)
        result = layer.process_array(noisy, dl_array, sample_rate=sr)
        return result.audio

    if pipeline_name == "hybrid_method_3":
        method3_cfg = load_method3_config(args.method3_config)
        backend = method3_cfg.dsp.backend()
        dsp_array = _compute_dsp_backend_array(backend.id, args, settings, noisy)
        dl_array = DeepFilterNetPipeline(settings, dl_enhancer).enhance_array(noisy, sr)

        hybrid_cfg = load_hybrid_config(args.hybrid_config)
        if hybrid_cfg.alignment.delay_samples != backend.dl_minus_dsp_delay_samples:
            # No existing code reconciles these two separately configured
            # delays (see docs/project_structure.md's separate config
            # table for hybrid_method_3). FixedWaveformBlender only ever
            # consumes hybrid.yaml's value -- surface the disagreement
            # loudly rather than silently trusting one over the other.
            logger.warning(
                "hybrid_method_3 delay mismatch: hybrid.yaml alignment.delay_samples=%d "
                "but hybrid_method_3.yaml backend %r dl_minus_dsp_delay_samples=%d. "
                "FixedWaveformBlender only uses the hybrid.yaml value.",
                hybrid_cfg.alignment.delay_samples,
                backend.id,
                backend.dl_minus_dsp_delay_samples,
            )

        blender = FixedWaveformBlender(
            hybrid_cfg, alpha=args.alpha, clipping_threshold=method3_cfg.clipping_threshold
        )
        result = blender.process_array(dsp_array, dl_array, sample_rate=sr)
        return result.audio

    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def _play_to_virtual_mic(audio: np.ndarray, sr: int, settings: AppSettings, loop: bool) -> None:
    import sounddevice as sd

    block_size = settings.audio.block_size
    channels = settings.audio.channels
    stream = sd.OutputStream(
        device=settings.audio.output_device,
        samplerate=sr,
        channels=channels,
        blocksize=block_size,
        dtype="float32",
    )
    stream.start()
    try:
        while True:
            for start in range(0, len(audio), block_size):
                chunk = audio[start : start + block_size]
                if len(chunk) < block_size:
                    padded = np.zeros(block_size, dtype=np.float32)
                    padded[: len(chunk)] = chunk
                    chunk = padded
                stream.write(chunk.reshape(-1, channels).astype("float32"))
            if not loop:
                break
            logger.info("Looping playback (Ctrl+C to stop)...")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stream.stop()
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audition an enhancement method through the virtual mic from a WAV file."
    )
    parser.add_argument("--pipeline", choices=PIPELINE_CHOICES)
    parser.add_argument("--input", help="Path to a noisy WAV file (never a clean/reference file).")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dsp-config", default="config/improved_dsp.yaml")
    parser.add_argument("--method1-config", default="config/hybrid_method_1.yaml")
    parser.add_argument(
        "--method1-variant",
        help="Required for --pipeline hybrid_method_1 (e.g. full_dl_phase). "
        "No default -- see config/hybrid_method_1.yaml method_1.variants for valid ids.",
    )
    parser.add_argument("--hybrid-config", default="config/hybrid.yaml")
    parser.add_argument("--method3-config", default="config/hybrid_method_3.yaml")
    parser.add_argument(
        "--alpha",
        type=float,
        help="Required for --pipeline hybrid_method_3 (0.0=all DSP, 1.0=all DL). "
        "No default -- config/hybrid_method_3.yaml notes no alpha is declared optimal.",
    )
    parser.add_argument("--loop", action="store_true", help="Repeat playback until Ctrl+C.")
    parser.add_argument(
        "--output-file",
        help="Write the enhanced result to this WAV path instead of opening an audio "
        "device -- lets you test without VB-Cable/audio hardware.",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        from senhance.audio.stream_manager import AudioStreamManager

        AudioStreamManager.list_devices()
        return

    if not args.pipeline:
        parser.error("--pipeline is required (unless --list-devices)")
    if not args.input:
        parser.error("--input is required (unless --list-devices)")
    if args.pipeline == "hybrid_method_1" and not args.method1_variant:
        parser.error("--pipeline hybrid_method_1 requires --method1-variant")
    if args.pipeline == "hybrid_method_3" and args.alpha is None:
        parser.error("--pipeline hybrid_method_3 requires --alpha")

    configure_logging()
    settings = load_settings(args.config)

    noisy, sr = sf.read(args.input, dtype="float32")
    if noisy.ndim != 1:
        raise ValueError(f"{args.input}: expected mono audio, got shape {noisy.shape}")
    logger.info(
        "Processing %s (%d samples, %d Hz) with %s pipeline",
        args.input,
        len(noisy),
        sr,
        args.pipeline,
    )

    enhanced = compute_enhanced_array(args.pipeline, args, settings, noisy, sr)

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_file, enhanced, sr)
        logger.info("Wrote enhanced output to %s", args.output_file)
        return

    logger.info(
        "Playing to virtual mic device=%s (Ctrl+C to stop%s)",
        settings.audio.output_device,
        ", looping" if args.loop else "",
    )
    _play_to_virtual_mic(enhanced, sr, settings, args.loop)


if __name__ == "__main__":
    main()
