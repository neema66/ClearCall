#!/usr/bin/env python
"""
Waveform visualization: runs a noisy WAV file through the enhancement
pipeline and plots the before/after time-domain waveform (and optionally
spectrograms), so the effect of the DSP filters is visible, not just
audible.

Usage:
    python scripts/plot_waveform.py --input data/noisy/sample1.wav
    python scripts/plot_waveform.py --input data/noisy/sample1.wav --output outputs/sample1_waveform.png
    python scripts/plot_waveform.py --input data/noisy/sample1.wav --spectrogram
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import spectrogram as _spectrogram  # noqa: E402

from senhance.config.settings import load_settings  # noqa: E402
from senhance.evaluation.evaluate import _build_pipeline, enhance_offline  # noqa: E402
from senhance.logging_setup.logger import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot before/after waveforms of the enhancement pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to input noisy WAV file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the plot as a PNG. If omitted, opens an interactive window instead.",
    )
    parser.add_argument("--pipeline", choices=["dsp", "dl"], default="dsp")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--spectrogram",
        action="store_true",
        help="Also plot before/after spectrograms below the waveforms.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings(args.config)

    noisy, sr = sf.read(args.input, dtype="float32")
    if sr != settings.audio.sample_rate:
        logger.warning(
            "Input file sample rate (%d Hz) does not match config audio.sample_rate "
            "(%d Hz) -- the pipeline's frame sizes are derived from the config, so "
            "framing will be off. Resample the file to %d Hz, or point --config at a "
            "file that matches, before trusting these results.",
            sr,
            settings.audio.sample_rate,
            settings.audio.sample_rate,
        )

    pipeline = _build_pipeline(args.pipeline, settings)
    logger.info("Processing %s (%d samples, %d Hz) with %s pipeline", args.input, len(noisy), sr, args.pipeline)
    enhanced = enhance_offline(pipeline, noisy, settings)

    # enhance_offline() drops a partial final frame, so it may be a little
    # shorter than the input -- align lengths for plotting.
    n = min(len(noisy), len(enhanced))
    noisy, enhanced = noisy[:n], enhanced[:n]
    time = np.arange(n) / sr

    n_rows = 4 if args.spectrogram else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True, constrained_layout=True)

    axes[0].plot(time, noisy, linewidth=0.5, color="tab:red")
    axes[0].set_title(f"Noisy (before) -- {Path(args.input).name}")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time, enhanced, linewidth=0.5, color="tab:blue")
    axes[1].set_title(f"Enhanced (after) -- {args.pipeline} pipeline")
    axes[1].set_ylabel("Amplitude")

    if args.spectrogram:
        # Use the pipeline's own analysis window/hop so the spectrogram
        # resolution matches what the DSP pipeline actually "sees", and
        # compute both spectrograms up front so they can share a single
        # dB color scale -- axes.specgram() auto-scales each subplot
        # independently, which makes "before" vs "after" brightness
        # meaningless to compare directly.
        nperseg = settings.frame_size_samples
        noverlap = nperseg - settings.hop_size_samples
        eps = 1e-12

        f, t_noisy, sxx_noisy = _spectrogram(noisy, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, t_enh, sxx_enh = _spectrogram(enhanced, fs=sr, nperseg=nperseg, noverlap=noverlap)
        db_noisy = 10 * np.log10(sxx_noisy + eps)
        db_enh = 10 * np.log10(sxx_enh + eps)

        vmax = max(db_noisy.max(), db_enh.max())
        vmin = vmax - 80  # 80 dB dynamic range, floors out the noise floor cleanly

        axes[2].pcolormesh(t_noisy, f, db_noisy, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
        axes[2].set_title("Noisy spectrogram")
        axes[2].set_ylabel("Frequency (Hz)")

        mesh = axes[3].pcolormesh(
            t_enh, f, db_enh, cmap="magma", vmin=vmin, vmax=vmax, shading="auto"
        )
        axes[3].set_title("Enhanced spectrogram (same color scale as above)")
        axes[3].set_ylabel("Frequency (Hz)")

        fig.colorbar(mesh, ax=axes[2:4], label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        logger.info("Saved waveform plot to %s", args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
