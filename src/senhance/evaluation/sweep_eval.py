"""
Sweep runner for benchmark_eval.py -- runs every requested filter
across a range of noise levels and plots the results, instead of a
single before/after number.

Sits next to benchmark_eval.py and imports it directly, so it uses the
exact same engine calls, alignment fix, and metric definitions -- this
is not a separate implementation, just a loop + matplotlib around it.

Usage:

    # Sweep every filter across noise levels -40..-5 dB in 8 steps,
    # pink noise (same as benchmark_eval's --noise-type):
    python sweep_eval.py --clean my_voice.wav --filters all \\
        --noise-type pink --noise-db-min -40 --noise-db-max -5 --steps 8

    # Just a couple filters, finer sweep:
    python sweep_eval.py --clean my_voice.wav \\
        --filters lowpass,original_dsp,improved_dsp \\
        --noise-db-min -30 --noise-db-max -10 --steps 6

Writes sweep_out/sweep_results.csv (+ .json) and a handful of PNGs:
    quality_vs_noise.png   -- SNR/SI-SDR/PESQ/STOI improvement, one
                               line per filter, x-axis = noise level
    performance_bars.png   -- RTF, latency, CPU, dropout rate per
                               filter (bar charts, averaged across the
                               sweep)
    alignment_shift.png    -- how many samples of delay each pipeline
                               introduces (useful on its own, and a
                               sanity check that the alignment fix in
                               benchmark_eval.py is doing something
                               sensible)
"""

from __future__ import annotations
from .benchmark_eval import (
    FILTER_CHOICES,
    NOISE_TYPE_CHOICES,
    build_engine,
    load_wav,
    match_length,
    measure_cpu,
    performance_metrics,
    quality_metrics,
    run_through_engine,
    synthesize_noisy,
)
import argparse
import csv
import json
from pathlib import Path

import numpy as np




def run_sweep(args):

    clean = load_wav(args.clean, args.sample_rate)

    noise_levels = np.linspace(
        args.noise_db_min,
        args.noise_db_max,
        args.steps,
    )

    filters = (
        FILTER_CHOICES if args.filters == "all"
        else [f.strip() for f in args.filters.split(",")]
    )


    records = []

    for filter_name in filters:

        for noise_db in noise_levels:

            print(
                "-- filter=%s  noise_db=%.1f --"
                % (filter_name, noise_db)
            )


            sweep_args = argparse.Namespace(**vars(args))

            sweep_args.noise_db = float(noise_db)


            try:

                noisy = synthesize_noisy(clean, sweep_args)

                clean_m, noisy_m = match_length(clean, noisy)


                engine = build_engine(sweep_args, filter_name)


                def _run():

                    return run_through_engine(
                        engine,
                        noisy_m,
                        args.block_size,
                    )


                (enhanced, timings), cpu_info = measure_cpu(_run)


                perf = performance_metrics(
                    timings,
                    args.block_size,
                    args.sample_rate,
                )

                perf.update(cpu_info)


                quality = quality_metrics(
                    clean_m,
                    noisy_m,
                    enhanced,
                    args.sample_rate,
                    max_shift_ms=args.max_shift_ms,
                )


                record = {
                    "filter": filter_name,
                    "noise_db": float(noise_db),
                    **perf,
                    **quality,
                    "error": None,
                }


            except Exception as e:

                import traceback

                traceback.print_exc()

                record = {
                    "filter": filter_name,
                    "noise_db": float(noise_db),
                    "error": str(e),
                }


            records.append(record)


    return records



def save_results(records, out_dir):

    out_dir.mkdir(parents=True, exist_ok=True)


    with open(out_dir / "sweep_results.json", "w") as f:

        json.dump(records, f, indent=2)


    fieldnames = sorted(
        {key for r in records for key in r.keys()}
    )

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for r in records:

            writer.writerow(r)


    print(
        "Wrote %s and %s"
        % (
            out_dir / "sweep_results.json",
            out_dir / "sweep_results.csv",
        )
    )



def plot_results(records, out_dir):

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib not installed (pip install matplotlib) -- "
            "skipping plots, CSV/JSON still written."
        )

        return


    ok_records = [r for r in records if r.get("error") is None]

    if not ok_records:

        print("No successful runs to plot.")

        return


    filters = sorted(
        {r["filter"] for r in ok_records}
    )

    colors = plt.cm.tab10(
        np.linspace(0, 1, max(len(filters), 1))
    )

    color_map = dict(zip(filters, colors))


    # ------------------------------------------------
    # Quality vs noise level: one figure, 4 subplots
    # ------------------------------------------------

    quality_fields = [
        ("snr_improvement_db", "SNR Improvement (dB)"),
        ("si_sdr_improvement_db", "SI-SDR Improvement (dB)"),
        ("pesq", "PESQ (enhanced)"),
        ("stoi", "STOI (enhanced)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes = axes.flatten()


    for ax, (field, title) in zip(axes, quality_fields):

        for filt in filters:

            pts = sorted(
                (
                    (r["noise_db"], r[field])
                    for r in ok_records
                    if r["filter"] == filt and r.get(field) is not None
                ),
                key=lambda p: p[0],
            )

            if not pts:

                continue

            xs, ys = zip(*pts)

            ax.plot(
                xs, ys,
                marker="o",
                label=filt,
                color=color_map[filt],
            )

        ax.set_xlabel("Noise level (dB)")

        ax.set_ylabel(title)

        ax.set_title(title)

        ax.grid(alpha=0.3)


    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle("Quality vs. Noise Level, by Filter")

    fig.tight_layout()

    fig.savefig(out_dir / "quality_vs_noise.png", dpi=150)

    plt.close(fig)


    # ------------------------------------------------
    # Performance bar charts, averaged per filter
    # ------------------------------------------------

    perf_fields = [
        ("real_time_factor", "Real-Time Factor (avg)"),
        ("latency_mean_ms", "Mean Latency (ms)"),
        ("cpu_usage_pct", "CPU Usage (%, avg)"),
        ("dropout_rate_pct", "Dropout Rate (%, avg)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes = axes.flatten()


    for ax, (field, title) in zip(axes, perf_fields):

        values = []

        labels = []

        for filt in filters:

            vals = [
                r[field] for r in ok_records
                if r["filter"] == filt and r.get(field) is not None
            ]

            if not vals:

                continue

            values.append(float(np.mean(vals)))

            labels.append(filt)


        bar_colors = [color_map[f] for f in labels]

        ax.bar(labels, values, color=bar_colors)

        ax.set_ylabel(title)

        ax.set_title(title)

        ax.tick_params(axis="x", rotation=30)

        ax.grid(alpha=0.3, axis="y")


    fig.suptitle("Performance, Averaged Across Sweep")

    fig.tight_layout()

    fig.savefig(out_dir / "performance_bars.png", dpi=150)

    plt.close(fig)


    # ------------------------------------------------
    # Alignment shift per filter (sanity check + its own metric)
    # ------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = []

    values = []

    for filt in filters:

        vals = [
            r["enhanced_alignment_shift_samples"]
            for r in ok_records
            if r["filter"] == filt
            and r.get("enhanced_alignment_shift_samples") is not None
        ]

        if not vals:

            continue

        labels.append(filt)

        values.append(float(np.mean(vals)))


    bar_colors = [color_map[f] for f in labels]

    ax.bar(labels, values, color=bar_colors)

    ax.set_ylabel("Alignment shift (samples)")

    ax.set_title("Pipeline-Introduced Delay (avg across sweep)")

    ax.tick_params(axis="x", rotation=30)

    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()

    fig.savefig(out_dir / "alignment_shift.png", dpi=150)

    plt.close(fig)


    print(
        "Wrote quality_vs_noise.png, performance_bars.png, "
        "alignment_shift.png to %s" % out_dir
    )



def parse_args():

    parser = argparse.ArgumentParser(
        description="Sweep benchmark_eval.py across noise levels and "
        "plot the results.",
    )


    parser.add_argument("--clean", required=True)

    parser.add_argument(
        "--filters", default="all",
        help="Comma-separated filter names, or 'all'. Choices: %s"
        % ", ".join(FILTER_CHOICES),
    )

    parser.add_argument("--noise-type", default="pink", choices=NOISE_TYPE_CHOICES)

    parser.add_argument("--background-file", default=None)

    parser.add_argument("--noise-db-min", type=float, default=-40)

    parser.add_argument("--noise-db-max", type=float, default=-5)

    parser.add_argument("--steps", type=int, default=8)


    parser.add_argument("--config", default=None)

    parser.add_argument("--dsp-config", default=None)

    parser.add_argument("--cutoff-low", type=float, default=300)

    parser.add_argument("--cutoff-high", type=float, default=3400)

    parser.add_argument("--order", type=int, default=4)


    parser.add_argument("--sample-rate", type=int, default=48000)

    parser.add_argument("--max-shift-ms", type=float, default=30.0)

    parser.add_argument("--block-size", type=int, default=480)


    parser.add_argument("--out-dir", default="sweep_out")


    return parser.parse_args()



def main():

    args = parse_args()

    out_dir = Path(args.out_dir)


    records = run_sweep(args)

    save_results(records, out_dir)

    plot_results(records, out_dir)



if __name__ == "__main__":

    main()