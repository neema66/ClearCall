# Evaluation Plan

This document explains how we measure whether the enhancement pipelines
actually work, and how to interpret the results.

## Metrics Used

| Metric | What it measures | Range | Reference |
|---|---|---|---|
| **SNR Improvement** | Signal-to-noise ratio gain from enhancement, relative to the original noisy signal | dB, higher is better (0 = no improvement) | Computed directly, see `senhance.evaluation.metrics.compute_snr_improvement` |
| **PESQ** | Perceptual Evaluation of Speech Quality -- an ITU-T standard estimate of subjective audio quality | Roughly -0.5 to 4.5, higher is better | A. Rix et al., "Perceptual evaluation of speech quality (PESQ)," ICASSP 2001 |
| **STOI** | Short-Time Objective Intelligibility -- estimates how intelligible speech is, not just how "clean" it sounds | 0 to 1, higher is better | C. Taal et al., "An algorithm for intelligibility prediction...," IEEE TASLP 2011 |

We use the `pesq` and `pystoi` PyPI packages rather than reimplementing
these metrics -- both are standardized, validated implementations, and a
subtly-wrong from-scratch version would quietly invalidate comparisons
between our two pipelines.

## Why Both PESQ and STOI?

PESQ and STOI measure different things and can disagree. A pipeline
might sound less "clean" (lower PESQ) but remain highly intelligible
(high STOI), or vice versa. Report both, and if they disagree for a
given clip, that disagreement itself is worth discussing in the final
report rather than picking whichever number looks better.

## Dataset

Evaluation runs against matched clean/noisy WAV pairs in `data/clean/`
and `data/noisy/` (see `data/README.md` for how to populate this).
We recommend using a subset of the
[Microsoft DNS Challenge dataset](https://github.com/microsoft/DNS-Challenge)
rather than the full multi-GB release, so evaluation stays fast enough
to run weekly.

## Running an Evaluation

```bash
python -m senhance.evaluation.evaluate --pipeline dsp
python -m senhance.evaluation.evaluate --pipeline dl
```

This prints a per-file breakdown and an averaged summary across the
dataset, e.g.:

```
sample1.wav: SNR improvement=4.32 dB, PESQ=2.10, STOI=0.81
sample2.wav: SNR improvement=3.87 dB, PESQ=1.95, STOI=0.78
=== DSP pipeline averages over 2 files: SNR=4.10 dB, PESQ=2.03, STOI=0.80 ===
```

## Sample Rate Handling

Our internal pipeline runs at 48kHz (see `config/default.yaml`,
`audio.sample_rate`), but PESQ/STOI expect 16kHz (wideband mode) or
8kHz (narrowband mode). `senhance.evaluation.metrics.resample_for_metrics`
handles this conversion automatically before computing scores -- SNR
improvement is computed at the original sample rate, since it doesn't
have this restriction.

## Latency Benchmarking

Separate from audio quality, we also track processing latency against
the budget defined in `config/default.yaml` (`latency_budget_ms`):

```bash
python -m senhance.evaluation.benchmark_latency --pipeline dsp --iterations 500
```

This isolates *processing time per frame*, not full mic-to-speaker
latency (which also includes OS audio buffering overhead not under our
direct control). A pipeline that fails this benchmark cannot run live
regardless of how good its PESQ/STOI scores are offline -- this is why
the DL pipeline is offline-only for the Week 4 demo (see
`docs/architecture.md`).

## Weekly Tracking (Manual, Safe Track Scope)

Per the Safe Track scope, there is no automated regression-tracking
harness yet (that's a documented stretch goal). Instead:

1. Run both evaluation commands above at the end of each week.
2. Save the console output under `outputs/` with a dated filename, e.g.
   `outputs/eval_week2_dsp.txt`.
3. Keep a simple running table (in a shared doc, or eventually in the
   final report) of how SNR/PESQ/STOI change week over week as the DSP
   algorithm is tuned.

This gives you a defensible, evidence-based "Results and Discussion"
section for the final report, rather than reconstructing evaluation
history from memory near the deadline.

## Interpreting a "Worse Than Expected" Result

If a metric looks worse than you expected for a given pipeline, this is
a valid and reportable finding, not a failure to hide. Possible causes
worth checking, in rough order of likelihood:

1. Sample rate / length mismatch between clean and enhanced signals
   (double-check alignment before computing metrics).
2. Over-aggressive spectral subtraction settings introducing "musical
   noise" (try lowering `oversubtraction_factor` in
   `config/default.yaml`).
3. The noise estimator's calibration-frame assumption being violated
   (see the known limitation noted in `docs/architecture.md`).
4. A genuine limitation of the approach worth discussing in the report
   (e.g., classical DSP methods are well known to underperform deep
   learning approaches on non-stationary noise -- this is expected and
   fine to report honestly).
