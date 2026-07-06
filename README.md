# Real-Time Speech Enhancement System

ENSC 429 (Digital Signal Processing) project -- Simon Fraser
University, Summer 2026.

## Project Overview

This project removes background noise from a speaker's microphone audio
in real time and outputs the enhanced result through a virtual
microphone, so it can be used as a drop-in input to Zoom, Microsoft
Teams, or similar video/voice calling applications. The goal is to make
it possible to sound clear on a call even from a noisy public
environment (a coffee shop, transit station, etc.).

We implement and compare two enhancement approaches:

1. **Classical DSP pipeline** (runs live): STFT-based spectral
   subtraction and a Wiener filter, grounded directly in ENSC 429 course
   material (FFT, filter design, random signal analysis).
2. **Deep learning pipeline** (offline comparison for the demo):
   [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), a
   pre-trained real-time-capable speech enhancement model.

> **Scope note:** For the Week 4 demo, this targets **Windows only**
> and the DL pipeline runs **offline-only** (not in the live loop). See
> `docs/architecture.md` for the full rationale and the checkpoints
> that govern when this scope might expand or need to be cut further.

## System Architecture

```
Microphone --> Capture --> Buffer --> DSP Enhancement --> Virtual Mic --> Zoom/Teams
                                            |
                                            v
                                (DeepFilterNet evaluated
                                 separately, offline, against
                                 a test dataset)
```

Both pipelines implement a common `EnhancementStrategy` interface, so
they can be swapped via configuration without touching the audio I/O or
evaluation code. See `docs/architecture.md` for the full design,
threading model, and module breakdown.

## Installation

```bash
git clone <your-repo-url>
cd speech-enhancement
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

Full setup instructions (including VB-Cable installation for the live
demo) are in `docs/setup.md`.

## Running Offline Processing

Process a single noisy WAV file and save the enhanced result, no
microphone or virtual audio device required:

```bash
python scripts/run_offline_demo.py --input data/noisy/sample1.wav --output outputs/sample1_enhanced.wav
```

## Running Real-Time Processing (Windows only)

Requires VB-Cable installed and configured -- see `docs/setup.md`,
step 4.

```bash
python scripts/run_live_demo.py
```

List available audio devices (to configure `config/default.yaml`):

```bash
python scripts/run_live_demo.py --list-devices
```

## Running Tests

```bash
pytest tests/unit -v
```

## Evaluating PESQ, STOI, and SNR

Populate `data/clean/` and `data/noisy/` with matched WAV pairs (see
`data/README.md`), then:

```bash
python -m senhance.evaluation.evaluate --pipeline dsp
python -m senhance.evaluation.evaluate --pipeline dl
```

Check processing latency against the real-time budget:

```bash
python -m senhance.evaluation.benchmark_latency --pipeline dsp
```

Full explanation of what each metric means and how to interpret results
is in `docs/evaluation_plan.md`.

## Team Roles

| Member | Role | Primary Area |
|---|---|---|
| Member 1 | Audio I/O & Systems Integration | `src/senhance/audio/`, live loop, virtual mic |
| Member 2 | Classical DSP | `src/senhance/pipeline/dsp/` |
| Member 3 | Deep Learning | `src/senhance/pipeline/dl/` |
| Member 4 | Evaluation & QA | `src/senhance/evaluation/`, documentation, report |

_(Update this table with actual names.)_

## Known Limitations

- Noise estimator assumes the first N frames are noise-only (no VAD
  gating yet) -- see `docs/architecture.md`, Known Limitations.
- DSP pipeline currently assumes `block_size == frame_size` in config
  (no internal multi-block accumulation yet).
- DeepFilterNet wrapper is a stub -- see TODOs in
  `src/senhance/pipeline/dl/deepfilternet_wrapper.py`.
- Windows-only for the live demo; macOS support is a post-demo stretch
  goal.
- Evaluation is currently a manual weekly process, not automated CI.

## Future Work

- Add voice activity detection (VAD) to stop the noise estimator from
  learning to suppress speech itself past the calibration window.
- Bring the DeepFilterNet pipeline to real-time, live operation within
  the latency budget.
- Add a second OS backend (macOS via BlackHole).
- Automate evaluation with a `benchmark_runner.py` that tracks
  metric regressions across commits.
- Upgrade the buffering layer to a stricter non-blocking, lock-free
  design if profiling shows the current queue-based approach introduces
  meaningful jitter.

## Project Documentation

- `docs/architecture.md` -- system design, scope rationale, threading
  model, known limitations
- `docs/setup.md` -- detailed installation and troubleshooting
- `docs/development_workflow.md` -- coding standards, testing, working
  in parallel as a team
- `docs/evaluation_plan.md` -- metrics explanation and how to interpret
  results
- `CONTRIBUTING.md` -- Git workflow, branching, PR, and code review
  rules
