# Architecture

This document describes the system design for the real-time speech
enhancement project, the scope decisions the team has made, and the
rationale behind them. If you're new to the codebase, read this before
writing any code.

## Project Scope (Windows-Only Safe Track)

After weighing available team hours against the original ambitious plan,
we settled on the following scope for the Week 4 demo:

- **One OS target: Windows**, using VB-Cable as the virtual microphone.
  Cross-platform (macOS) support is a post-demo stretch goal.
- **DSP pipeline runs live**, in the real-time microphone -> virtual mic
  loop.
- **Deep learning (DeepFilterNet) pipeline runs offline-only** for the
  demo -- it's benchmarked against the DSP pipeline on a test dataset,
  but is not required to meet the live latency budget by Week 4.
- **Simplified threading**: a plain thread-safe queue (`queue.Queue`)
  connects the audio callback to the processing loop, rather than a
  hand-rolled lock-free ring buffer. This is much faster to get correct;
  upgrading to a stricter non-blocking design is an explicit stretch goal
  if time allows (see "Checkpoints" below).
- **VAD is skipped** for the demo (a known limitation -- see the
  noise estimator's docstring for what this means in practice).
- **Manual evaluation**, not a fully automated benchmark harness with
  regression tracking.

None of this is set in stone -- see "Checkpoints and Fallback Triggers"
below for exactly when and how scope gets revisited.

## Data Flow

```
Microphone --> AudioStreamManager (capture callback)
                        |
                        v
                 AudioBuffer (queue.Queue)
                        |
                        v
              EnhancementStrategy.process()
              (DSPPipeline, live)
                        |
                        v
                 Output audio stream
                        |
                        v
              Virtual Microphone (VB-Cable)
                        |
                        v
                 Zoom / Teams / etc.
```

Separately, offline evaluation runs entirely outside this live loop:

```
data/noisy/*.wav --> EnhancementStrategy.process() (DSP or DL)
                            |
                            v
                data/clean/*.wav (reference) --> metrics.py
                            |
                            v
                  PESQ / STOI / SNR results
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `senhance.audio.stream_manager` | Opens mic input / virtual mic output via `sounddevice`, runs the real-time callback and the processing loop |
| `senhance.audio.buffer` | Thread-safe queue connecting the callback thread to the processing loop |
| `senhance.pipeline.base` | `EnhancementStrategy` abstract interface -- both DSP and DL pipelines implement this |
| `senhance.pipeline.dsp.*` | STFT, noise estimation, spectral subtraction, Wiener filter, and the `DSPPipeline` that wires them together |
| `senhance.pipeline.dl.deepfilternet_wrapper` | DeepFilterNet wrapper (offline-only stub, see TODOs in the file) |
| `senhance.config.settings` | Loads and validates `config/*.yaml` into typed dataclasses -- the single source of truth for all tunable parameters |
| `senhance.logging_setup.logger` | Application-wide logging setup |
| `senhance.evaluation.metrics` | PESQ/STOI/SNR wrappers |
| `senhance.evaluation.evaluate` | Batch evaluation script comparing pipelines against the dataset |
| `senhance.evaluation.benchmark_latency` | Measures per-frame processing latency against the budget in config |

## Threading Model

- **Real-time callback** (`AudioStreamManager._audio_callback`): runs on
  `sounddevice`'s audio thread. Only copies data into `AudioBuffer` --
  never logs, allocates heavily, or calls into the enhancement pipeline
  directly.
- **Processing loop** (`AudioStreamManager.run`): runs on the main thread
  (or a worker thread, if you choose to spawn one) and does the actual
  enhancement work. This is deliberately simple for the Safe Track scope
  -- a blocking loop reading from the queue, not a fully decoupled
  non-blocking architecture.

**Upgrade path (stretch goal):** if profiling shows the queue is
introducing meaningful latency or jitter, replace `AudioBuffer`'s
internal `queue.Queue` with a proper lock-free ring buffer. Because
`AudioBuffer`'s public interface (`put_frame` / `get_frame`) doesn't
change, this is a contained change, not a rewrite of the rest of the
system.

## Latency Budget

See `config/default.yaml`, `latency_budget_ms`. The DSP pipeline is
expected to fit comfortably within the 40ms total budget; the DL
pipeline's budget is aspirational (for when/if it moves to live
processing) and not a hard requirement for the Week 4 demo.

Use `python -m senhance.evaluation.benchmark_latency --pipeline dsp` to
check actual processing time against this budget.

## Checkpoints and Fallback Triggers

These were agreed on with the team's technical mentor to make
"simplify as we go" concrete instead of something discovered under
pressure late in the project:

| Checkpoint | If NOT met... | Fallback |
|---|---|---|
| End of Week 1 | Live passthrough (mic -> virtual mic, no enhancement) isn't stable on Windows | Re-check `AudioBuffer` sizing and callback overrun counts before anything else; do not start DSP work until this is solid |
| End of Week 2 | DeepFilterNet isn't hitting real-time inference speed on team hardware | Confirmed: DL pipeline stays offline-only for the demo (this is already the default plan either way) |
| End of Week 3 | Live DSP pipeline isn't running within the latency budget | Cut VAD and any benchmark automation work first; protect the core live loop above all else |

Revisit this table honestly at each checkpoint. It's fine to be ahead of
plan and pull forward a stretch goal (second OS, live DL, stricter
threading) -- but a missed checkpoint should trigger the documented
fallback, not silent scope creep in either direction.

## Known Limitations (Current)

- The noise estimator (`senhance.pipeline.dsp.noise_estimator`) assumes
  the first `noise_estimation_frames` frames are noise-only. Without a
  VAD gate, if speech starts before calibration ends, the estimator will
  partially learn to suppress the speech itself. This is a known,
  documented limitation -- see the TODO in that file.
- `DSPPipeline.process()` assumes `block_size == frame_size` in config
  (i.e., no internal accumulation across multiple audio callback blocks
  into one STFT frame yet). See the TODO in `processor.py`.
- `DeepFilterNetPipeline` is currently a stub -- see TODOs in
  `deepfilternet_wrapper.py` for exactly what needs implementing.

## Further Reading

- `docs/setup.md` -- installation and environment setup
- `docs/development_workflow.md` -- Git workflow, code review, testing
  expectations
- `docs/evaluation_plan.md` -- how PESQ/STOI/SNR evaluation works and
  how to interpret results
