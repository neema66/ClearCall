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

Separately, offline evaluation runs entirely outside this live loop. The DSP
uses its framed strategy while DeepFilterNet uses its validated whole-array
boundary:

```
data/noisy/*.wav --> DSP process() or DL enhance_array()
                            |
                            +---- data/clean/*.wav (reference)
                            |
                            v
                    PESQ / STOI / SNR results
```

Independent offline Hybrid Method 1 adds a second whole-array path. The outer
orchestrator runs DeepFilterNet once and caches its exact-length result;
Method 1 then receives only the noisy and cached-DL arrays:

```text
noisy -> DeepFilterNet once -> cached DL array
  |                              |
  +------------------------------+
                 |
                 v
       Method1SafetyLayer
         -> fixed alignment
         -> matching STFTs
         -> guarded DL keep map
         -> bounded DSP safety controls
         -> noisy- or DL-phase reconstruction
         -> one WOLA
         -> exact-length float32 output
```

The clean array is never supplied to enhancement. It is used afterward for
metrics and exported as the listening reference. Original DSP and
`improved_dsp` are Method 1 comparators, not internal Method 1 branches.

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `senhance.audio.stream_manager` | Opens mic input / virtual mic output via `sounddevice`, runs the real-time callback and the processing loop |
| `senhance.audio.buffer` | Thread-safe queue connecting the callback thread to the processing loop |
| `senhance.pipeline.base` | `EnhancementStrategy` abstract interface -- both DSP and DL pipelines implement this |
| `senhance.pipeline.dsp.*` | STFT, noise estimation, spectral subtraction, Wiener filter, and the `DSPPipeline` that wires them together |
| `senhance.pipeline.improved_dsp.*` | Independent MCRA-based improved DSP implementation and its offline/live adapters |
| `senhance.pipeline.dl.deepfilternet_wrapper` | Strict, injectable DeepFilterNet whole-array boundary and observed model metadata; offline only |
| `senhance.pipeline.hybrid.method3.*` | Independent Method 3 waveform and frequency-band fusion of injected complete-DSP and DL arrays |
| `senhance.pipeline.hybrid.method1.*` | Independent Method 1 alignment, paired STFT/WOLA, guarded DL keep map, DSP safety controls, phase selection, and diagnostics |
| `senhance.config.settings` | Loads and validates general application settings; Method 3 and Method 1 also own separate strict typed config loaders for their algorithm-specific YAML files |
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

- The noise estimator (`senhance.pipeline.dsp.noise_estimator`) now uses
  a minimum-statistics tracker (Martin, 2001) over a sliding window
  (`dsp.noise_estimation_window_sec`) instead of a fixed silent
  calibration period, so it no longer requires a silence lead-in.
  Without a VAD gate it can still be slow to track a noise floor that
  rises sharply mid-stream (bounded by the window length), which is an
  acceptable simplification for this project's scope.
- `DSPPipeline.process()` assumes `block_size == frame_size` in config
  (i.e., no internal accumulation across multiple audio callback blocks
  into one STFT frame yet). See the TODO in `processor.py`.
- `DeepFilterNetPipeline.enhance_array()` is complete for sequential offline
  inference, but its `process()` method intentionally does not claim
  frame-level streaming support. DeepFilterNet 0.5.x also uses process-global
  model/device configuration, so this project supports one serialized real
  model instance per process.
- Offline Method 1 and its automated development evaluation are complete, but
  there is no selected perceptual winner yet. Human listening and a fresh
  multi-speaker holdout remain pending. Its current full safety tuple causes a
  substantial objective-score regression and must not be presented as an
  optimized setting.
- Method 1 is not integrated into the live audio loop. Its current upstream
  DeepFilterNet boundary is whole-array only, so an offline real-time factor
  below one is not evidence of live-call latency or callback safety.

## Independent Method Boundaries

The maintained method IDs are `original_dsp`, `improved_dsp`,
`deepfilternet3`, `hybrid_method_1`, and `hybrid_method_3`. Original DSP,
improved DSP, and DeepFilterNet3 are standalone methods: they do not import a
hybrid or one another. Method 1 and Method 3 are sibling packages under the
hybrid namespace and do not import each other. See
[`project_structure.md`](project_structure.md) for the canonical ownership,
input/output, configuration, and evidence map.

Method 3 has two mandatory upstream branches. The noisy array is enhanced once
by the selected complete DSP (`ImprovedDSPPipeline` today) and once by
DeepFilterNet. A replaceable orchestration adapter injects both exact-length
arrays into the Method 3 core:

```text
noisy -> selected DSP --+
                        +-> align -> DSP/DL waveform or band blend
noisy -> DeepFilterNet -+
```

The blend core imports neither concrete enhancer. Switching to the original
`DSPPipeline` changes the injected DSP signal but not the blend equation; the
DSP-to-DL delay, tuning, runtime, metrics, and listening evidence must be
regenerated.

Method 1 remains independent from Method 3. It consumes noisy and cached
DeepFilterNet arrays and applies a hybrid-owned DSP safety controller to the
inferred keep map:

```text
Method1SafetyLayer(noisy, cached DL)
  -> Method-1-owned fixed alignment
  -> Method-1-owned matching 960/480 Hann STFTs
  -> guarded |DL|/|noisy| keep map
  -> temporal smoothing
  -> frequency smoothing
  -> gain floor
  -> per-hop rise/drop limits
  -> noisy or DL phase reconstruction
  -> one Method-1-owned WOLA
  -> exact-length float32 output
```

Its alignment and paired-STFT/WOLA algorithms live in the separate
`senhance.pipeline.hybrid.method1` package rather than being imported from
Method 3. The Method 1 core imports neither concrete enhancer nor Method 3.
Switching Method 3 from
`improved_dsp` to the original DSP therefore does not change Method 1;
changing the DeepFilterNet model or signal geometry still requires Method 1
alignment, tuning, metrics, and listening to be repeated.

## Further Reading

- `docs/setup.md` -- installation and environment setup
- `docs/project_structure.md` -- canonical method names, ownership boundaries,
  array contracts, and evidence layout
- `docs/development_workflow.md` -- Git workflow, code review, testing
  expectations
- `docs/evaluation_plan.md` -- how PESQ/STOI/SNR evaluation works and
  how to interpret results
