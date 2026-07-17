# Project Structure and Team Boundaries

This is the team map for the project. It defines where each method belongs,
what each method is allowed to depend on, and where evaluation evidence is
stored. Use the canonical IDs in code review, experiment names, tables, and
new documentation:

- `original_dsp`
- `improved_dsp`
- `deepfilternet3`
- `hybrid_method_1`
- `hybrid_method_3`

The five IDs describe five separately owned method families. A hybrid may
consume arrays produced by another method, but it must not modify that
method's implementation or hide enhancer construction inside its math core.

## Two Project Areas

```text
429pjt/
|-- ClearCall/                         versioned source repository
|   |-- config/                        reviewed method configuration
|   |-- docs/                          team-facing design and workflow
|   |-- scripts/                       application entry points
|   |-- src/senhance/pipeline/
|   |   |-- dsp/                       original_dsp
|   |   |-- improved_dsp/              improved_dsp
|   |   |-- dl/                        deepfilternet3
|   |   `-- hybrid/
|   |       |-- method1/               hybrid_method_1
|   |       `-- method3/               hybrid_method_3
|   `-- tests/unit/                    tests mirror the source ownership
`-- hybrid/                            generated evaluation evidence and tools
    |-- README.md                      evidence index and handling rules
    |-- method_1/                      frozen Method 1 machine evidence
    |-- method_3/                      Method 3 navigation index
    |-- milestone_0/ ... milestone_4_5/ frozen Method 3 milestone evidence
    `-- validate_*.py                  historical/reproducible evaluators
```

`ClearCall/` is the Git repository and the source of truth for implementation,
configuration, unit tests, and maintained design documents. The sibling
`hybrid/` directory is an experiment/evidence workspace. It is outside the
`ClearCall` Git boundary, so it must be archived or shared separately when a
teammate needs the exact generated WAV/CSV/JSON evidence.

The evidence directories keep their historical names because their manifests
contain hashes and some validators have path-sensitive contracts. Moving or
copying those files would weaken reproducibility. In particular, do not add
files inside `hybrid/method_1/`: its evaluator enforces an exact output
allowlist. The navigation file for Method 1 therefore stays beside that
directory, while `hybrid/method_3/` indexes the older milestone directories.

## Method Ownership Map

| Canonical ID | Maintained source | Main input | Main output | May depend on |
|---|---|---|---|---|
| `original_dsp` | `src/senhance/pipeline/dsp/` | noisy mono audio and sample rate | enhanced audio | shared base/config/logging only |
| `improved_dsp` | `src/senhance/pipeline/improved_dsp/` | noisy mono audio and sample rate | enhanced audio | shared base/config/logging only |
| `deepfilternet3` | `src/senhance/pipeline/dl/` | noisy mono audio and sample rate | enhanced audio and model metadata | DeepFilterNet boundary plus shared base/config/logging |
| `hybrid_method_1` | `src/senhance/pipeline/hybrid/method1/` | noisy audio and an already computed DL array | enhanced audio and Method 1 diagnostics | its own modules and neutral shared types only |
| `hybrid_method_3` | `src/senhance/pipeline/hybrid/method3/` | already computed DSP and DL arrays | fused audio and Method 3 diagnostics | its own modules and neutral shared types only |

Paths in this document are relative to `ClearCall/` unless stated otherwise.
The hybrid namespace contains two sibling packages; neither hybrid is an
extension of the other.

## Architecture and Replaceable DSP Selection

Enhancer selection belongs to an outer script, evaluator, or application
orchestrator:

```text
                           orchestration layer
noisy audio -> selected complete DSP ------------------+
                                                       +-> hybrid_method_3
noisy audio -> deepfilternet3 -------------------------+

noisy audio -------------------------------------------+
                                                       +-> hybrid_method_1
cached deepfilternet3 output --------------------------+
```

For `hybrid_method_3`, the selected complete DSP is `improved_dsp` today. A
future switch to `original_dsp` changes the orchestration adapter or config,
not Method 3's alignment or fusion implementation. The switch is mandatory to
support architecturally, but it is not expected to produce identical results:
delay checks, tuning, metrics, runtime, and listening evidence must be rerun
for the newly selected DSP signal.

`hybrid_method_1` uses DL plus Method-1-owned DSP operations such as STFT,
gain smoothing, bounds, and WOLA reconstruction. It does not call either
complete standalone DSP pipeline. The two standalone DSP methods remain
independent comparators.

## Stable Array Contracts

All offline method boundaries use mono, finite NumPy audio arrays and an
explicit sample rate. A successful array result must be contiguous
`float32`, finite, and exactly the expected sample count. Methods report
diagnostics separately; they do not silently clip or normalize the waveform.

- Standalone methods: `noisy + sample_rate -> enhanced`.
- `hybrid_method_3`: `dsp_result + dl_result + sample_rate -> fused_result`.
- `hybrid_method_1`: `noisy + dl_result + sample_rate -> safety_result`.
- Clean reference audio is evaluation data only. It must never enter an
  enhancer or hybrid processing call.

An orchestrator may cache a standalone result and pass it to several hybrid
variants. Hybrid code must treat input arrays as read-only and must not load a
model, instantiate a concrete DSP pipeline, or choose a backend internally.

## Dependency Rules

These rules keep parallel team work safe:

1. `original_dsp`, `improved_dsp`, and `deepfilternet3` do not import one
   another and do not import either hybrid.
2. `hybrid_method_1` and `hybrid_method_3` do not import each other.
3. A hybrid core does not import a concrete standalone enhancer. The outer
   orchestrator injects completed arrays.
4. Method-specific alignment, framing, state, and configuration remain inside
   that method. Do not create a shared helper merely because two algorithms
   currently look similar.
5. Compatibility modules may re-export a canonical symbol during a documented
   migration, but they must contain no second implementation.
6. Configuration is explicit. A method owns its algorithm settings; adapter
   selection belongs to orchestration configuration.

The reviewed source hashes after this package-only migration are recorded in
[`method_layout_manifest.json`](method_layout_manifest.json). Historical
validation JSON continues to describe the old paths that produced those runs;
the current Method 1 evaluator applies the manifest when it protects the
canonical Method 3 source.

## Configuration and Tests

The reviewed configuration files are deliberately separate:

| Method | Configuration |
|---|---|
| `original_dsp` | `config/default.yaml` |
| `improved_dsp` | `config/improved_dsp.yaml` |
| `deepfilternet3` | model/runtime options at its wrapper boundary |
| `hybrid_method_1` | `config/hybrid_method_1.yaml` |
| `hybrid_method_3` | `config/hybrid_method_3.yaml` and `config/hybrid_method_3_bands.yaml` |

Unit tests should mirror source ownership under `tests/unit/`. A method owner
runs that method's focused tests while developing. Before integration, run the
whole unit suite plus the dependency-boundary tests. Generated metric files
are evidence, not unit-test fixtures, unless a test explicitly documents a
small frozen fixture.

## Team Change Protocol

Use this ownership register when assigning names. Replace each role label with
the teammate's name in the team tracker; a primary owner should not be the only
reviewer of the same change.

| Area | Primary owner role | Required review role |
|---|---|---|
| `original_dsp` | Original DSP owner | DSP/evaluation reviewer |
| `improved_dsp` | Improved DSP owner | DSP/evaluation reviewer |
| `deepfilternet3` | DL owner | DL/integration reviewer |
| `hybrid_method_1` | Method 1 owner | DSP and DL boundary reviewers |
| `hybrid_method_3` | Method 3 owner | DSP and DL boundary reviewers |
| adapters, datasets, and comparison reports | Evaluation/integration owner | every affected method owner |

The evaluation/integration owner maintains adapters, common dataset handling,
comparison tables, and evidence manifests, but does not edit a method's math
without that method owner's review.

For each pull request:

1. State the canonical method ID being changed.
2. Keep implementation, config, tests, and maintained docs in the same change.
3. List any public input/output contract change.
4. Confirm that unrelated method source files were not changed.
5. If orchestration selects a different DSP, rerun the affected hybrid
   alignment, tuning, metric, runtime, and listening workflow.
6. Record generated evidence outside the source package and link its manifest;
   never commit ad-hoc WAV output into a method's implementation directory.

## Where to Start

- [`architecture.md`](architecture.md) explains signal flow and processing
  responsibilities.
- [`development_workflow.md`](development_workflow.md) explains branches,
  review, formatting, and testing.
- [`evaluation_plan.md`](evaluation_plan.md) explains the objective metrics.
- [`../../hybrid/README.md`](../../hybrid/README.md) indexes the generated
  hybrid evidence available in this workspace.
