# Development Workflow

See `CONTRIBUTING.md` in the repo root for the full Git/PR workflow
rules. This document covers the day-to-day practices for working in
this codebase specifically.

## Coding Standards

- **Type hints** on all function signatures (parameters and return
  types). This project uses `from __future__ import annotations`
  throughout, so type hints are cheap to add and don't cause circular
  import issues.
- **Docstrings** (Google-style, as used throughout the starter code) on
  every public class and function. Explain *why*, not just *what*, when
  a design decision isn't obvious from the code alone.
- **Logging, not print statements** -- use `senhance.logging_setup.logger.get_logger(__name__)`
  everywhere. The one exception is the real-time audio callback, which
  should not log synchronously at all (see `docs/architecture.md`,
  Threading Model).
- **No hardcoded parameters** -- anything tunable (sample rate, frame
  size, thresholds, file paths) belongs in `config/default.yaml`, read
  through `AppSettings`. If you find yourself typing a literal number
  into pipeline code, stop and add it to the config instead.
- **Formatting/linting**: run `black .` and `ruff check .` before
  committing. Config for both lives in `pyproject.toml`.

## Working in Parallel (Four-Person Team)

The codebase is organized so each team member's primary files rarely
overlap with anyone else's:

| Member | Primary files |
|---|---|
| Audio I/O & Integration | `src/senhance/audio/*`, `src/senhance/main.py`, `scripts/run_live_demo.py` |
| Classical DSP | `src/senhance/pipeline/dsp/*` |
| Deep Learning | `src/senhance/pipeline/dl/*` |
| Evaluation & Report | `src/senhance/evaluation/*`, `docs/*`, `data/*` |

The main shared contract everyone codes against is
`senhance.pipeline.base.EnhancementStrategy` -- if you need to change
this interface, flag it to the whole team first (see CONTRIBUTING.md's
PR rules), since it affects every pipeline implementation.

## Running Tests Locally

```bash
pytest tests/unit -v
```

Run this before every commit that touches `src/`. All new pipeline
logic (DSP or DL) should come with at least one unit test -- see
`tests/unit/test_spectral_subtraction.py` or `test_wiener_filter.py` for
the pattern (construct known inputs, assert a specific property of the
output, like "gain never exceeds 1.0").

## Weekly Evaluation Routine

Per the Safe Track scope (see `docs/architecture.md`), evaluation is a
manual step, not automated CI. At the end of each week:

```bash
python -m senhance.evaluation.evaluate --pipeline dsp
python -m senhance.evaluation.benchmark_latency --pipeline dsp
```

Save the console output (or redirect to a file under `outputs/`) so you
have a record of how metrics changed week over week -- this becomes the
evidence for your final report's Results section.

## Adding a New Config Parameter

1. Add the field (with a sensible default) to the relevant dataclass in
   `src/senhance/config/settings.py`.
2. Add the corresponding key to `config/default.yaml` with a comment
   explaining what it does.
3. Add or update a test in `tests/unit/test_config.py` if the parameter
   affects a derived value (like `frame_size_samples`).

## Debugging the Live Loop

Start with `python scripts/run_live_demo.py --list-devices` to confirm
your input/output devices are configured correctly before assuming
there's a bug in the enhancement code. Most early live-demo issues are
device misconfiguration, not algorithm bugs.
