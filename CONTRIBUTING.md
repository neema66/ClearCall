# Contributing Guide

Git/GitHub workflow rules for our four-person team. The goal is to keep
`main` always in a working, demoable state while letting everyone work
in parallel without stepping on each other.

## Branch Naming Convention

```
feature/<your-name>-<short-description>
fix/<your-name>-<short-description>
docs/<your-name>-<short-description>
```

Examples:
- `feature/salem-deepfilternet-wrapper`
- `fix/ali-callback-overrun`
- `docs/mukun-evaluation-plan`

Keep branches short-lived (a few days at most) to avoid painful merges.

## Commit Message Format

```
<type>: <short summary, imperative mood, under ~72 chars>

<optional longer description explaining why, not just what>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

Examples:
```
feat: add Wiener filter decision-directed SNR estimate

fix: correct STFT overlap-add double-windowing bug

docs: add evaluation plan explaining PESQ vs STOI tradeoffs
```

## Pull Request Rules

- **Every change to `main` goes through a PR** -- no direct pushes to
  `main`, even for "small" fixes.
- PR description should state: what changed, why, and how it was tested
  (e.g. "added unit test," "ran offline demo on sample1.wav and listened
  to output").
- Link any related GitHub Issue in the PR description.
- Keep PRs focused and reasonably small -- one feature/fix per PR, not a
  grab-bag of unrelated changes.

## Code Review Expectations

- **At least one reviewer required** before merging (rotate reviewers
  across the team so everyone stays familiar with the whole codebase,
  not just their own module).
- Reviewers should specifically check:
  - Does this touch the real-time audio callback path
    (`senhance.audio.stream_manager._audio_callback`)? If so, look
    closely for blocking calls, logging, or heavy allocation -- these
    are the highest-risk changes in the whole codebase.
  - Are new parameters hardcoded, or added to `config/default.yaml`
    properly?
  - Is there a unit test for new pipeline logic?
  - Do docstrings explain *why*, not just restate the code?
- Reviews should be constructive and specific -- point to the line and
  suggest an alternative, don't just say "this looks wrong."

## Merge Rules

- Squash-merge preferred, to keep `main`'s history clean and readable.
- Delete the feature branch after merging.
- The person who opened the PR resolves conflicts (rebase onto `main`)
  before requesting final re-review, unless they ask for help.

## Testing Requirements Before Merging

- `pytest tests/unit -v` must pass locally before opening a PR.
- Any new algorithm (DSP or DL) must include at least one unit test
  demonstrating a specific, checkable property of its output (see
  `tests/unit/test_spectral_subtraction.py` for the pattern).
- If your change affects the live audio loop, note in the PR description
  that you tested it live (even informally) -- unit tests alone don't
  catch real-time audio issues.

## Handling Merge Conflicts

1. Rebase your branch onto the latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Resolve conflicts file by file, re-run `pytest tests/unit` after
   resolving.
3. Force-push your branch (`git push --force-with-lease`) and notify
   your reviewer that the PR was rebased.
4. If a conflict involves a design decision (not just overlapping
   lines), raise it in the team chat rather than guessing -- especially
   for anything touching `senhance.pipeline.base.EnhancementStrategy`,
   since that interface affects everyone.

## Issue Tracking Guidelines

- Use GitHub Issues for anything more than a quick fix.
- Label issues with:
  - Type: `bug`, `feature`, `docs`, `integration-risk`
  - Week: `week-1` through `week-4`
  - Owner: `member:1` through `member:4`
- Use a simple Kanban board (To Do / In Progress / Review / Done) so
  everyone can see status at a glance without a stand-up meeting.
- If you hit a design question while implementing (e.g., "should the
  Wiener filter run before or after spectral subtraction?"), open an
  issue to discuss rather than deciding solo and surprising reviewers
  later.

## Documentation Requirements

- Any new module needs a module-level docstring explaining its purpose
  (see any existing file under `src/senhance/` for the expected style).
- Any new public class/function needs a docstring (Google-style, see
  existing code).
- If your change affects setup steps, update `docs/setup.md`.
- If your change affects the architecture or scope decisions, update
  `docs/architecture.md` -- especially the Known Limitations and
  Checkpoints sections, which should stay accurate as the project
  evolves.
