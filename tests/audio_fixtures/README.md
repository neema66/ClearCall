# Test Audio Fixtures

Place small (a few seconds) WAV files here for use in unit/integration
tests that need real audio rather than synthetic noise, e.g. testing
that the DSP pipeline doesn't crash on real speech, or spot-checking
output quality by ear.

Keep files small (a few hundred KB each) -- this directory is meant for
quick test fixtures, not the evaluation dataset (which lives in `data/`
and is gitignored due to size, see data/README.md).

TODO: add 2-3 short sample clips here (e.g. one clean speech clip, one
noisy clip, one silence/edge-case clip) once available.
