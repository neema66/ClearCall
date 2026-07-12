# Data Directory

This folder holds evaluation audio data. Our current VoiceBank+DEMAND
subset (19 matched clean/noisy pairs, ~17MB) is small enough that it's
committed directly to the repo, so `git pull` is all teammates need to
get it -- no separate download step. If the dataset grows substantially
(e.g. the full VoiceBank+DEMAND test set, ~150MB+), reconsider this and
add `data/` to `.gitignore` instead.

## Structure

```
data/
├── clean/    # Clean (reference) speech WAV files
└── noisy/    # Corresponding noisy versions, same filenames as clean/
```

Files are matched by filename: `data/clean/sample1.wav` should have a
corresponding `data/noisy/sample1.wav` (same speech content, with noise
mixed in).

## Getting Sample Data

**Quick start (manual):** Record or download a few short (5-10 second)
WAV files. For a clean/noisy pair, you can:
1. Record yourself talking in a quiet room -> save as `data/clean/test1.wav`
2. Play back background noise (café sounds, traffic) while re-recording
   the same speech, or mix noise into the clean file with Audacity/`sox`
   -> save as `data/noisy/test1.wav`

**Dataset (recommended for real evaluation):** The Microsoft DNS
Challenge dataset provides thousands of clean/noisy pairs designed
exactly for this kind of evaluation:
https://github.com/microsoft/DNS-Challenge

Download a small subset (not the full multi-GB dataset) for weekly
iteration -- see `scripts/download_sample_data.py` (currently a
placeholder -- TODO for the Evaluation lead to implement).

## Sample Rate

All files should be at the sample rate configured in
`config/default.yaml` under `audio.sample_rate` (48000 Hz by default).
Resample with `sox` or `librosa` if your source files differ.
