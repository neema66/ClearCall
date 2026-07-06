# Setup Guide

This project targets **Windows** for the live demo (Safe Track scope --
see `docs/architecture.md`). Offline evaluation (no live audio) works on
any OS.

## 1. Prerequisites

- Python 3.10 or later
- Git
- (For live demo only) Windows 10/11
- (For live demo only) [VB-Cable](https://vb-audio.com/Cable/) -- free
  virtual audio cable

## 2. Clone and Set Up a Virtual Environment

```bash
git clone <your-repo-url>
cd speech-enhancement

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux (offline evaluation only, no live demo)
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt

# Install the package itself in editable mode so `senhance` is importable
# from anywhere without manually messing with PYTHONPATH:
pip install -e .
```

If `pip install -e .` isn't working for you yet, the scripts in
`scripts/` add `src/` to the path manually as a fallback, so you can
still run things directly.

## 4. Install VB-Cable (Windows, for the live demo)

1. Download VB-Cable from https://vb-audio.com/Cable/
2. Run the installer (may require a reboot)
3. After installation, you should see "CABLE Input" and "CABLE Output"
   as audio devices in Windows Sound settings
4. In Zoom/Teams, set your **microphone** to "CABLE Output" -- this is
   how the app receives our enhanced audio
5. Our pipeline's **output device** (in `config/default.yaml`,
   `audio.output_device`) should be set to "CABLE Input" -- this is
   where we send the enhanced audio

Run this to see the exact device names on your machine and find the
right one to put in your config:

```bash
python scripts/run_live_demo.py --list-devices
```

## 5. (Optional, for the DL pipeline) Install DeepFilterNet

DeepFilterNet is intentionally **not** in `requirements.txt` because it
pulls in its own PyTorch version constraints that can conflict with the
rest of the project if installed by default. Install it separately:

```bash
pip install deepfilternet
```

TODO (Member 3 / DL lead): confirm the exact package name and version
once you've worked through the DeepFilterNet repo's own install
instructions (https://github.com/Rikorose/DeepFilterNet), and update
this section with the exact command and any gotchas you hit.

## 6. Verify the Install

Run the unit tests -- these don't need any audio hardware or VB-Cable,
so they're a good first check that your environment is set up correctly:

```bash
pytest tests/unit -v
```

You should see all tests passing (11 tests as of this starter codebase).

## 7. Add Sample Data (for evaluation)

See `data/README.md` for how to populate `data/clean/` and `data/noisy/`
with a few test WAV files before running the evaluation scripts.

## 8. Run the Offline Demo

```bash
python scripts/run_offline_demo.py --input data/noisy/sample1.wav --output outputs/sample1_enhanced.wav
```

## 9. Run the Live Demo (Windows only)

Make sure `config/default.yaml` has `audio.output_device` set to your
VB-Cable "CABLE Input" device name (see step 4), then:

```bash
python scripts/run_live_demo.py
```

Speak into your microphone -- the enhanced audio should be routed to
"CABLE Output," which Zoom/Teams will pick up if you've set that as your
microphone in the app's audio settings.

## Troubleshooting

- **"No module named senhance"**: run `pip install -e .` from the repo
  root, or use the scripts in `scripts/` which manually add `src/` to
  the path.
- **Audio glitches/dropouts in the live demo**: check the log output at
  shutdown for "Dropped frames" and "callback overruns" counts -- if
  these are high, see `docs/architecture.md`'s Threading Model section
  for what to investigate.
- **PESQ import errors**: the `pesq` package requires a C compiler to
  build on install. On Windows, installing the
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  usually resolves this.
