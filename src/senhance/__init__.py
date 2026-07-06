"""
senhance: Real-time speech enhancement system.

Implements two enhancement pipelines behind a common interface:
  - Classical DSP (STFT-based spectral subtraction + Wiener filtering)
  - Deep learning (DeepFilterNet), currently offline-only per project scope

See docs/architecture.md for the full system design and rationale.
"""

__version__ = "0.1.0"
