"""Optional frequency and temporal smoothing for suppression gains."""

from __future__ import annotations

import numpy as np

from senhance.pipeline.improved_dsp.config import FinalGainSmoothingConfig


class FinalGainSmoother:
    """Regularize per-bin gains while retaining causal stream state."""

    def __init__(
        self,
        num_freq_bins: int,
        minimum_gain: float,
        config: FinalGainSmoothingConfig,
    ) -> None:
        if num_freq_bins <= 0:
            raise ValueError("num_freq_bins must be positive")
        if not 0.0 <= minimum_gain <= 1.0:
            raise ValueError("minimum_gain must satisfy 0 <= value <= 1")

        self.num_freq_bins = num_freq_bins
        self.minimum_gain = minimum_gain
        self.config = config
        self._previous_gain = np.ones(num_freq_bins, dtype=np.float32)

    def apply(self, gain: np.ndarray) -> np.ndarray:
        """Smooth and bound one real gain spectrum."""

        current = np.asarray(gain, dtype=np.float32)
        if current.shape != (self.num_freq_bins,):
            raise ValueError(f"Expected gain shape ({self.num_freq_bins},), got {current.shape}")

        bounded = np.nan_to_num(
            current,
            nan=self.minimum_gain,
            posinf=1.0,
            neginf=self.minimum_gain,
        )
        bounded = np.clip(bounded, self.minimum_gain, 1.0)
        if not self.config.enabled:
            return bounded.astype(np.float32, copy=False)

        frequency_smoothed = bounded.copy()
        if self.num_freq_bins >= 3:
            left, center, right = self.config.frequency_weights
            frequency_smoothed[1:-1] = (
                np.float32(left) * bounded[:-2]
                + np.float32(center) * bounded[1:-1]
                + np.float32(right) * bounded[2:]
            )

        alpha = np.float32(self.config.temporal_smoothing)
        smoothed = alpha * self._previous_gain + (np.float32(1.0) - alpha) * frequency_smoothed
        smoothed = np.clip(smoothed, self.minimum_gain, 1.0).astype(np.float32, copy=False)
        self._previous_gain = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        """Restore the neutral all-pass previous gain."""

        self._previous_gain.fill(1.0)
