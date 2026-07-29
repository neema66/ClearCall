"""Offline deep-learning enhancement interfaces.

The optional Torch and DeepFilterNet dependencies are imported lazily so
core/base test environments do not require either package or a checkpoint.
"""

from senhance.pipeline.dl.deepfilternet_wrapper import (
    ArrayEnhancer,
    DeepFilterNetModelInfo,
    DeepFilterNetPipeline,
)

__all__ = ["ArrayEnhancer", "DeepFilterNetModelInfo", "DeepFilterNetPipeline"]
