"""
DeepFilterNet wrapper.

Project scope note (see docs/architecture.md, "What Changed" section):
For the Week 4 demo, this pipeline runs OFFLINE ONLY -- i.e. it processes
whole audio files for comparison against the DSP pipeline in
docs/evaluation_plan.md, and is not required to run inside the live
audio callback loop under the <40ms latency budget.

It still implements the same EnhancementStrategy interface as
DSPPipeline so that (a) the evaluation script can call both pipelines
identically, and (b) upgrading to live/real-time later (Ambitious Track)
does not require touching any other part of the codebase.

Setup: DeepFilterNet is not in requirements.txt because it pulls in its
own PyTorch/onnxruntime version constraints -- follow docs/setup.md to
install it in a way that doesn't conflict with the rest of the project's
dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from senhance.config.settings import AppSettings
from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy

logger = get_logger(__name__)


class DeepFilterNetPipeline(EnhancementStrategy):
    """
    Wraps a pre-trained DeepFilterNet model for offline batch processing.

    TODO (Member 3 / DL lead): fill in `_load_model` and `process` using
    the DeepFilterNet Python API once installed (see docs/setup.md). The
    structure below shows where each piece belongs; the actual model
    calls are intentionally left as a starter stub so the interface
    compiles and can be swapped in for the DSP pipeline in
    scripts/run_offline_demo.py without any other code changes.
    """

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._model = None
        self._df_state = None  # DeepFilterNet's internal state object, if applicable

        if settings.deep_learning.enabled:
            self._load_model()
        else:
            logger.info(
                "DeepFilterNetPipeline created with deep_learning.enabled=False "
                "in config -- model not loaded. Set config/default.yaml "
                "deep_learning.enabled: true once ready to use this pipeline."
            )

    def _load_model(self) -> None:
        """
        Load the pre-trained DeepFilterNet checkpoint.

        TODO: Implement using the DeepFilterNet package, e.g.:
            from df.enhance import init_df
            self._model, self._df_state, _ = init_df(
                model_base_dir=None  # downloads/caches the default checkpoint
            )
        See docs/setup.md for installation instructions and where
        checkpoints are cached.
        """
        # ============== added===========
        try:
            from df.enhance import init_df
        except ImportError as exc:
            raise ImportError(
                "DeepFilterNet is not installed. Activate the DL env and run: "
                "python -m pip install deepfilternet"
            ) from exc

        logger.info(
            "Loading DeepFilterNet model '%s' on device '%s'",
            self.settings.deep_learning.model_name,
            self.settings.deep_learning.device,
        )
        self._model, self._df_state, _ = init_df()
        # -----------------------------

        # logger.info(
        #     "TODO: load DeepFilterNet model '%s' on device '%s'",
        #     self.settings.deep_learning.model_name,
        #     self.settings.deep_learning.device,
        # )
        # raise NotImplementedError(
        #     "DeepFilterNet model loading not yet implemented. "
        #     "See TODO in DeepFilterNetPipeline._load_model."
        # )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance one frame (or, for offline use, you may prefer to call
        `process_file` on a whole clip instead -- DeepFilterNet's own
        real-time framing conventions may not match our block size).

        TODO: implement using the DeepFilterNet package, e.g.:
            from df.enhance import enhance
            enhanced = enhance(self._model, self._df_state, frame)
        """
        raise NotImplementedError(
            "TODO: implement frame-level DeepFilterNet inference. "
            "For the Safe Track demo, prefer process_file() for whole-clip "
            "offline evaluation instead of frame-by-frame processing."
        )

    def process_file(self, input_path: str | Path, output_path: str | Path) -> None:
        """
        Offline convenience method: enhance a whole audio file at once.
        This is the primary entry point used for the Safe Track demo
        (see scripts/run_offline_demo.py and
        senhance.evaluation.evaluate).

        TODO (Member 3): implement using DeepFilterNet's file-based API,
        e.g.:
            from df.enhance import enhance, load_audio, save_audio
            audio, _ = load_audio(input_path, sr=self._df_state.sr())
            enhanced = enhance(self._model, self._df_state, audio)
            save_audio(output_path, enhanced, self._df_state.sr())
        """
        # ============== added===========
        if self._model is None or self._df_state is None:
            self._load_model()

        from df.enhance import enhance, load_audio, save_audio

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, _ = load_audio(input_path, sr=self._df_state.sr())
        enhanced = enhance(self._model, self._df_state, audio)
        save_audio(output_path, enhanced, self._df_state.sr())

        logger.info("Wrote DeepFilterNet output to %s", output_path)
        # ---------------------------------
        # raise NotImplementedError(
        #     "TODO: implement whole-file DeepFilterNet processing. "
        #     f"input_path={input_path}, output_path={output_path}"
        # )

    def reset(self) -> None:
        """DeepFilterNet's internal state (if any) reset hook."""
        # TODO: reset self._df_state if the model is stateful across calls.
        pass
