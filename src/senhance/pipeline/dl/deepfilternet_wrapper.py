"""Strict offline array boundary for DeepFilterNet.

DeepFilterNet remains offline-only in the current project scope.  The public
``enhance_array`` method is the boundary used by evaluation and future hybrid
pipelines; frame-level streaming is deliberately not implemented here.

Torch and DeepFilterNet are imported only by the production backend.  Tests
and higher-level pipelines can inject an :class:`ArrayEnhancer` without those
optional packages being installed.
"""

from __future__ import annotations

import os
import re
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from senhance.config.settings import AppSettings
from senhance.logging_setup.logger import get_logger
from senhance.pipeline.base import EnhancementStrategy

logger = get_logger(__name__)

_CANONICAL_SAMPLE_RATE = 48_000
_DF_GLOBAL_LOCK = threading.RLock()
_ACTIVE_DF_BACKEND: weakref.ReferenceType[object] | None = None
_DF_BACKEND_LOADING = False


@dataclass(frozen=True)
class DeepFilterNetModelInfo:
    """Observed, read-only information about the loaded enhancement model."""

    requested_model_name: str
    loaded_model_name: str
    model_base_dir: str | None
    effective_device: str
    sample_rate: int
    fft_size: int
    hop_size: int
    inference_pad: bool
    output_delay_compensated: bool
    delay_compensation_samples: int
    model_pad_mode: str | None = None


class ArrayEnhancer(Protocol):
    """Minimal backend contract accepted by :class:`DeepFilterNetPipeline`."""

    @property
    def model_info(self) -> DeepFilterNetModelInfo:
        """Return metadata for the loaded model/backend."""

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """Enhance one contiguous mono float32 array at the model sample rate."""


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    """Temporarily set one environment variable and restore it exactly."""

    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _normalize_device_spec(device: str) -> str:
    """Accept only CPU or an unambiguous non-negative CUDA device index."""

    normalized = device.strip().lower()
    match = re.fullmatch(r"cpu|cuda(?::([0-9]+))?", normalized)
    if match is None:
        raise ValueError(
            "deep_learning.device must be 'cpu', 'cuda', or 'cuda:<index>', "
            f"got {device!r}"
        )
    index = match.group(1)
    if index is not None:
        normalized = f"cuda:{int(index)}"
    return normalized


def _reserve_production_backend() -> None:
    """Reserve DeepFilterNet's process-global configuration for one backend."""

    global _ACTIVE_DF_BACKEND, _DF_BACKEND_LOADING
    with _DF_GLOBAL_LOCK:
        active = _ACTIVE_DF_BACKEND() if _ACTIVE_DF_BACKEND is not None else None
        if active is not None or _DF_BACKEND_LOADING:
            raise RuntimeError(
                "Only one production DeepFilterNet backend may be active per process; "
                "reuse one DeepFilterNetPipeline or inject a fake backend"
            )
        _ACTIVE_DF_BACKEND = None
        _DF_BACKEND_LOADING = True


def _release_production_backend_reservation() -> None:
    global _DF_BACKEND_LOADING
    with _DF_GLOBAL_LOCK:
        _DF_BACKEND_LOADING = False


def _register_production_backend(backend: object) -> None:
    global _ACTIVE_DF_BACKEND, _DF_BACKEND_LOADING
    with _DF_GLOBAL_LOCK:
        _ACTIVE_DF_BACKEND = weakref.ref(backend)
        _DF_BACKEND_LOADING = False


class _DeepFilterNetArrayEnhancer:
    """Lazy-importing adapter around DeepFilterNet 0.5.x's array API."""

    def __init__(self, model_name: str, device: str) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("deep_learning.model_name must not be empty")
        device = _normalize_device_spec(device)

        _reserve_production_backend()
        try:
            self._initialize(model_name, device)
        except BaseException:
            _release_production_backend_reservation()
            raise
        _register_production_backend(self)

    def _initialize(self, model_name: str, device: str) -> None:
        try:
            import torch
            from df.config import config as df_config
            from df.enhance import enhance, init_df
            from df.model import ModelParams
        except ImportError as exc:
            raise ImportError(
                "DeepFilterNet is not installed. Activate the DL environment and run "
                "the installation steps in docs/setup.md."
            ) from exc

        device_type, separator, raw_index = device.partition(":")
        requested_index = int(raw_index) if separator else None
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"DeepFilterNet device {device!r} was requested but CUDA is unavailable"
                )
            if (
                requested_index is not None
                and requested_index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"DeepFilterNet device {device!r} does not exist; "
                    f"available CUDA device count is {torch.cuda.device_count()}"
                )
        requested_device = torch.device(device)

        logger.info(
            "Loading DeepFilterNet model '%s' on requested device '%s'",
            model_name,
            device,
        )

        # DeepFilterNet 0.5.x has no device argument. Its get_device() reads
        # the process-wide DEVICE variable/config both while loading and while
        # enhancing. Serializing these operations makes the configured device
        # effective and avoids leaking the temporary environment mutation.
        with _DF_GLOBAL_LOCK, _temporary_environment("DEVICE", str(requested_device)):
            model, df_state, loaded_model_name = init_df(
                model_base_dir=model_name,
                default_model=model_name,
                log_file=None,
            )

        first_parameter = next(model.parameters(), None)
        if first_parameter is None:
            raise RuntimeError("Loaded DeepFilterNet model has no parameters")
        actual_device = first_parameter.device
        if actual_device.type != requested_device.type or (
            requested_device.index is not None
            and actual_device.index != requested_device.index
        ):
            raise RuntimeError(
                "DeepFilterNet did not honor the configured device: "
                f"requested={requested_device}, actual={actual_device}"
            )

        sample_rate = int(df_state.sr())
        fft_size = int(df_state.fft_size())
        hop_size = int(df_state.hop_size())
        params = ModelParams()
        config_path = Path(df_config.path).resolve() if df_config.path else None

        self._torch = torch
        self._df_config = df_config
        self._enhance = enhance
        self._model = model
        self._df_state = df_state
        self._model_info = DeepFilterNetModelInfo(
            requested_model_name=model_name,
            loaded_model_name=str(loaded_model_name),
            model_base_dir=str(config_path.parent) if config_path is not None else None,
            effective_device=str(actual_device),
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_size=hop_size,
            inference_pad=True,
            output_delay_compensated=True,
            delay_compensation_samples=fft_size - hop_size,
            model_pad_mode=str(params.pad_mode),
        )

        # Persist the verified device in DeepFilterNet's loaded configuration
        # for calls made after the temporary environment has been restored.
        with _DF_GLOBAL_LOCK:
            self._df_config.set("DEVICE", str(actual_device), str, "train")

    @property
    def model_info(self) -> DeepFilterNetModelInfo:
        return self._model_info

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(audio).unsqueeze(0)
        with _DF_GLOBAL_LOCK, _temporary_environment(
            "DEVICE", self._model_info.effective_device
        ):
            self._df_config.set(
                "DEVICE", self._model_info.effective_device, str, "train"
            )
            enhanced = self._enhance(
                self._model,
                self._df_state,
                tensor,
                pad=self._model_info.inference_pad,
            )
        return enhanced[0].detach().cpu().numpy()


def _create_deepfilternet_enhancer(model_name: str, device: str) -> ArrayEnhancer:
    """Create the production backend; kept separate for dependency injection."""

    return _DeepFilterNetArrayEnhancer(model_name, device)


class DeepFilterNetPipeline(EnhancementStrategy):
    """Validated, reusable, whole-array DeepFilterNet pipeline.

    ``deep_learning.enabled`` controls eager loading only. Offline callers may
    construct the pipeline with it disabled and load lazily on the first array.
    An injected enhancer always bypasses production model loading.
    """

    def __init__(
        self,
        settings: AppSettings,
        enhancer: ArrayEnhancer | None = None,
    ) -> None:
        self.settings = settings
        self._enhancer = enhancer

        # Compatibility attributes for older diagnostic code. New code must
        # use model_info rather than reaching into DeepFilterNet internals.
        self._model = getattr(enhancer, "_model", None)
        self._df_state = getattr(enhancer, "_df_state", None)

        if enhancer is not None:
            self._validate_model_info(enhancer.model_info)
        elif settings.deep_learning.enabled:
            self._load_model()
        else:
            logger.info(
                "DeepFilterNet model loading deferred; enhance_array() will "
                "load it on first offline use"
            )

    def _load_model(self) -> None:
        """Load the selected production model exactly once."""

        if self._enhancer is not None:
            return
        enhancer = _create_deepfilternet_enhancer(
            self.settings.deep_learning.model_name,
            self.settings.deep_learning.device,
        )
        self._validate_model_info(enhancer.model_info)
        self._enhancer = enhancer
        self._model = getattr(enhancer, "_model", None)
        self._df_state = getattr(enhancer, "_df_state", None)

    def _ensure_enhancer(self) -> ArrayEnhancer:
        if self._enhancer is None:
            self._load_model()
        if self._enhancer is None:  # Defensive guard for custom factories.
            raise RuntimeError("DeepFilterNet enhancer failed to initialize")
        return self._enhancer

    def _validate_model_info(self, info: DeepFilterNetModelInfo) -> None:
        if not isinstance(info, DeepFilterNetModelInfo):
            raise TypeError("enhancer.model_info must be DeepFilterNetModelInfo")
        if info.sample_rate != _CANONICAL_SAMPLE_RATE:
            raise ValueError(
                "DeepFilterNet model sample rate must be 48000 Hz, "
                f"got {info.sample_rate}"
            )
        if self.settings.audio.sample_rate != info.sample_rate:
            raise ValueError(
                "Project and DeepFilterNet sample rates differ: "
                f"project={self.settings.audio.sample_rate}, model={info.sample_rate}"
            )
        if info.fft_size <= 0 or info.hop_size <= 0 or info.hop_size > info.fft_size:
            raise ValueError(
                "Invalid DeepFilterNet geometry: "
                f"fft_size={info.fft_size}, hop_size={info.hop_size}"
            )
        expected_compensation = info.fft_size - info.hop_size
        if not info.inference_pad or not info.output_delay_compensated:
            raise ValueError(
                "DeepFilterNet backend must use pad=True output-delay compensation"
            )
        if info.delay_compensation_samples != expected_compensation:
            raise ValueError(
                "DeepFilterNet delay compensation does not match fft_size - hop_size: "
                f"expected={expected_compensation}, "
                f"actual={info.delay_compensation_samples}"
            )

    @property
    def model_info(self) -> DeepFilterNetModelInfo:
        """Load if needed and expose the actual model geometry and behavior."""

        return self._ensure_enhancer().model_info

    def enhance_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance a complete mono array without file-I/O quantization.

        Valid floating input is copied into contiguous float32 storage before
        inference. The backend must return a finite mono floating array with
        exactly the original number of samples; this method never resamples,
        normalizes, clips, pads, or trims a backend result.
        """

        if not isinstance(audio, np.ndarray):
            raise TypeError("audio must be a numpy.ndarray")
        if audio.ndim != 1:
            raise ValueError(
                f"audio must be a one-dimensional mono array, got shape {audio.shape}"
            )
        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError(f"audio must have a real floating dtype, got {audio.dtype}")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains NaN or infinite samples")

        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate, (int, np.integer)
        ):
            raise TypeError("sample_rate must be an integer")
        if int(sample_rate) != _CANONICAL_SAMPLE_RATE:
            raise ValueError(
                f"audio sample rate must be {_CANONICAL_SAMPLE_RATE} Hz, "
                f"got {sample_rate}"
            )

        enhancer = self._ensure_enhancer()
        expected_sample_rate = enhancer.model_info.sample_rate
        if int(sample_rate) != expected_sample_rate:
            raise ValueError(
                "audio sample rate does not match the loaded model: "
                f"audio={sample_rate}, model={expected_sample_rate}"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            model_input = np.array(audio, dtype=np.float32, order="C", copy=True)
        if not np.all(np.isfinite(model_input)):
            raise ValueError("audio contains values outside the float32 finite range")
        if model_input.size == 0:
            return model_input

        enhanced = enhancer.enhance(model_input)
        if not isinstance(enhanced, np.ndarray):
            raise TypeError("enhancer output must be a numpy.ndarray")
        if enhanced.ndim != 1:
            raise ValueError(
                "enhancer output must be a one-dimensional mono array, "
                f"got shape {enhanced.shape}"
            )
        if not np.issubdtype(enhanced.dtype, np.floating):
            raise TypeError(
                f"enhancer output must have a real floating dtype, got {enhanced.dtype}"
            )
        if enhanced.shape != model_input.shape:
            raise ValueError(
                "enhancer output length must equal input length: "
                f"input={model_input.size}, output={enhanced.size}"
            )
        if not np.all(np.isfinite(enhanced)):
            raise ValueError("enhancer output contains NaN or infinite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            output = np.array(enhanced, dtype=np.float32, order="C", copy=True)
        if not np.all(np.isfinite(output)):
            raise ValueError(
                "enhancer output contains values outside the float32 finite range"
            )
        return output

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Frame-level streaming is outside the Milestone 1 offline scope."""

        raise NotImplementedError(
            "DeepFilterNetPipeline is whole-array/offline only; use "
            "enhance_array(audio, sample_rate). A stateful streaming API is "
            "required before using DeepFilterNet in the live callback."
        )

    def process_file(self, input_path: str | Path, output_path: str | Path) -> None:
        """Read one mono WAV, delegate to ``enhance_array``, and write float WAV."""

        import soundfile as sf

        input_path = Path(input_path)
        output_path = Path(output_path)
        audio, sample_rate = sf.read(input_path, dtype="float32", always_2d=False)
        enhanced = self.enhance_array(audio, sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, enhanced, sample_rate, subtype="FLOAT")
        logger.info("Wrote DeepFilterNet float output to %s", output_path)

    def reset(self) -> None:
        """Whole-array inference resets DeepFilterNet recurrent state per call."""

        return None
