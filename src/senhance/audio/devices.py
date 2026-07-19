"""
Audio device discovery and selection.

Automatically selects suitable microphone and speaker devices
for the current computer.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import sounddevice as sd

from senhance.logging_setup.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDevices:
    input_device: int
    output_device: int


def list_devices():
    """Print all available audio devices."""
    devices = sd.query_devices()

    for i, device in enumerate(devices):
        print(
            f"{i}: {device['name']} "
            f"(inputs={device['max_input_channels']}, "
            f"outputs={device['max_output_channels']})"
        )


def get_host_api_name(device_index: int) -> str:
    """
    Return the host API name for a device.
    Example: Windows WASAPI, MME, ALSA.
    """
    device = sd.query_devices(device_index)
    api_index = device["hostapi"]

    return sd.query_hostapis(api_index)["name"]


def find_input_device() -> int:
    """
    Select the best microphone device.
    """

    devices = sd.query_devices()

    candidates = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            api = get_host_api_name(i)

            score = 0

            # Prefer modern low-latency APIs
            if "WASAPI" in api:
                score += 100

            if "WDM-KS" in api:
                score += 150

            # Prefer real microphones over virtual devices
            name = device["name"].lower()

            if "virtual" in name:
                score -= 50

            if "mic" in name or "microphone" in name:
                score += 20

            candidates.append((score, i))

    if not candidates:
        raise RuntimeError("No microphone found")

    candidates.sort(reverse=True)

    selected = candidates[0][1]

    logger.info(
        f"Selected input device: "
        f"{sd.query_devices(selected)['name']}"
    )

    return selected


def find_output_device() -> int:
    """
    Select the best speaker/headphone device.
    """

    devices = sd.query_devices()

    candidates = []

    for i, device in enumerate(devices):
        if device["max_output_channels"] > 0:

            api = get_host_api_name(i)

            score = 0

            if "WASAPI" in api:
                score += 100

            if "WDM-KS" in api:
                score += 150

            name = device["name"].lower()

            # Avoid virtual outputs
            if "virtual" in name:
                score -= 50

            if (
                "speaker" in name
                or "headphone" in name
                or "headset" in name
            ):
                score += 20

            candidates.append((score, i))

    if not candidates:
        raise RuntimeError("No output device found")

    candidates.sort(reverse=True)

    selected = candidates[0][1]

    logger.info(
        f"Selected output device: "
        f"{sd.query_devices(selected)['name']}"
    )

    return selected


def get_best_devices() -> AudioDevices:
    """
    Automatically select the best microphone and speaker.
    """

    return AudioDevices(
        input_device=find_input_device(),
        output_device=find_output_device()
    )


if __name__ == "__main__":

    print("Available devices:")
    list_devices()

    print("\nSelected devices:")

    devices = get_best_devices()

    print(
        "Input:",
        sd.query_devices(devices.input_device)["name"]
    )

    print(
        "Output:",
        sd.query_devices(devices.output_device)["name"]
    )