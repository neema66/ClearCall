import sounddevice as sd
import numpy as np


INPUT_DEVICE = None
OUTPUT_DEVICE = None


def callback(
    indata,
    outdata,
    frames,
    time,
    status,
):

    noise = np.random.normal(
        0,
        0.01,
        outdata.shape
    ).astype(
        np.float32
    )

    outdata[:] = noise



print(
    sd.query_devices()
)


with sd.Stream(
    device=(
        INPUT_DEVICE,
        OUTPUT_DEVICE
    ),
    samplerate=48000,
    channels=(2,2),
    dtype="float32",
    callback=callback,
):

    print("Running passthrough")
    print("Ctrl+C to stop")


    while True:
        sd.sleep(1000)