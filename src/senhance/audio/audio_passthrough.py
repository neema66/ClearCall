import sounddevice as sd

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    outdata[:] = indata


with sd.Stream(
    samplerate=48000,
    channels=1,
    blocksize=480,
    device=(14, 12),
    callback=callback
) as stream:
    print("Running...")
    print("Latency:", stream.latency)
    input("Press Enter to stop\n")