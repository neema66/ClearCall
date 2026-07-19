"""
Simple real-time audio engine.

Pipeline:

Microphone
    |
    v
Add artificial noise
    |
    v
Apply causal IIR filter
    |
    v
Speaker

Filters:
- Low pass
- High pass
- Band pass

Uses scipy.signal.lfilter, which is causal and suitable
for real-time processing.
"""


import numpy as np
import sounddevice as sd

from scipy.signal import butter, lfilter



class SimpleAudioEngine:


    def __init__(self):

        self.sample_rate = 48000

        self.block_size = 480


        self.stream = None


        # Noise

        self.add_noise_enabled = False

        self.noise_only = False

        self.noise_db = -20



        # Filter

        self.filter_type = "none"

        self.low_cutoff = 300

        self.high_cutoff = 3400

        self.filter_order = 4


        # Filter memory

        self.zi = None

        self.b = None

        self.a = None

        self.update_filter()



    # ----------------------------
    # Settings
    # ----------------------------


    def set_noise_level(self, db):

        self.noise_db = db



    def set_filter_type(self, filter_type):

        self.filter_type = filter_type

        self.update_filter()



    def set_cutoffs(self, low, high):

        self.low_cutoff = low

        self.high_cutoff = high

        self.update_filter()



    def set_filter_order(self, order):

        self.filter_order = int(order)

        self.update_filter()



    # ----------------------------
    # Filter creation
    # ----------------------------


    def update_filter(self):

        nyquist = self.sample_rate / 2


        try:

            if self.filter_type == "none":

                self.b = np.array([1.0])

                self.a = np.array([1.0])



            elif self.filter_type == "lowpass":

                cutoff = self.high_cutoff / nyquist

                self.b, self.a = butter(
                    self.filter_order,
                    cutoff,
                    btype="low"
                )



            elif self.filter_type == "highpass":

                cutoff = self.low_cutoff / nyquist

                self.b, self.a = butter(
                    self.filter_order,
                    cutoff,
                    btype="high"
                )



            elif self.filter_type == "bandpass":

                low = self.low_cutoff / nyquist

                high = self.high_cutoff / nyquist


                self.b, self.a = butter(
                    self.filter_order,
                    [
                        low,
                        high
                    ],
                    btype="band"
                )



            self.zi = np.zeros(
                max(
                    len(self.a),
                    len(self.b)
                ) - 1
            )


        except Exception as e:

            print(
                "Filter error:",
                e
            )



    # ----------------------------
    # Audio callback
    # ----------------------------


    def callback(
        self,
        indata,
        outdata,
        frames,
        time,
        status
    ):


        audio = indata[:,0].copy()



        # Add noise

        if self.add_noise_enabled:


            rms = 10 ** (
                self.noise_db / 20
            )


            noise = np.random.normal(
                0,
                rms,
                len(audio)
            )


            if self.noise_only:

                audio = noise

            else:

                audio = audio + noise



        # Apply filter

        if self.filter_type != "none":

            audio, self.zi = lfilter(
                self.b,
                self.a,
                audio,
                zi=self.zi
            )



        outdata[:,0] = audio




    # ----------------------------
    # Stream control
    # ----------------------------


    def start(self):

        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self.callback
        )


        self.stream.start()



    def stop(self):

        if self.stream:

            self.stream.stop()

            self.stream.close()

            self.stream = None