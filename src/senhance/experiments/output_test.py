"""
Simple real-time audio experiment.

Pipeline:

Microphone
    |
    |-- optional artificial noise
    |
    |-- optional causal filter
    |
Speaker


This is intentionally simple:
- no scipy
- no ML
- no STFT
- no buffering

Goal:
Understand the real-time audio pipeline first.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd



class SimpleAudioEngine:

    def __init__(
        self,
        sample_rate=48000,
        block_size=480,
    ):

        self.sample_rate = sample_rate
        self.block_size = block_size


        self.running = False
        self.stream = None


        # -------------------------
        # Noise settings
        # -------------------------

        self.add_noise_enabled = False
        self.noise_only = False

        # dB scale
        self.noise_db = -20



        # -------------------------
        # Filter settings
        # -------------------------

        self.filter_enabled = False

        self.filter_type = "none"


        self.low_cutoff = 300
        self.high_cutoff = 3400



        # State for causal filters

        self.prev_input = 0.0
        self.prev_output = 0.0



    # -------------------------------------------------
    # Noise
    # -------------------------------------------------

    def set_noise_level(self, db):

        self.noise_db = db



    def add_noise(self, audio):

        """
        Convert dB level to amplitude.

        0 dB:
            very loud

        -20 dB:
            noticeable

        -60 dB:
            very quiet
        """

        amplitude = 10 ** (
            self.noise_db / 20
        )


        noise = np.random.normal(
            0,
            amplitude,
            audio.shape,
        )


        return audio + noise.astype(
            np.float32
        )



    # -------------------------------------------------
    # Filters
    # -------------------------------------------------

    def set_filter_type(self, filter_type):

        self.filter_type = filter_type


        self.prev_input = 0.0
        self.prev_output = 0.0



    def set_cutoffs(
        self,
        low,
        high,
    ):

        self.low_cutoff = low
        self.high_cutoff = high



    def apply_filter(self, audio):

        """
        Very simple causal filters.

        These are not production filters.
        They are here to demonstrate:
        
        input[n] -> output[n]

        """

        if self.filter_type == "none":

            return audio



        output = np.zeros_like(audio)



        if self.filter_type == "lowpass":


            # RC low-pass approximation

            dt = 1 / self.sample_rate

            rc = 1 / (
                2*np.pi*self.high_cutoff
            )


            alpha = dt / (
                rc + dt
            )


            previous = self.prev_output


            for i,x in enumerate(audio):

                previous = (
                    previous
                    +
                    alpha*(x-previous)
                )

                output[i] = previous



            self.prev_output = previous



        elif self.filter_type == "highpass":


            dt = 1 / self.sample_rate

            rc = 1 / (
                2*np.pi*self.low_cutoff
            )


            alpha = rc / (
                rc + dt
            )


            previous_y = self.prev_output
            previous_x = self.prev_input


            for i,x in enumerate(audio):

                y = alpha * (
                    previous_y
                    +
                    x
                    -
                    previous_x
                )

                output[i] = y

                previous_y = y
                previous_x = x



            self.prev_output = previous_y
            self.prev_input = previous_x



        elif self.filter_type == "bandpass":

            # Simple bandpass:
            # highpass then lowpass

            temp = audio.copy()


            # highpass

            dt = 1/self.sample_rate

            rc = 1/(2*np.pi*self.low_cutoff)

            alpha = rc/(rc+dt)


            prev_y = 0
            prev_x = self.prev_input


            for i,x in enumerate(temp):

                y = alpha*(prev_y+x-prev_x)

                temp[i]=y

                prev_y=y
                prev_x=x



            # lowpass

            rc = 1/(2*np.pi*self.high_cutoff)

            alpha = dt/(rc+dt)


            prev = 0


            for i,x in enumerate(temp):

                prev = prev + alpha*(x-prev)

                output[i]=prev



        return output.astype(
            np.float32
        )



    # -------------------------------------------------
    # Audio callback
    # -------------------------------------------------

    def callback(
        self,
        indata,
        outdata,
        frames,
        time,
        status,
    ):


        if status:

            print(status)



        audio = indata[:,0]



        if self.noise_only:

            audio = np.zeros_like(
                audio
            )



        if self.add_noise_enabled:

            audio = self.add_noise(
                audio
            )



        if self.filter_enabled:

            audio = self.apply_filter(
                audio
            )



        audio = np.clip(
            audio,
            -1,
            1,
        )


        outdata[:,0] = audio



    # -------------------------------------------------
    # Stream control
    # -------------------------------------------------

    def start(self):

        if self.running:

            return


        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self.callback,
        )


        self.stream.start()


        self.running = True



    def stop(self):

        if not self.running:

            return


        self.stream.stop()

        self.stream.close()

        self.stream = None


        self.running = False