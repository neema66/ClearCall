"""
Simple real-time audio engine.

Pipeline:

Input Device 1  -->
                     + (mixed) --> Add artificial noise --> Filter --> Output Device
Input Device 2  -->

Uses separate InputStream(s) and an OutputStream so input and output
devices can be selected independently, and so a second mic input can be
mixed in (or left as None / silence).

Filters:
- Low pass / High pass / Band pass (scipy.signal.lfilter, causal)
- DSP Pipeline (Wiener + Spectral Subtraction, from
  senhance.pipeline.dsp.processor.DSPPipeline)
"""


import numpy as np
import sounddevice as sd

from scipy.signal import butter, lfilter

from senhance.config.settings import AppSettings, load_settings
from senhance.pipeline.dsp.processor import DSPPipeline



class SimpleAudioEngine:


    def __init__(self):

        self.sample_rate = 48000

        self.block_size = 480


        self.input_stream1 = None

        self.input_stream2 = None

        self.output_stream = None


        # ----------------------------
        # Noise
        # ----------------------------

        self.add_noise_enabled = False

        self.noise_only = False

        self.noise_db = -20



        # ----------------------------
        # Filter
        # ----------------------------

        self.filter_type = "none"

        self.low_cutoff = 300

        self.high_cutoff = 3400

        self.filter_order = 4


        # Filter memory

        self.zi = None

        self.b = None

        self.a = None


        # Live DSP strategy (DSPBlockStrategy / ImprovedDSPBlockStrategy
        # wrapping DSPPipeline / ImprovedDSPPipeline). Built lazily on
        # first use since it doesn't need to exist unless one of those
        # filter modes is selected.

        self.live_strategy = None

        self.dsp_settings = None


        self.update_filter()


        # Debug meter throttling (prints ~twice/sec instead of every
        # callback, which fires ~100x/sec at 480 samples @ 48kHz)

        self._debug_counter = 0

        self._debug_every_n = 50



    # ----------------------------
    # Settings
    # ----------------------------


    def set_noise_level(self, db):

        self.noise_db = db



    def set_filter_type(self, filter_type):

        self.filter_type = filter_type

        if filter_type == "dsp_pipeline":

            self._init_dsp_pipeline()

        elif filter_type == "improved_dsp":

            self._init_improved_dsp_pipeline()

        else:

            self.update_filter()



    def set_config_path(self, config_path):

        # Called by the GUI before switching to dsp_pipeline/improved_dsp
        # if the user wants to load a real config/*.yaml instead of the
        # AppSettings() defaults. None means "use defaults".

        self.config_path = config_path



    def set_improved_dsp_config_path(self, path):

        self.improved_dsp_config_path = path



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



    def _init_dsp_pipeline(self):

        # If the GUI called set_config_path(...) with a real
        # config/*.yaml, load it (matches senhance.main's behavior);
        # otherwise fall back to AppSettings()' built-in defaults
        # (frame_size_ms=20, overlap_ratio=0.5 @ 48kHz -> frame_size=960,
        # hop_size=480, which line up with this engine's block_size).

        from senhance.pipeline.dsp.live_strategy import DSPBlockStrategy

        if getattr(self, "config_path", None):

            self.dsp_settings = load_settings(self.config_path)

        else:

            self.dsp_settings = AppSettings()


        self.dsp_settings.audio.sample_rate = self.sample_rate

        self.dsp_settings.audio.block_size = self.block_size


        pipeline = DSPPipeline(self.dsp_settings)

        self._check_hop_alignment(self.dsp_settings)


        # DSPBlockStrategy wraps pipeline.process_block(), which is the
        # correct live entry point -- it builds each STFT analysis frame
        # internally from the previous + current hop via DSPFrameAdapter.
        # (pipeline.process() is for offline whole-frame use only, and
        # calling it here would lock the pipeline's _select_mode() into
        # "frame" mode, conflicting with any block-mode call.)

        self.live_strategy = DSPBlockStrategy(pipeline)



    def _init_improved_dsp_pipeline(self):

        # Mirrors senhance.main's --pipeline improved_dsp handling:
        # ImprovedDSPPipeline needs both AppSettings and its own
        # algorithm config (config/improved_dsp.yaml by default).
        # ImprovedDSPBlockStrategy is the block-mode adapter, matching
        # DSPBlockStrategy's role for DSPPipeline.

        from senhance.pipeline.improved_dsp import (
            ImprovedDSPBlockStrategy,
            ImprovedDSPPipeline,
            load_improved_dsp_config,
        )


        if getattr(self, "config_path", None):

            self.dsp_settings = load_settings(self.config_path)

        else:

            self.dsp_settings = AppSettings()


        self.dsp_settings.audio.sample_rate = self.sample_rate

        self.dsp_settings.audio.block_size = self.block_size


        dsp_config_path = getattr(
            self,
            "improved_dsp_config_path",
            "config/improved_dsp.yaml"
        )

        improved_config = load_improved_dsp_config(dsp_config_path)


        pipeline = ImprovedDSPPipeline(
            self.dsp_settings,
            improved_config
        )

        self._check_hop_alignment(self.dsp_settings)


        self.live_strategy = ImprovedDSPBlockStrategy(pipeline)



    def _check_hop_alignment(self, settings):

        # Same check senhance.main.main() runs before starting the live
        # loop: process_block() expects exactly hop_size samples per
        # call, and this engine always hands it block_size samples, so
        # they must match.

        hop_size = settings.hop_size_samples

        if hop_size != self.block_size:

            print(
                "WARNING: pipeline hop_size (%d) does not match "
                "block_size (%d); frames will be mis-aligned. Adjust "
                "dsp.frame_size_ms / dsp.overlap_ratio in the config, or "
                "the engine's block_size, to match."
                % (hop_size, self.block_size)
            )



    # ----------------------------
    # Debug meter
    # ----------------------------


    def _debug_meter(self, mic_input, output):

        self._debug_counter += 1

        if self._debug_counter % self._debug_every_n != 0:

            return


        mic_rms = float(
            np.sqrt(
                np.mean(mic_input.astype(np.float64) ** 2)
            )
        )

        out_rms = float(
            np.sqrt(
                np.mean(output.astype(np.float64) ** 2)
            )
        )

        print(
            "mic_rms=%.5f  out_rms=%.5f  noise=%s  filter=%s"
            % (
                mic_rms,
                out_rms,
                self.add_noise_enabled,
                self.filter_type,
            )
        )



    # ----------------------------
    # Audio processing
    # ----------------------------


    def process_audio(self, audio):


        # Guard against clipping when two mics are summed

        audio = audio * 0.5


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



        # Filter

        if self.filter_type in ("dsp_pipeline", "improved_dsp"):

            audio = self._process_dsp_pipeline(audio)


        elif self.filter_type != "none":

            audio, self.zi = lfilter(
                self.b,
                self.a,
                audio,
                zi=self.zi
            )


        audio = np.clip(
            audio,
            -1,
            1
        )


        return audio.astype(
            np.float32
        )



    def _process_dsp_pipeline(self, audio):

        # self.live_strategy is a DSPBlockStrategy or
        # ImprovedDSPBlockStrategy, exposing pipeline.process_block() as
        # .process(). process_block() expects exactly hop_size samples
        # (== block_size here) and handles frame-building internally via
        # DSPFrameAdapter, so no manual windowing is needed anymore.

        return self.live_strategy.process(
            audio.astype(np.float32)
        )



    # ----------------------------
    # Stream callbacks
    # ----------------------------


    def input_callback1(
        self,
        indata,
        frames,
        time,
        status
    ):

        if status:

            print("Input 1 status:", status)


        self.input1_data = indata[:, 0].copy()



    def input_callback2(
        self,
        indata,
        frames,
        time,
        status
    ):

        if status:

            print("Input 2 status:", status)


        self.input2_data = indata[:, 0].copy()



    def output_callback(
        self,
        outdata,
        frames,
        time,
        status
    ):

        if status:

            print("Output status:", status)


        try:

            # Mix the two mic inputs together

            mixed = (
                self.input1_data
                +
                self.input2_data
            )


            processed = self.process_audio(
                mixed
            )


            outdata[:, 0] = processed


            self._debug_meter(
                mixed,
                processed
            )


        except Exception as e:

            # Without this, an exception here silently aborts the
            # PortAudio stream -- the GUI keeps saying "Running" but
            # nothing plays and nothing is printed.

            print("Error in output_callback:", repr(e))

            outdata[:, 0] = 0



    def _on_stream_finished(self):

        # PortAudio calls this whenever the output stream stops, whether
        # you called .stop() yourself or it aborted due to an error/device
        # unplug. If you see this print without having clicked Stop, the
        # stream died silently and that's why nothing was playing.

        print("Output stream finished (stopped or aborted).")



    # ----------------------------
    # Stream control
    # ----------------------------


    def start(
        self,
        input_device1=None,
        input_device2=None,
        output_device=None,
    ):


        self.input1_data = np.zeros(
            self.block_size,
            dtype=np.float32
        )


        self.input2_data = np.zeros(
            self.block_size,
            dtype=np.float32
        )


        # Diagnostic: if a device's native default sample rate isn't
        # 48000, WASAPI/PortAudio has to resample under the hood, which
        # can be a source of exactly this kind of subtle robotic/aliased
        # artifact -- especially if the resampling is low quality.

        for label, dev in (
            ("Input 1", input_device1),
            ("Input 2", input_device2),
            ("Output", output_device),
        ):

            if dev is None:

                continue

            info = sd.query_devices(dev)

            print(
                "%s: %s -- native default_samplerate=%s (requesting %s)"
                % (
                    label,
                    info["name"],
                    info["default_samplerate"],
                    self.sample_rate,
                )
            )


        self.input_stream1 = sd.InputStream(

            device=input_device1,

            samplerate=self.sample_rate,

            blocksize=self.block_size,

            channels=1,

            dtype="float32",

            latency="high",

            callback=self.input_callback1
        )



        if input_device2 is not None:

            self.input_stream2 = sd.InputStream(

                device=input_device2,

                samplerate=self.sample_rate,

                blocksize=self.block_size,

                channels=1,

                dtype="float32",

                latency="high",

                callback=self.input_callback2
            )


        else:

            self.input_stream2 = None



        self.output_stream = sd.OutputStream(

            device=output_device,

            samplerate=self.sample_rate,

            blocksize=self.block_size,

            channels=1,

            dtype="float32",

            latency="high",

            callback=self.output_callback,

            finished_callback=self._on_stream_finished,
        )



        self.input_stream1.start()

        if self.input_stream2 is not None:

            self.input_stream2.start()

        self.output_stream.start()



    def stop(self):


        if self.input_stream1:


            self.input_stream1.stop()

            self.input_stream1.close()

            self.input_stream1 = None



        if self.input_stream2:


            self.input_stream2.stop()

            self.input_stream2.close()

            self.input_stream2 = None



        if self.output_stream:


            self.output_stream.stop()

            self.output_stream.close()

            self.output_stream = None