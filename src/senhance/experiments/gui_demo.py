"""
GUI for ClearCall DSP experiment.

Controls:
- Start/Stop audio
- Input/output device selection
- Artificial noise level in dB
- Noise only mode
- Filter selection
- Cutoff frequencies
- Filter order
"""

import sys
import sounddevice as sd


from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QScrollArea,
)

from PySide6.QtCore import Qt, QThread, Signal


from senhance.experiments.simple_filter import SimpleAudioEngine



class FocusLineEdit(QLineEdit):

    def __init__(self, callback, text=""):

        super().__init__(text)

        self.callback = callback



    def keyPressEvent(self, event):

        if event.key() in (
            Qt.Key_Return,
            Qt.Key_Enter
        ):

            self.callback()

            self.clearFocus()

        else:

            super().keyPressEvent(event)



    def focusOutEvent(self, event):

        self.callback()

        super().focusOutEvent(event)




class _BackgroundFileLoader(QThread):

    # Loads/resamples a background noise file off the GUI thread --
    # reading a long WAV and resample_poly()'ing it to the engine's
    # sample rate can take a noticeable moment, and doing that on the
    # main thread froze the whole window until it finished.

    loaded = Signal(str)

    failed = Signal(str, str)


    def __init__(self, engine, path):

        super().__init__()

        self.engine = engine

        self.path = path



    def run(self):

        try:

            self.engine.set_background_file(
                self.path
            )

        except Exception as e:

            import traceback

            self.failed.emit(
                self.path,
                "%s\n%s" % (e, traceback.format_exc())
            )

            return


        self.loaded.emit(
            self.path
        )




class ClearCallGUI(QWidget):

    def __init__(
        self,
        initial_config_path=None,
        initial_dsp_config_path=None,
        initial_pipeline=None,
    ):

        super().__init__()


        self.engine = SimpleAudioEngine()

        self._loader_thread = None


        self._initial_config_path = initial_config_path or ""

        self._initial_dsp_config_path = (
            initial_dsp_config_path or "config/improved_dsp.yaml"
        )

        self._initial_pipeline = initial_pipeline


        self.setWindowTitle(
            "ClearCall DSP Demo -- Real-Time Mic to Speaker"
        )


        layout = QVBoxLayout()



        # ----------------------
        # Status
        # ----------------------

        self.status = QLabel(
            "Stopped"
        )

        layout.addWidget(
            self.status
        )



        # ----------------------
        # Audio Devices
        # ----------------------

        layout.addWidget(
            QLabel(
                "Input Device 1"
            )
        )


        self.input_box = QComboBox()

        layout.addWidget(
            self.input_box
        )



        layout.addWidget(
            QLabel(
                "Input Device 2"
            )
        )


        self.input_box2 = QComboBox()

        layout.addWidget(
            self.input_box2
        )



        layout.addWidget(
            QLabel(
                "Output Device"
            )
        )


        self.output_box = QComboBox()

        layout.addWidget(
            self.output_box
        )


        self.input_devices = []

        self.output_devices = []


        self.populate_devices()



        # ----------------------
        # Pipeline config paths (for DSP Pipeline / Improved DSP modes)
        # ----------------------

        layout.addWidget(
            QLabel(
                "Config File (optional -- blank uses AppSettings defaults)"
            )
        )

        self.config_path_input = FocusLineEdit(
            self.update_config_path,
            ""
        )

        layout.addWidget(
            self.config_path_input
        )


        layout.addWidget(
            QLabel(
                "Improved DSP Config (used only for Improved DSP Pipeline mode)"
            )
        )

        self.improved_dsp_config_input = FocusLineEdit(
            self.update_improved_dsp_config_path,
            "config/improved_dsp.yaml"
        )

        layout.addWidget(
            self.improved_dsp_config_input
        )


        list_devices_button = QPushButton(
            "List Devices (prints to console, like --list-devices)"
        )

        list_devices_button.clicked.connect(
            self.list_devices
        )

        layout.addWidget(
            list_devices_button
        )


        offline_info_button = QPushButton(
            "About Offline-Only Methods (dl / hybrid_method_1 / hybrid_method_3)"
        )

        offline_info_button.clicked.connect(
            self.show_offline_methods_info
        )

        layout.addWidget(
            offline_info_button
        )



        # ----------------------
        # Audio buttons
        # ----------------------

        start_button = QPushButton(
            "Start Audio"
        )

        start_button.clicked.connect(
            self.start_audio
        )


        layout.addWidget(
            start_button
        )



        stop_button = QPushButton(
            "Stop Audio"
        )

        stop_button.clicked.connect(
            self.stop_audio
        )


        layout.addWidget(
            stop_button
        )



        # ----------------------
        # Noise
        # ----------------------

        self.noise_checkbox = QCheckBox(
            "Add Artificial Noise"
        )

        self.noise_checkbox.toggled.connect(
            self.noise_changed
        )

        layout.addWidget(
            self.noise_checkbox
        )



        self.noise_only_checkbox = QCheckBox(
            "Noise Only"
        )


        self.noise_only_checkbox.toggled.connect(
            self.noise_only_changed
        )


        layout.addWidget(
            self.noise_only_checkbox
        )



        layout.addWidget(
            QLabel(
                "Noise Level (dB)"
            )
        )


        self.noise_input = FocusLineEdit(
            self.update_noise,
            "-20"
        )


        layout.addWidget(
            self.noise_input
        )


        layout.addWidget(
            QLabel(
                "Noise Type"
            )
        )


        self.noise_type_box = QComboBox()

        self.noise_type_box.addItems(
            [
                "White",
                "Pink",
                "Brown",
                "Background Audio File...",
            ]
        )

        self.noise_type_box.currentTextChanged.connect(
            self.noise_type_changed
        )

        layout.addWidget(
            self.noise_type_box
        )


        # Only shown/used when "Background Audio File..." is selected.
        # Point this at any looping ambience recording -- coffee shop
        # chatter, street noise, office hum, etc -- and it gets mixed
        # in at the same Noise Level (dB) slider above.

        self.background_label = QLabel(
            "Background Audio File (looped; e.g. a coffee shop "
            "ambience recording)"
        )

        layout.addWidget(
            self.background_label
        )

        background_row = QHBoxLayout()

        self.background_path_input = QLineEdit()

        self.background_path_input.setPlaceholderText(
            "path/to/coffee_shop.wav"
        )

        background_row.addWidget(
            self.background_path_input
        )

        self.background_browse_button = QPushButton(
            "Browse..."
        )

        self.background_browse_button.clicked.connect(
            self.browse_background_file
        )

        background_row.addWidget(
            self.background_browse_button
        )

        layout.addLayout(
            background_row
        )

        self.background_label.hide()

        self.background_path_input.hide()

        self.background_browse_button.hide()



        # ----------------------
        # Filter type
        # ----------------------

        layout.addWidget(
            QLabel(
                "Filter Type"
            )
        )


        self.filter_box = QComboBox()


        self.filter_box.addItems(
            [
                "None",
                "Low Pass",
                "High Pass",
                "Band Pass",
                "DSP Pipeline (Wiener + Spectral Subtraction)",
                "Improved DSP Pipeline (Live)",
            ]
        )


        self.filter_box.currentTextChanged.connect(
            self.filter_changed
        )


        layout.addWidget(
            self.filter_box
        )



        # ----------------------
        # Frequency controls
        # ----------------------

        self.low_label = QLabel(
            "Low cutoff frequency (Hz)"
        )


        self.low_input = FocusLineEdit(
            self.update_cutoffs,
            "300"
        )



        self.high_label = QLabel(
            "High cutoff frequency (Hz)"
        )


        self.high_input = FocusLineEdit(
            self.update_cutoffs,
            "3400"
        )


        layout.addWidget(
            self.low_label
        )

        layout.addWidget(
            self.low_input
        )


        layout.addWidget(
            self.high_label
        )

        layout.addWidget(
            self.high_input
        )



        # ----------------------
        # Filter order
        # ----------------------

        layout.addWidget(
            QLabel(
                "Filter order"
            )
        )


        self.order_input = FocusLineEdit(
            self.update_order,
            "4"
        )


        layout.addWidget(
            self.order_input
        )



        # Wrap everything in a scroll area so the window is usable
        # (and all controls reachable) even when it's taller than the
        # screen -- there are a lot of controls stacked in one column.

        content = QWidget()

        content.setLayout(
            layout
        )

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        scroll.setWidget(
            content
        )

        outer_layout = QVBoxLayout()

        outer_layout.setContentsMargins(0, 0, 0, 0)

        outer_layout.addWidget(
            scroll
        )

        self.setLayout(
            outer_layout
        )

        self.resize(520, 800)


        # Apply any initial values passed in from the launcher window
        # (mirrors --config / --dsp-config / --pipeline from the old CLI).

        if self._initial_config_path:

            self.config_path_input.setText(
                self._initial_config_path
            )

            self.engine.set_config_path(
                self._initial_config_path
            )


        self.improved_dsp_config_input.setText(
            self._initial_dsp_config_path
        )

        self.engine.set_improved_dsp_config_path(
            self._initial_dsp_config_path
        )


        pipeline_to_label = {

            "original_dsp": "DSP Pipeline (Wiener + Spectral Subtraction)",

            "improved_dsp": "Improved DSP Pipeline (Live)",

        }

        initial_label = pipeline_to_label.get(
            self._initial_pipeline
        )

        if initial_label:

            self.filter_box.setCurrentText(
                initial_label
            )


        self.update_frequency_visibility()



    # ======================
    # Device loading
    # ======================

    def populate_devices(self):

        devices = sd.query_devices()


        default_input, default_output = sd.default.device


        host_apis = sd.query_hostapis()


        input_index = 0
        output_index = 0


        # First item lets the user disable the second input entirely

        self.input_box2.addItem(
            "None"
        )


        for i, device in enumerate(devices):


            if device["max_input_channels"] > 0:

                self.input_devices.append(i)

                api_name = host_apis[
                    device["hostapi"]
                ]["name"]

                label = "%s (%s)" % (
                    device["name"],
                    api_name,
                )

                self.input_box.addItem(
                    label
                )

                self.input_box2.addItem(
                    label
                )


                if i == default_input:

                    input_index = (
                        self.input_box.count()-1
                    )

                elif (
                    "WASAPI" in api_name
                    and input_index == 0
                ):

                    input_index = (
                        self.input_box.count()-1
                    )



            if device["max_output_channels"] > 0:

                self.output_devices.append(i)

                api_name = host_apis[
                    device["hostapi"]
                ]["name"]

                label = "%s (%s)" % (
                    device["name"],
                    api_name,
                )

                self.output_box.addItem(
                    label
                )


                if i == default_output:

                    output_index = (
                        self.output_box.count()-1
                    )

                elif (
                    "WASAPI" in api_name
                    and output_index == 0
                ):

                    output_index = (
                        self.output_box.count()-1
                    )



        self.input_box.setCurrentIndex(
            input_index
        )


        # Default input 2 to "None" until the user picks one

        self.input_box2.setCurrentIndex(
            0
        )


        self.output_box.setCurrentIndex(
            output_index
        )



    # ======================
    # Audio
    # ======================

    def start_audio(self):

        input_device1 = self.input_devices[
            self.input_box.currentIndex()
        ]


        if self.input_box2.currentIndex() == 0:

            input_device2 = None

        else:

            input_device2 = self.input_devices[
                self.input_box2.currentIndex() - 1
            ]


        output_device = self.output_devices[
            self.output_box.currentIndex()
        ]



        print(
            "Starting audio: input1=%r input2=%r output=%r"
            % (
                self.input_box.currentText(),
                (
                    "None"
                    if input_device2 is None
                    else self.input_box2.currentText()
                ),
                self.output_box.currentText(),
            )
        )


        try:

            self.engine.start(
                input_device1=input_device1,
                input_device2=input_device2,
                output_device=output_device
            )


            self.status.setText(
                "Running"
            )


        except Exception as e:

            import traceback

            traceback.print_exc()


            self.status.setText(
                "Failed to start: %s" % e
            )



    def stop_audio(self):

        self.engine.stop()


        self.status.setText(
            "Stopped"
        )



    # ======================
    # Noise
    # ======================

    def noise_changed(self, checked):

        self.engine.add_noise_enabled = checked

        print("add_noise_enabled ->", checked)



    def noise_only_changed(self, checked):

        self.engine.noise_only = checked

        print("noise_only ->", checked)



    def update_noise(self):

        try:

            self.engine.set_noise_level(
                float(
                    self.noise_input.text()
                )
            )

        except ValueError:

            pass



    def noise_type_changed(self, text):

        types = {

            "White": "white",

            "Pink": "pink",

            "Brown": "brown",

            "Background Audio File...": "background",

        }

        noise_type = types.get(
            text,
            "white"
        )

        self.engine.set_noise_type(
            noise_type
        )


        is_background = (
            noise_type == "background"
        )

        self.background_label.setVisible(
            is_background
        )

        self.background_path_input.setVisible(
            is_background
        )

        self.background_browse_button.setVisible(
            is_background
        )


        if is_background:

            path = self.background_path_input.text().strip()

            if path:

                self._load_background_file(
                    path
                )



    def browse_background_file(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a background audio file (e.g. coffee shop ambience)",
            "",
            "Audio files (*.wav *.flac *.ogg *.aiff *.aif);;All files (*)"
        )

        if not path:

            return


        self.background_path_input.setText(
            path
        )

        self._load_background_file(
            path
        )



    def _load_background_file(self, path):

        # Non-blocking: reading + resampling a long file can take a
        # moment, so it runs on a worker thread instead of freezing
        # the GUI. Disable the controls until it finishes so a second
        # load can't be kicked off on top of the first.

        if self._loader_thread is not None and self._loader_thread.isRunning():

            return


        self.background_browse_button.setEnabled(False)

        self.background_path_input.setEnabled(False)

        self.background_label.setText(
            "Background Audio File (loading %r...)" % path
        )


        self._loader_thread = _BackgroundFileLoader(
            self.engine,
            path
        )

        self._loader_thread.loaded.connect(
            self._background_file_loaded
        )

        self._loader_thread.failed.connect(
            self._background_file_failed
        )

        self._loader_thread.start()



    def _background_file_loaded(self, path):

        self.background_browse_button.setEnabled(True)

        self.background_path_input.setEnabled(True)

        self.background_label.setText(
            "Background Audio File (looped; e.g. a coffee shop "
            "ambience recording)"
        )



    def _background_file_failed(self, path, error_text):

        self.background_browse_button.setEnabled(True)

        self.background_path_input.setEnabled(True)

        self.background_label.setText(
            "Background Audio File (looped; e.g. a coffee shop "
            "ambience recording)"
        )

        QMessageBox.critical(
            self,
            "Failed to load background audio",
            "Could not load %r as background noise:\n%s\n\n"
            "(Non-WAV formats like mp3/ogg need the 'soundfile' "
            "package installed.)" % (path, error_text)
        )



    # ======================
    # Filters
    # ======================

    def filter_changed(self, text):

        filters = {

            "None": "none",

            "Low Pass": "lowpass",

            "High Pass": "highpass",

            "Band Pass": "bandpass",

            "DSP Pipeline (Wiener + Spectral Subtraction)": "dsp_pipeline",

            "Improved DSP Pipeline (Live)": "improved_dsp",

        }


        # Push the current config path fields to the engine so a
        # DSP Pipeline / Improved DSP switch below picks them up.

        self.engine.set_config_path(
            self.config_path_input.text().strip() or None
        )

        self.engine.set_improved_dsp_config_path(
            self.improved_dsp_config_input.text().strip()
            or "config/improved_dsp.yaml"
        )


        try:

            self.engine.set_filter_type(
                filters[text]
            )

        except Exception as e:

            import traceback

            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Pipeline error",
                "Failed to switch to %r:\n%s" % (text, e)
            )


        self.update_frequency_visibility()



    def update_frequency_visibility(self):

        mode = self.filter_box.currentText()


        if mode in (
            "None",
            "DSP Pipeline (Wiener + Spectral Subtraction)",
            "Improved DSP Pipeline (Live)",
        ):

            self.low_label.hide()
            self.low_input.hide()

            self.high_label.hide()
            self.high_input.hide()


        elif mode == "Band Pass":

            self.low_label.show()
            self.low_input.show()

            self.high_label.show()
            self.high_input.show()


        else:

            self.low_label.show()
            self.low_input.show()

            self.high_label.hide()
            self.high_input.hide()



    def update_cutoffs(self):

        try:

            self.engine.set_cutoffs(
                float(
                    self.low_input.text()
                ),
                float(
                    self.high_input.text()
                )
            )

        except ValueError:

            pass



    def update_order(self):

        try:

            self.engine.set_filter_order(
                int(
                    self.order_input.text()
                )
            )

        except ValueError:

            pass



    # ======================
    # Pipeline config / device listing / offline methods
    # ======================


    def update_config_path(self):

        self.engine.set_config_path(
            self.config_path_input.text().strip() or None
        )



    def update_improved_dsp_config_path(self):

        self.engine.set_improved_dsp_config_path(
            self.improved_dsp_config_input.text().strip()
            or "config/improved_dsp.yaml"
        )



    def list_devices(self):

        # Mirrors `python -m senhance.main --list-devices`. Printed to
        # the console rather than sys.exit()'d, since the GUI needs to
        # keep running.

        try:

            from senhance.audio.stream_manager import AudioStreamManager

            AudioStreamManager.list_devices()

        except Exception as e:

            import traceback

            traceback.print_exc()

            print("Falling back to sd.query_devices():")

            print(sd.query_devices())



    def show_offline_methods_info(self):

        QMessageBox.information(
            self,
            "Offline-Only Methods",
            "dl (DeepFilterNet3), hybrid_method_1, and hybrid_method_3 "
            "have no streaming implementation -- they only process a "
            "complete whole-clip array via .process_array(...), so they "
            "can't run in this live GUI loop.\n\n"
            "Audition them instead with:\n"
            "  python -m scripts.run_virtual_mic_test --pipeline dl\n"
            "  python -m scripts.run_virtual_mic_test --pipeline hybrid_method_1\n"
            "  python -m scripts.run_virtual_mic_test --pipeline hybrid_method_3\n\n"
            "That script feeds a WAV file through their real APIs and "
            "plays the result out the same virtual-mic device this live "
            "GUI uses, so it's a fair comparison -- just not live."
        )




if __name__ == "__main__":


    app = QApplication(
        sys.argv
    )


    window = ClearCallGUI()

    window.show()


    sys.exit(
        app.exec()
    )