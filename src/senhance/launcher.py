"""
Launcher window for the ClearCall / senhance speech enhancement system.

This is what `python -m senhance.main` opens now instead of a headless
CLI loop. From here you can:

- Launch the real-time mic -> speaker GUI (original_dsp / improved_dsp,
  two-mic mixing, synthetic noise injection -- senhance.experiments.gui_demo)
- Run one of the offline-only methods (dl / hybrid_method_1 /
  hybrid_method_3) against a WAV file, since those have no streaming
  implementation (see senhance.main's old docstring / scripts/run_virtual_mic_test.py)
- List audio devices (mirrors the old --list-devices flag)

Shared Config File / Improved DSP Config fields at the top apply to
whichever window/run you launch next.
"""

from __future__ import annotations

import shlex
import subprocess
import sys

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


OFFLINE_PIPELINES = [
    "dl",
    "hybrid_method_1",
    "hybrid_method_3",
]


class LauncherWindow(QWidget):

    def __init__(
        self,
        initial_config_path="config/default.yaml",
        initial_dsp_config_path="config/improved_dsp.yaml",
        initial_pipeline=None,
    ):

        super().__init__()

        self.setWindowTitle(
            "ClearCall -- senhance"
        )


        self._live_gui = None

        self._offline_process = None


        layout = QVBoxLayout()


        layout.addWidget(
            QLabel(
                "<b>ClearCall / senhance Speech Enhancement</b>"
            )
        )


        # ------------------------------------------------
        # Shared config fields
        # ------------------------------------------------

        layout.addWidget(
            QLabel(
                "Config File (used by both live and offline runs; "
                "blank uses AppSettings defaults)"
            )
        )

        self.config_path_input = QLineEdit(
            initial_config_path or ""
        )

        layout.addWidget(
            self.config_path_input
        )


        layout.addWidget(
            QLabel(
                "Improved DSP Config (only used for the improved_dsp "
                "pipeline)"
            )
        )

        self.dsp_config_input = QLineEdit(
            initial_dsp_config_path
        )

        layout.addWidget(
            self.dsp_config_input
        )


        # ------------------------------------------------
        # Real-time mic -> speaker
        # ------------------------------------------------

        layout.addWidget(
            QLabel(
                "<b>Real-Time</b>: live microphone(s) -> enhancement -> "
                "speaker/virtual mic. Supports original_dsp and "
                "improved_dsp (the only two pipelines that can actually "
                "stream), plus the classic lowpass/highpass/bandpass "
                "filters, two-mic mixing, and synthetic noise injection."
            )
        )

        realtime_row = QHBoxLayout()

        realtime_row.addWidget(
            QLabel("Initial pipeline:")
        )

        self.realtime_pipeline_box = QComboBox()

        self.realtime_pipeline_box.addItems(
            [
                "none",
                "original_dsp",
                "improved_dsp",
            ]
        )

        if initial_pipeline in (
            "original_dsp",
            "improved_dsp",
        ):

            self.realtime_pipeline_box.setCurrentText(
                initial_pipeline
            )

        realtime_row.addWidget(
            self.realtime_pipeline_box
        )

        layout.addLayout(
            realtime_row
        )

        realtime_button = QPushButton(
            "Open Real-Time Mic \u2192 Speaker"
        )

        realtime_button.clicked.connect(
            self.open_realtime_gui
        )

        layout.addWidget(
            realtime_button
        )


        # ------------------------------------------------
        # Offline-only methods
        # ------------------------------------------------

        layout.addWidget(
            QLabel(
                "<b>Offline</b>: dl (DeepFilterNet3), hybrid_method_1, "
                "and hybrid_method_3 have no streaming implementation "
                "-- they only process a complete whole-clip array. This "
                "runs scripts/run_virtual_mic_test.py against a WAV file "
                "and plays the result out the same virtual-mic device "
                "the live GUI uses."
            )
        )

        offline_row = QHBoxLayout()

        offline_row.addWidget(
            QLabel("Pipeline:")
        )

        self.offline_pipeline_box = QComboBox()

        self.offline_pipeline_box.addItems(
            OFFLINE_PIPELINES
        )

        offline_row.addWidget(
            self.offline_pipeline_box
        )

        layout.addLayout(
            offline_row
        )

        wav_row = QHBoxLayout()

        self.wav_path_input = QLineEdit()

        wav_row.addWidget(
            self.wav_path_input
        )

        browse_button = QPushButton(
            "Browse WAV..."
        )

        browse_button.clicked.connect(
            self.browse_wav
        )

        wav_row.addWidget(
            browse_button
        )

        layout.addLayout(
            wav_row
        )

        # The exact CLI flags scripts/run_virtual_mic_test.py accepts
        # beyond --pipeline weren't available when this was written --
        # the command is editable here so you can fix flag names before
        # running rather than the launcher guessing wrong silently.

        layout.addWidget(
            QLabel(
                "Command (edit if the flags below don't match your "
                "actual script):"
            )
        )

        self.offline_command_input = QLineEdit()

        layout.addWidget(
            self.offline_command_input
        )

        self.offline_pipeline_box.currentTextChanged.connect(
            self._update_offline_command_preview
        )

        self.wav_path_input.textChanged.connect(
            self._update_offline_command_preview
        )

        self._update_offline_command_preview()


        run_offline_button = QPushButton(
            "Run Offline Method"
        )

        run_offline_button.clicked.connect(
            self.run_offline_method
        )

        layout.addWidget(
            run_offline_button
        )

        self.offline_output = QTextEdit()

        self.offline_output.setReadOnly(
            True
        )

        self.offline_output.setPlaceholderText(
            "Offline run output will appear here..."
        )

        layout.addWidget(
            self.offline_output
        )


        # ------------------------------------------------
        # Devices
        # ------------------------------------------------

        list_devices_button = QPushButton(
            "List Audio Devices"
        )

        list_devices_button.clicked.connect(
            self.list_devices
        )

        layout.addWidget(
            list_devices_button
        )


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



    # ======================
    # Real-time GUI
    # ======================


    def open_realtime_gui(self):

        from senhance.experiments.gui_demo import ClearCallGUI

        pipeline = self.realtime_pipeline_box.currentText()

        if pipeline == "none":

            pipeline = None


        self._live_gui = ClearCallGUI(
            initial_config_path=self.config_path_input.text().strip(),
            initial_dsp_config_path=self.dsp_config_input.text().strip(),
            initial_pipeline=pipeline,
        )

        self._live_gui.show()



    # ======================
    # Offline methods
    # ======================


    def browse_wav(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a WAV file",
            "",
            "WAV files (*.wav)"
        )

        if path:

            self.wav_path_input.setText(
                path
            )



    def _update_offline_command_preview(self):

        pipeline = self.offline_pipeline_box.currentText()

        wav_path = self.wav_path_input.text().strip()


        parts = [
            "python", "-m", "scripts.run_virtual_mic_test",
            "--pipeline", pipeline,
        ]

        if wav_path:

            parts += ["--input", wav_path]


        self.offline_command_input.setText(
            " ".join(shlex.quote(p) for p in parts)
        )



    def run_offline_method(self):

        command_text = self.offline_command_input.text().strip()

        if not command_text:

            QMessageBox.warning(
                self,
                "No command",
                "Fill in a WAV file (or edit the command box directly) "
                "before running."
            )

            return


        try:

            args = shlex.split(command_text)

        except ValueError as e:

            QMessageBox.critical(
                self,
                "Invalid command",
                "Could not parse the command: %s" % e
            )

            return


        self.offline_output.append(
            "$ %s\n" % command_text
        )


        try:

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=600,
            )

            self.offline_output.append(
                result.stdout
            )

            if result.stderr:

                self.offline_output.append(
                    result.stderr
                )

            self.offline_output.append(
                "\n(exit code %d)\n" % result.returncode
            )


        except Exception as e:

            import traceback

            self.offline_output.append(
                "Failed to run: %s\n%s" % (e, traceback.format_exc())
            )



    # ======================
    # Devices
    # ======================


    def list_devices(self):

        try:

            from senhance.audio.stream_manager import AudioStreamManager

            AudioStreamManager.list_devices()

            QMessageBox.information(
                self,
                "Devices",
                "Device list printed to the console."
            )


        except Exception as e:

            import traceback

            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Failed to list devices",
                str(e)
            )




if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = LauncherWindow()

    window.show()

    sys.exit(
        app.exec()
    )