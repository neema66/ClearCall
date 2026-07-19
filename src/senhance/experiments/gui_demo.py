"""
GUI for ClearCall DSP experiment.

Controls:
- Start/Stop audio
- Artificial noise level in dB
- Noise only mode
- Filter selection
- Cutoff frequencies
- Filter order
"""

import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QComboBox,
    QLineEdit,
)

from PySide6.QtCore import Qt

from senhance.experiments.simple_filter import SimpleAudioEngine



class FocusLineEdit(QLineEdit):
    """
    Line edit that updates when:
    - Enter is pressed
    - focus leaves the box
    """

    def __init__(self, callback, text=""):

        super().__init__(text)

        self.callback = callback


    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:

            self.callback()

            self.clearFocus()

        else:

            super().keyPressEvent(event)



    def focusOutEvent(self, event):

        self.callback()

        super().focusOutEvent(event)




class ClearCallGUI(QWidget):

    def __init__(self):

        super().__init__()


        self.engine = SimpleAudioEngine()


        self.setWindowTitle(
            "ClearCall DSP Demo"
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



        self.setLayout(
            layout
        )


        self.update_frequency_visibility()



    # ======================
    # Global mouse update
    # ======================


    def mousePressEvent(self, event):

        self.update_noise()

        self.update_cutoffs()

        self.update_order()

        super().mousePressEvent(event)



    # ======================
    # Audio
    # ======================


    def start_audio(self):

        self.engine.start()

        self.status.setText(
            "Running"
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

        print(
            "Noise enabled:",
            checked
        )



    def noise_only_changed(self, checked):

        self.engine.noise_only = checked

        print(
            "Noise only:",
            checked
        )



    def update_noise(self):

        try:

            value = float(
                self.noise_input.text()
            )

            self.engine.set_noise_level(
                value
            )

            print(
                "Noise dB:",
                value
            )


        except ValueError:

            pass



    # ======================
    # Filters
    # ======================


    def filter_changed(self, text):


        filters = {

            "None": "none",

            "Low Pass": "lowpass",

            "High Pass": "highpass",

            "Band Pass": "bandpass",

        }


        self.engine.set_filter_type(
            filters[text]
        )


        self.update_frequency_visibility()



    def update_frequency_visibility(self):


        mode = self.filter_box.currentText()



        if mode == "None":

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

            low = float(
                self.low_input.text()
            )


            high = float(
                self.high_input.text()
            )


            self.engine.set_cutoffs(
                low,
                high
            )


            print(
                "Cutoffs:",
                low,
                high
            )


        except ValueError:

            pass



    def update_order(self):

        try:

            order = int(
                self.order_input.text()
            )


            self.engine.set_filter_order(
                order
            )


            print(
                "Filter order:",
                order
            )


        except ValueError:

            pass





if __name__ == "__main__":


    app = QApplication(
        sys.argv
    )


    window = ClearCallGUI()

    window.show()


    sys.exit(
        app.exec()
    )