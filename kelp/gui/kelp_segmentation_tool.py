"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-18
Description: GUI for the Kelp Segmentation Tool
"""

import sys
import tempfile
from pathlib import Path

import boto3
from PyQt5 import QtWidgets, uic


class PreferencesWindow(QtWidgets.QFrame):
    _s3_bucket_name = "hakai-deep-learning-datasets"

    def __init__(self):
        super().__init__()  # Call the inherited classes __init__ method
        self.ui = uic.loadUi('kelp_gui/kelp_segmentation_preferences.ui', self)  # Load the .ui file

        self._segmentation_type = 0
        self.weight_files = []

        self._s3_bucket = boto3.resource("s3").Bucket(self._s3_bucket_name)

        self.radio_seg_type_change()
        self.get_weights_options()

    @property
    def aws_s3_weights_prefix(self):
        return f'{["kelp_species", "kelp"][self._segmentation_type]}/weights/'

    @property
    def selected_weights(self):
        return self.weight_files[self.ui.combo_weights.currentIndex()]

    def radio_seg_type_change(self):
        if self.ui.radio_species.isChecked():
            self._segmentation_type = 0
        else:
            self._segmentation_type = 1

        self.get_weights_options()

    def get_weights_options(self):
        prefix = self.aws_s3_weights_prefix
        self.weight_files = [obj.key for obj in self._s3_bucket.objects.filter(Prefix=prefix)]
        self.weight_files = list(reversed(self.weight_files))

        self.ui.combo_weights.clear()
        self.ui.combo_weights.addItems([w[len(prefix):] for w in self.weight_files])

    def download_weights(self, cache_dir=tempfile.gettempdir(), callback=lambda _: None):
        out_path = Path(cache_dir).joinpath(self.selected_weights[len(self.aws_s3_weights_prefix):])
        self._s3_bucket.download_file(self.selected_weights, str(out_path), Callback=callback)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # Call the inherited classes __init__ method
        uic.loadUi('kelp_gui/kelp_segmentation_tool.ui', self)  # Load the .ui file

        # Setup the preferences window
        self.preferences = PreferencesWindow()

        self.show()  # Show the GUI

    def btn_add_img_clicked(self):
        pass

    def btn_remove_img_clicked(self):
        pass

    def btn_run_clicked(self):
        pass

    def btn_output_dir_clicked(self):
        pass

    @classmethod
    def action_new_triggered(cls):
        return cls()

    def action_open_triggered(self):
        pass

    def action_save_triggered(self):
        pass

    def action_preferences_triggered(self):
        self.preferences.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
    window = MainWindow()  # Create an instance of our class
    app.exec_()  # Start the application
