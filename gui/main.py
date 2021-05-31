# This Python file uses the following encoding: utf-8
import os
import signal
import sys
from os import path

import torch
from PySide6 import QtWidgets
from PySide6.QtCore import QFile, QObject, Slot
from PySide6.QtUiTools import QUiLoader
# noinspection PyUnresolvedReferences
from __feature__ import snake_case, true_property
from loguru import logger

from gui.worker import WorkerThread

MODELS_PATH = path.join(path.dirname(__file__), 'models')
VERSION = open(path.join(path.dirname(__file__), 'VERSION')).readline()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__(None)
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        central_widget = self.load_ui()
        central_widget.setWindowTitle(f"Hakai Kelp-O-Matic 9000 v{VERSION}")

        # Get handles to some components
        self.widget_progress = self.find_child(QObject, "widget_progress")
        self.push_button_run = self.find_child(QObject, "pushButton_run")
        self.tool_button_output_directory = self.find_child(QObject, "toolButton_output_directory")
        self.push_button_open_images = self.find_child(QObject, "pushButton_open_images")
        self.push_button_clear_images = self.find_child(QObject, "pushButton_clear_images")
        self.list_widget_image_files = self.find_child(QObject, "listWidget_image_files")
        self.line_edit_output_directory = self.find_child(QObject, "lineEdit_output_directory")
        self.line_edit_output_filename_pattern = self.find_child(QObject,
                                                                 "lineEdit_output_filename_pattern")
        self.combo_box_classification_model = self.find_child(QObject,
                                                              "comboBox_classification_model")
        self.progress_bar_overall = self.find_child(QObject, "progressBar_overall")
        self.progress_bar_image = self.find_child(QObject, "progressBar_img")
        self.label_current_file = self.find_child(QObject, "label_current_file")
        self.check_box_cuda = self.find_child(QObject, "checkBox_cuda")
        self.spin_box_batch_size = self.find_child(QObject, "spinBox_batch_size")
        self.spin_box_crop_dimension = self.find_child(QObject, "spinBox_crop_dimension")
        self.spin_box_crop_padding = self.find_child(QObject, "spinBox_crop_padding")

        # Set initial state for some components
        self.line_edit_output_directory.text = str(os.getenv('HOME'))
        self.widget_progress.visible = False
        self.find_child(QObject, "groupBox_advanced_options").visible = False
        if not torch.cuda.is_available():
            self.check_box_cuda.checked = False
            self.check_box_cuda.checkable = False

        # Add model options
        self.combo_box_classification_model.add_item(
            "Slow species (Deeplab v3 ResNet101, mIoU=0.9198) [1=Macro 2=Nereo] ",
            path.join(MODELS_PATH, "DeepLabV3_ResNet101_kelp_species_jit.pt"))
        self.combo_box_classification_model.add_item(
            "Quick species (LR-ASPP MobileNet v3, mIoU=0.8945) [1=Macro 2=Nereo] ",
            path.join(MODELS_PATH, "LRASPP_MobileNetV3_kelp_species_jit.pt"))
        self.combo_box_classification_model.add_item(
            "Slow presence/absence (Deeplab v3 ResNet101, mIoU=0.9393) [1=Kelp]",
            path.join(MODELS_PATH, "DeepLabV3_ResNet101_kelp_presence_jit.pt"))
        self.combo_box_classification_model.add_item(
            "Quick presence/absence (LR-ASPP MobileNet v3, mIoU=0.9218) [1=Kelp]",
            path.join(MODELS_PATH, "LRASPP_MobileNetV3_kelp_presence_jit.pt"))

        # Hookup signals and slots
        self.tool_button_output_directory.clicked.connect(self.handle_output_directory_clicked)
        self.push_button_open_images.clicked.connect(self.handle_image_open_clicked)
        self.push_button_clear_images.clicked.connect(self.handle_image_clear_clicked)
        self.push_button_run.clicked.connect(self.handle_run_clicked)
        self.check_box_cuda.toggled.connect(lambda: logger.debug(f"{self.device=}"))

        self.thread = None

        # Render
        central_widget.show()

    def load_ui(self):
        loader = QUiLoader(self)
        ui_file = QFile(path.join(path.dirname(__file__), "form.ui"))
        ui_file.open(QFile.ReadOnly)
        widget = loader.load(ui_file, self)
        ui_file.close()
        return widget

    @property
    def batch_size(self):
        return 2 ** int(self.spin_box_batch_size.value)

    @property
    def crop_dimension(self):
        return int(self.spin_box_crop_dimension.value)

    @property
    def crop_padding(self):
        return int(self.spin_box_crop_padding.value)

    @property
    def device(self):
        if self.check_box_cuda.checked and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    @property
    def image_paths(self):
        return [self.list_widget_image_files.item(i).text() for i in
                range(self.list_widget_image_files.count)]

    @Slot()
    def handle_output_directory_clicked(self):
        directory_path = QtWidgets.QFileDialog.get_existing_directory(
            self.tool_button_output_directory,
            "Select Output Directory", os.getenv('HOME'),
            options=QtWidgets.QFileDialog.ShowDirsOnly)
        logger.debug(f"{directory_path=}")
        if directory_path != "":
            self.line_edit_output_directory.text = str(directory_path)

    @Slot()
    def handle_image_open_clicked(self):
        image_paths, _ = QtWidgets.QFileDialog.get_open_file_names(
            self.push_button_open_images,
            "Select Images to Process",
            os.getenv('HOME'),
            "Image Files (*.tif *.tiff)",
        )
        logger.debug(f"{image_paths=}")
        if len(image_paths) > 0:
            new_paths = sorted(list(set(self.image_paths + image_paths)))
            self.list_widget_image_files.clear()
            self.list_widget_image_files.add_items(new_paths)

    @Slot()
    def handle_image_clear_clicked(self):
        self.list_widget_image_files.clear()

    @Slot(int)
    def handle_processing_start(self, num_files):
        # Setup progress bar
        self.progress_bar_overall.maximum = num_files
        self.progress_bar_overall.value = 0
        self.widget_progress.visible = True
        self.widget_progress.update()

    @Slot(int, str, str)
    def handle_image_start(self, num_batches, _, out_path):
        self.progress_bar_image.maximum = num_batches
        self.progress_bar_image.value = 0
        self.label_current_file.text = f"Writing: {out_path}"
        self.widget_progress.update()

    @Slot(int)
    def handle_file_progress(self, value):
        self.progress_bar_overall.value = value

    @Slot(int)
    def handle_image_progress(self, value):
        self.progress_bar_image.value = value

    @Slot()
    def handle_run_clicked(self):
        if not self.thread or not self.thread.running:
            output_path_pattern = path.join(self.line_edit_output_directory.text,
                                            self.line_edit_output_filename_pattern.text)
            model_path = self.combo_box_classification_model.current_data

            if len(self.image_paths) < 1:
                return

            self.thread = WorkerThread(
                parent=self,
                image_paths=self.image_paths,
                output_path_pattern=output_path_pattern,
                model_path=model_path,
                device=self.device,
                batch_size=self.batch_size,
                crop_size=self.crop_dimension,
                crop_pad=self.crop_padding
            )

            self.thread.files_started.connect(self.handle_processing_start)
            self.thread.image_started.connect(self.handle_image_start)
            self.thread.files_progress.connect(self.handle_file_progress)
            self.thread.image_progress.connect(self.handle_image_progress)

            self.thread.start()
            # self.thread.wait()

    def exit_gracefully(self):
        logger.info("Exiting gracefully")
        if self.thread and self.thread.running:
            logger.info("Requesting thread interrupt")
            self.thread.exit(0)
            self.thread.wait(2)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    sys.exit(app.exec())

# TODO:
# Package app, fix model location
# Show errors and handle them gracefully
