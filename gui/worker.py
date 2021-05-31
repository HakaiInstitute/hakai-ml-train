# This Python file uses the following encoding: utf-8
from os import path

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
# noinspection PyUnresolvedReferences
from __feature__ import snake_case, true_property
from geotiff_crop_dataset import CropDatasetWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset.GeoTiffDataset import GeoTiffReader

geotiff_transforms = transforms.Compose([
    transforms.Lambda(lambda img: np.asarray(img)[:, :, :3]),
    transforms.Lambda(lambda img: np.clip(img, 0, 255).astype(np.uint8)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WorkerThread(QThread):
    files_started = Signal(int)
    image_started = Signal(int, str, str)
    files_progress = Signal(int)
    image_progress = Signal(int)
    files_finished = Signal()

    def __init__(self, parent, image_paths, output_path_pattern, model_path, device, batch_size,
                 crop_size, crop_pad):
        super().__init__(parent)
        self.running = False
        self.image_paths = image_paths
        self.output_path_pattern = output_path_pattern
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.crop_pad = crop_pad

    def _get_output_name(self, input_path):
        basename, ext = path.splitext(path.basename(input_path))
        return self.output_path_pattern.format(basename=basename)

    def _load_model(self):
        return torch.jit.load(self.model_path, map_location=self.device)

    def run(self):
        self.running = True
        self.files_started.emit(len(self.image_paths))
        self.files_progress.emit(0)

        model = self._load_model()

        for file_idx, img_path in enumerate(self.image_paths):
            out_path = self._get_output_name(img_path)

            # Load dataset
            reader = GeoTiffReader(img_path, transform=geotiff_transforms,
                                   crop_size=self.crop_size, padding=self.crop_pad)
            writer = CropDatasetWriter.from_reader(out_path, reader, count=1, dtype='uint8',
                                                   nodata=0)
            dataloader = DataLoader(reader, shuffle=False, batch_size=self.batch_size,
                                    pin_memory=True, num_workers=0)

            # Do the image classification
            self.image_started.emit(len(dataloader), img_path, out_path)
            self.image_progress.emit(0)

            for batch_idx, x in enumerate(tqdm(dataloader, desc="Writing file")):
                pred = model(x.to(self.device)).argmax(dim=1).detach().cpu().numpy()
                writer.write_batch(batch_idx, self.batch_size, pred)
                self.image_progress.emit(batch_idx + 1)
            self.files_progress.emit(file_idx + 1)

        self.running = False
        self.files_finished.emit()
