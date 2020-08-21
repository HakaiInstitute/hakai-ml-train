"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-21
Description: 
"""
import os
from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl
import torch
from geotiff_crop_dataset import CropDatasetWriter
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.GeoTiffDataset import GeoTiffReader
from utils.dataset.transforms import transforms as t


class GeoTiffPredictionMixin(pl.LightningModule, metaclass=ABCMeta):
    @abstractmethod
    @auto_move_data
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of predicted labels for sample x."""
        raise NotImplemented

    def pred_dataloader(self, geotiff_path: str):
        ds_pred = GeoTiffReader(geotiff_path, transform=t.test_transforms,
                                crop_size=self.hparams.crop_size, padding=self.hparams.padding)
        return DataLoader(ds_pred, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=os.cpu_count())

    def predict_geotiff(self, geotiff_path: str, out_path: str):
        pred_dataloader = self.pred_dataloader(geotiff_path)
        writer = CropDatasetWriter.from_reader(out_path, pred_dataloader.dataset, bands=1, dtype='uint8', nodata=0)

        for idx, x in enumerate(tqdm(pred_dataloader, desc="Writing output")):
            pred = self.predict(x).detach().cpu().numpy()
            writer.write_batch(idx, self.hparams.batch_size, pred)
