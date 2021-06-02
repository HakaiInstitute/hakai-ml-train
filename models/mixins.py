"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-21
Description: 
"""
from abc import ABCMeta

import pytorch_lightning as pl
from geotiff_crop_dataset import CropDatasetWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.utils.GeoTiffDataset import GeoTiffReader
from datasets.utils.transforms import reusable_transforms as t


class GeoTiffPredictionMixin(pl.LightningModule, metaclass=ABCMeta):
    def pred_dataloader(self, geotiff_path: str):
        ds_pred = GeoTiffReader(geotiff_path, transform=t.geotiff_transforms,
                                crop_size=self.hparams.crop_size, padding=self.hparams.crop_pad)
        return DataLoader(ds_pred, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=0)

    def predict_geotiff(self, geotiff_path: str, out_path: str):
        pred_dataloader = self.pred_dataloader(geotiff_path)
        writer = CropDatasetWriter.from_reader(out_path, pred_dataloader.dataset, count=1, dtype='uint8', nodata=0)

        for idx, x in enumerate(tqdm(pred_dataloader, desc="Writing output")):
            pred = self(x).argmax(dim=1).detach().cpu().numpy()
            writer.write_batch(idx, self.hparams.batch_size, pred)
