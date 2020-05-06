#!/usr/bin/env python
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.dataset.GeoTiffDataset import GeoTiffDataset, GeoTiffWriter
from utils.dataset.transforms import transforms as T


def _is_empty(tensor, epsilon=1e-5):
    """Utility function to test if an image is blank."""
    return T.inv_normalize(tensor).sum() < epsilon


def predict_tiff(model, device, img_path, dest_path, transform, crop_size=200, pad=0, batch_size=8):
    """
    Predict segmentation classes on a GeoTiff image.
    Args:
        model: A PyTorch Model class
        device: A torch device
        img_path: Path to the image to classify
        dest_path: Location to save the new image
        transform: PyTorch data transforms for input data patches
        crop_size: Size of patches to predict on before stitching them back together
        pad: Padding context to add to image crops
        batch_size: The size of mini images to process at one time. Must be greater than 1.

    Returns: str Path to classified GeoTiff segmentation
    """
    model.eval()

    # Process PNG for better multiprocessing performance
    # img_png_path = Path(img_path).with_suffix('.jpeg')
    # rasterio.shutil.copy(img_path, img_png_path, driver='JPEG')

    ds = GeoTiffDataset(img_path, transform, crop_size=crop_size, pad=pad)
    writer = GeoTiffWriter(dest_path, ds.raster.height, ds.raster.width, ds.raster.crs, ds.raster.transform,
                           crop_size, pad)
    dataloader = DataLoader(ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    for i, xs in enumerate(tqdm(dataloader, file=sys.stdout)):
        xs = xs.to(device)
        batch_size = xs.shape[0]

        # skip classification entirely if batch is all empty images
        if all([_is_empty(x) for x in xs]):
            continue

        # Filter out empty images
        batch_indices = [j for j in range(batch_size) if not _is_empty(xs[j])]
        xs = torch.stack([x for x in xs if not _is_empty(x)])

        # Do segmentation
        segmentation = model(xs)['out']
        # torch.cuda.synchronize(device=device)  # For profiling only

        segmentation = F.softmax(segmentation, dim=1)
        segmentation = segmentation[:, 1] * 255  # For likelihood output
        segmentation = segmentation.detach().cpu().numpy()
        segmentation = np.expand_dims(segmentation, axis=1)
        segmentation = segmentation.astype(np.uint8)

        # Save part of tiff wither rasterio TODO: Multi-process this
        for j, seg in zip(batch_indices, segmentation):
            y0, x0 = ds.get_origin(i * batch_size + j)
            writer.write_index(y0, x0, seg)
