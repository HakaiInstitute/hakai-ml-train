#!/usr/bin/env python
import sys
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset.GeoTiffDataset import GeoTiffDataset, GeoTiffWriter
from utils.loss import iou


def eval_model(model, device, data_loaders, num_classes, criterion):
    model.eval()

    for phase in ['train', 'eval']:
        sum_loss = 0.
        sum_iou = np.zeros(num_classes)

        for x, y in tqdm(iter(data_loaders[phase]), desc=phase, file=sys.stdout):
            y = y.to(device)
            x = x.to(device)

            pred = model(x)['out']
            loss = criterion(pred.float(), y)

            # Compute metrics
            sum_loss += loss.detach().cpu().item()
            sum_iou += iou(y, pred.float()).detach().cpu().numpy()

        print(phase.capitalize())
        print('Loss', np.around(sum_loss / len(data_loaders[phase]), 4))
        print('IoU/Mean', np.around(np.mean(sum_iou / len(data_loaders[phase])), 4))
        print('IoU/BG', np.around(sum_iou[0] / len(data_loaders[phase]), 4))
        print('IoU/Kelp', np.around(sum_iou[1] / len(data_loaders[phase]), 4))


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
    ds = GeoTiffDataset(img_path, transform, crop_size=crop_size, pad=pad)
    writer = GeoTiffWriter(ds, dest_path)
    dataloader = DataLoader(ds, batch_size, shuffle=False, num_workers=1, pin_memory=True)

    for i, xs in enumerate(tqdm(iter(dataloader), file=sys.stdout)):
        xs = xs.to(device)

        # Do segmentation
        segmentation = model(xs)['out']

        # torch.cuda.synchronize() # For profiling only
        segmentation = F.softmax(segmentation, dim=1)
        segmentation = segmentation[:, 1] * 255  # For likelihood output
        segmentation = segmentation.detach().cpu().numpy()
        # segmentation = np.argmax(segmentation, axis=1)  # For discrete output
        segmentation = np.expand_dims(segmentation, axis=1)
        segmentation = segmentation.astype(np.uint8)

        # Save part of tiff wither rasterio
        for j, seg in enumerate(segmentation):
            writer.write_index(i*batch_size + j, seg)
