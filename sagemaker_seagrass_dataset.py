#!/usr/bin/env python
import os
from pathlib import Path

import boto3
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.TransformDataset import TransformDataset
from utils.dataset.transforms import transforms as T

disable_cuda = False
train_ratio = 0.8

out_dir = Path("unet/seagrass/train_input/data")
s3_bucket_name = 'hakai-deep-learning-datasets'
ds_paths = [x for x in Path("data/seagrass/processed").iterdir() if x.is_dir()]


def get_indices_of_seagrass_images(dataset):
    ds = TransformDataset(dataset, transform=T.test_transforms, target_transform=T.test_target_transforms)
    dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    indices = []
    for i, (_, y) in enumerate(tqdm(iter(dl))):
        if torch.any(y > 0):
            indices.append(i)
    return indices


if __name__ == '__main__':
    # Make split reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Filter non-seagrass images
    full_ds = torch.utils.data.ConcatDataset([SegmentationDataset(path) for path in ds_paths])
    seagrass_indices = get_indices_of_seagrass_images(full_ds)
    seagrass_ds = torch.utils.data.Subset(full_ds, seagrass_indices)

    # Split into train and val
    train_num = int(len(seagrass_ds) * train_ratio)
    val_num = len(seagrass_ds) - train_num
    splits = torch.utils.data.random_split(seagrass_ds, [train_num, val_num])

    # Save images and labels to sagemaker dir and s3 bucket
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(s3_bucket_name)

    for ds, phase_name in zip(splits, ['train', 'eval']):
        for i, (x, y) in enumerate(tqdm(ds)):
            out_x_path = out_dir.joinpath(phase_name, 'x', f'{i}.png')
            out_y_path = out_dir.joinpath(phase_name, 'y', f'{i}.png')

            out_x_path.parents[0].mkdir(parents=True, exist_ok=True)
            out_y_path.parents[0].mkdir(parents=True, exist_ok=True)

            # Save locally
            x.save(out_x_path, 'PNG')
            y.save(out_y_path, 'PNG')

            # Upload to Amazon S3
            s3_bucket.upload_file(str(out_x_path), f'seagrass/{phase_name}/x/{i}.png')
            s3_bucket.upload_file(str(out_y_path), f'seagrass/{phase_name}/y/{i}.png')
