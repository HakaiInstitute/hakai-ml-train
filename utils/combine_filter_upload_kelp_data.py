#!/usr/bin/env python
import os
from pathlib import Path

import boto3
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.SegmentationDataset import SegmentationDataset
from dataset.TransformDataset import TransformDataset
from dataset.transforms import transforms as T


def get_indices_of_kelp_images(dataset):
    ds = TransformDataset(dataset, transform=T.test_transforms, target_transform=T.test_target_transforms)
    dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    indices = []
    for i, (_, y) in enumerate(tqdm(iter(dl))):
        if torch.any(y > 0):
            indices.append(i)
    return indices


def main(*ds_paths, out_dir="deeplabv3/kelp/train_input/data", s3_bucket="hakai-deep-learning-datasets",
         dataset_name="kelp", train_ratio=0.8):
    out_dir = Path(out_dir)
    # ds_paths = [x for x in Path("data/kelp/processed").iterdir() if x.is_dir()]

    # Make split reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Filter non-kelp images
    print("Filtering images containing no kelp whatsoever.")
    full_ds = torch.utils.data.ConcatDataset([SegmentationDataset(path) for path in ds_paths])
    kelp_indices = get_indices_of_kelp_images(full_ds)
    kelp_ds = torch.utils.data.Subset(full_ds, kelp_indices)

    # Split into train and val
    train_num = int(len(kelp_ds) * train_ratio)
    val_num = len(kelp_ds) - train_num
    splits = torch.utils.data.random_split(kelp_ds, [train_num, val_num])

    # Save images and labels to sagemaker dir and s3 bucket
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(s3_bucket)

    for ds, phase_name in zip(splits, ['train', 'eval']):
        print(f"Uploading {phase_name} dataset to Amazon S3")
        for i, (x, y) in enumerate(tqdm(ds)):
            out_x_path = out_dir.joinpath(phase_name, 'x', f'{i}.png')
            out_y_path = out_dir.joinpath(phase_name, 'y', f'{i}.png')

            out_x_path.parents[0].mkdir(parents=True, exist_ok=True)
            out_y_path.parents[0].mkdir(parents=True, exist_ok=True)

            # Save locally
            x.save(out_x_path, 'PNG')
            y.save(out_y_path, 'PNG')

            # Upload to Amazon S3
            s3_bucket.upload_file(str(out_x_path), f'{dataset_name}/{phase_name}/x/{i}.png')
            s3_bucket.upload_file(str(out_y_path), f'{dataset_name}/{phase_name}/y/{i}.png')


if __name__ == '__main__':
    fire.Fire(main)
