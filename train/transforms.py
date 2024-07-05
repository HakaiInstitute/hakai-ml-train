from typing import Callable, Iterable

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def _remap_species_labels(y: np.ndarray, **kwargs):
    new_y = y.copy()
    new_y[new_y == 0] = 3
    return new_y - 1


extra_transforms = {
    "species_label_transform": A.Lambda(
        name="remap_labels", mask=_remap_species_labels
    ),
}


def get_test_transforms(
    tile_size: int = 1024,
    extra_transforms: Iterable[Callable] = (),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return A.Compose(
        [
            *extra_transforms,
            A.PadIfNeeded(tile_size, tile_size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(tile_size, tile_size, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )


def get_train_transforms(
    tile_size: int = 1024,
    extra_transforms: Iterable[Callable] = (),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return A.Compose(
        [
            *extra_transforms,
            A.PadIfNeeded(tile_size, tile_size, border_mode=0, value=0, p=1.0),
            A.RandomCrop(tile_size, tile_size, p=1.0),
            A.D4(p=1.0),
            A.Downscale(
                p=0.1,
                scale_min=0.5,
                scale_max=0.9,
                interpolation_pair={
                    "upscale": cv2.INTER_LINEAR,
                    "downscale": cv2.INTER_NEAREST,
                },
            ),

            # Colour transforms
            A.OneOf(
                [
                    A.ColorJitter(
                        p=1,
                        brightness=(0.7, 1.3),
                        contrast=(0.7, 1.3),
                        saturation=(0.7, 1.3),
                        hue=(-0.3, 0.3),
                    ),
                    A.RandomGamma(p=1, gamma_limit=(70, 130)),
                    A.CLAHE(p=1),
                ],
                p=0.8,
            ),

            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=1),
                    A.OpticalDistortion(p=1),
                    A.Perspective(p=1),
                ],
                p=0.8,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=0.7),

                ],
                p=0.5,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )
