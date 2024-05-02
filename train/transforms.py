from typing import Callable, Iterable

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


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
            # A.ToFloat(p=1),
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
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.Downscale(
                scale_min=0.5,
                scale_max=0.9,
                interpolation=A.Downscale.Interpolation(
                    downscale=cv2.INTER_AREA, upscale=cv2.INTER_LINEAR
                ),
                p=0.1,
            ),
            # Colour transforms
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=0.8,
            ),
            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=1),
                    A.OpticalDistortion(p=1),
                    A.Perspective(p=1),
                    # A.RandomGridShuffle(p=1),
                ],
                p=0.8,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.02,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            # A.ToFloat(p=1),
            ToTensorV2(),
        ],
        p=1,
    )
