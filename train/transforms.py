from typing import Callable, Iterable

import albumentations as A
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
            A.Downscale(p=0.05),
            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=0.25),
                    A.GridDistortion(p=0.25),
                    A.OpticalDistortion(p=0.25),
                    A.Perspective(p=0.25),
                ],
                p=0.8,
            ),
            # Colour transforms
            A.OneOf(
                [
                    A.ColorJitter(p=0.25),
                    A.HueSaturationValue(p=0.25),
                    A.RandomBrightnessContrast(p=0.25),
                    A.RandomGamma(p=0.25),
                ],
                p=0.8,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                    A.MultiplicativeNoise(p=0.5),
                ],
                p=0.5,
            ),
            # blur
            A.OneOf(
                [
                    A.GaussianBlur(p=0.7),
                    A.MotionBlur(p=0.3),
                ],
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )
