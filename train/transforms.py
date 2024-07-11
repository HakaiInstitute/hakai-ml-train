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
            A.PadIfNeeded(tile_size, tile_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
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
            A.PadIfNeeded(
                tile_size, tile_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0
            ),
            A.RandomCrop(tile_size, tile_size, p=1.0),
            A.D4(p=1.0),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=1
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(5.0, 30.0), p=1),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.GaussianBlur(blur_limit=3, p=1),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CoarseDropout(
                        max_holes=4, max_height=16, max_width=16, fill_value=0, p=1
                    ),
                    A.GridDistortion(num_steps=3, distort_limit=0.2, p=1),
                ],
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # Drone mosaic-specific augmentations
            A.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.3
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.3
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )
