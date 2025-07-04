from collections.abc import Callable, Iterable

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
            A.PadIfNeeded(
                tile_size, tile_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0
            ),
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
    fill: int = 0,
    fill_mask: int = 0,
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
                        hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=15, p=1
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.044), p=1),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=1),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=1),
                    A.MedianBlur(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 64),
                        hole_height_range=(1, 5),
                        hole_width_range=(1, 5),
                        fill=fill,
                        fill_mask=fill_mask,
                        p=1,
                    ),
                    A.GridDistortion(num_steps=10, distort_limit=(-0.1, 0.1), p=0.3),
                ],
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
            # Drone mosaic-specific augmentations
            A.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.3
            ),
            A.Affine(
                scale=(0.9, 1.1),
                keep_ratio=True,
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                fill=fill,
                fill_mask=fill_mask,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )
