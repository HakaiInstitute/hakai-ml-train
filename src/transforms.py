import io

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def get_test_transforms(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )


def get_train_transforms(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    fill: int = 0,
    fill_mask: int = 0,
):
    return A.Compose(
        [
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


if __name__ == "__main__":
    test_t = A.Compose(
        [
            A.PadIfNeeded(1024, 1024, border_mode=0, fill=0, fill_mask=0, p=1.0),
            A.Normalize(
                max_pixel_value=4000.0,
                normalization="standard",
                mean=[0.43, 0.42875, 0.47825, 0.522, 0.5685, 0.5725, 0.65325, 0.9925],
                std=[0.18675, 0.1745, 0.18475, 0.192, 0.21225, 0.217, 0.21225, 0.2285],
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1,
    )

    train_t = A.Compose(
        [
            A.RandomCrop(height=1024, width=1024, pad_if_needed=True, p=1.0),
            A.ToFloat(max_value=4000.0, p=1.0),
            A.SquareSymmetry(p=1.0),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0, 32),
                hole_width_range=(0, 32),
                fill=0.0,
                fill_mask=0.0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(0, 0),
                brightness_by_max=False,
                p=0.3,
            ),
            A.Normalize(
                max_pixel_value=1.0,
                normalization="standard",
                mean=[0.43, 0.42875, 0.47825, 0.522, 0.5685, 0.5725, 0.65325, 0.9925],
                std=[0.18675, 0.1745, 0.18475, 0.192, 0.21225, 0.217, 0.21225, 0.2285],
                p=1.0,
            ),
            A.ToTensorV2(p=1.0),
        ],
        p=1.0,
        seed=42,
    )

    x = np.random.randint(0, 255, (1024, 1024, 4), dtype=np.uint8)

    train_t(image=x)

    with io.StringIO() as f:
        A.save(train_t, f, data_format="yaml")
        f.seek(0)
        print(f.read())
