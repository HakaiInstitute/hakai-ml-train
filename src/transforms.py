import io

import albumentations as A
import numpy as np
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
            # Geometric
            A.D4(p=1.0),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                fill=fill,
                fill_mask=fill_mask,
                p=0.5,
            ),
            # Color / lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.25
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
            # Sharpness — simulates resolution variation across flights
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.UnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.5), p=1.0),
                ],
                p=0.15,
            ),
            # Channel regularization
            A.OneOf(
                [
                    A.ToGray(p=1.0),
                    A.ChannelDropout(p=1.0),
                ],
                p=0.1,
            ),
            # Noise regularization
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.044), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=1.0),
                ],
                p=0.15,
            ),
            # Blur regularization
            A.OneOf(
                [
                    A.MotionBlur(p=1.0),
                    A.MedianBlur(p=1.0),
                    A.GaussianBlur(p=1.0),
                ],
                p=0.1,
            ),
            # Dropout regularization
            A.OneOf(
                [
                    A.GridDropout(ratio=0.2, fill=fill, fill_mask=fill_mask, p=1.0),
                    A.CoarseDropout(
                        num_holes_range=(1, 8),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32),
                        fill=fill,
                        fill_mask=fill_mask,
                        p=1.0,
                    ),
                ],
                p=0.25,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        p=1,
    )


if __name__ == "__main__":
    test_t = get_test_transforms()

    train_t = get_train_transforms()

    x = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    train_t(image=x)

    with io.StringIO() as f:
        A.save(train_t, f, data_format="yaml")
        f.seek(0)
        print(f.read())
