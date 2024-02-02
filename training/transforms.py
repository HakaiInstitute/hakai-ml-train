from typing import Optional, Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_test_transforms(tile_size: int = 1024,
                        extra_transforms: Optional[list[Callable]] = None):
    if extra_transforms is None:
        extra_transforms = []
    return A.Compose(
        [
            *extra_transforms,
            A.ToFloat(p=1),
            A.PadIfNeeded(
                tile_size, tile_size, border_mode=0, value=0, p=1.0
            ),
            ToTensorV2(),
        ],
        p=1,
    )


def get_train_transforms(tile_size: int = 1024,
                         extra_transforms: Optional[list[Callable]] = None):
    if extra_transforms is None:
        extra_transforms = []

    return A.Compose(
        [
            *extra_transforms,
            A.ToFloat(p=1),
            A.ShiftScaleRotate(
                scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7
            ),
            A.PadIfNeeded(
                tile_size, tile_size, border_mode=0, value=0, p=1.0
            ),
            A.RandomCrop(tile_size, tile_size, p=1.0),
            A.Flip(p=0.75),
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
            A.MaskDropout(
                max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1
            ),
            # Colour transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    A.RandomGamma(gamma_limit=(70, 130), p=1),
                    # A.ChannelShuffle(p=0.2),
                    # A.HueSaturationValue(
                    #     hue_shift_limit=30,
                    #     sat_shift_limit=40,
                    #     val_shift_limit=30,
                    #     p=1,
                    # ),
                    # A.RGBShift(
                    #     r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1
                    # ),
                ],
                p=0.8,
            ),
            # distortion
            A.OneOf(
                [
                    A.ElasticTransform(p=1),
                    A.OpticalDistortion(p=1),
                    A.GridDistortion(p=1),
                    A.Perspective(p=1),
                ],
                p=0.2,
            ),
            # noise transforms
            A.OneOf(
                [
                    A.GaussNoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.Sharpen(p=1),
                    A.GaussianBlur(p=1),
                ],
                p=0.2,
            ),
            ToTensorV2(),
        ],
        p=1,
    )
