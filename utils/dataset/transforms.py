from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms as t


def _target_to_tensor_func(mask: np.ndarray) -> torch.Tensor:
    return (F.to_tensor(mask) * 255).long().squeeze(dim=0)


# noinspection PyTypeChecker
target_to_tensor = t.Lambda(_target_to_tensor_func)

normalize = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
inv_normalize = t.Compose([t.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                           t.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                           ])


class PadOut(object):
    def __init__(self, height: int = 128, width: int = 128):
        self.height = height
        self.width = width

    def __call__(self, x: Image) -> Image:
        """
        Pad out a pillow image so it is the correct size as specified by self.height and self.width

        :param x: PIL Image
        :return: PIL Image
        """
        w, h = x.size

        if h == self.height and w == self.width:
            return x

        wpad = self.width - w
        hpad = self.height - h

        return F.pad(x, [0, 0, wpad, hpad])


class Clamp(object):
    def __init__(self, min_: int = 0, max_: int = 1):
        self.min = min_
        self.max = max_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp the pixel values so they fall within [self.min, self.max]

        :param x: PIL Image
        :return: PIL Image
        """
        return torch.clamp(x, self.min, self.max)


def add_veg_indices(x):
    """Add a slew of RGB vegetation indices to the image.
    x: torch.Tensor RGB image of shape (n, c, h, w)

    returns modified image tensor with bands in order
        n, [rgbvi, vari, gli, ngrdi, r, g, b], h, w
    """
    x = add_ngrdi_band(x)
    x = add_gli_band(x)
    x = add_vari_band(x)
    x = add_rgbvi_band(x)
    return x


def add_rgbvi_band(img):
    """Add RGBVI as band 0.
    img: a Torch tensor of shape (n, c, h, w)
     It is assumed there are at least 3 channels,
        with RGB located at index -3, -2, -1, respectively

    returns img of shape (n, c+1, h, w)
    """
    r, g, b = img[:, -3, :, :], img[:, -2, :, :], img[:, -1, :, :]

    rgbvi = (torch.mul(r, r) - torch.mul(r, b)) / (torch.mul(g, g) + torch.mul(r, b))
    return torch.cat((rgbvi.unsqueeze(1), img), dim=1)


def add_gli_band(img):
    """Add GLI as band 0.
    img: a Torch tensor of shape (n, c, h, w)

    returns img of shape (n, c+1, h, w)
    """
    r, g, b = img[:, -3, :, :], img[:, -2, :, :], img[:, -1, :, :]
    gli = (2 * g - r - b) / (2 * g + r + b)
    return torch.cat((gli.unsqueeze(1), img), dim=1)


def add_vari_band(img):
    """Add VARI as band 0.
    img: a Torch tensor of shape (n, c, h, w)

    returns img of shape (n, c+1, h, w)
    """
    r, g, b = img[:, -3, :, :], img[:, -2, :, :], img[:, -1, :, :]
    vari = (g - r) / (g + r - b)
    return torch.cat((vari.unsqueeze(1), img), dim=1)


def add_ngrdi_band(img):
    """Add NGRDI as band 0.
    img: a Torch tensor of shape (n, c, h, w)

    returns img of shape (n, c+1, h, w)
    """
    r, g, b = img[:, -3, :, :], img[:, -2, :, :], img[:, -1, :, :]
    ngrdi = (g - r) / (g + r)
    return torch.cat((ngrdi.unsqueeze(1), img), dim=1)


class ImageClip(object):
    """Changes the pixel domain of PIL Image img to [0, 255] and convert to type uint8."""

    def __init__(self, min_: int = 0, max_: int = 255) -> None:
        """
        Parameters
        ----------
        min_: Change all pixel values less than this value to this value.
        max_: Change all pixel values greater than this value to this value.
        """
        self.min = min_
        self.max = max_

    def __call__(self, img: Image):
        """
        Parameters
        ----------
        img: the Image to clip.

        Returns
        -------
        Image: The clipped image.
        """
        img_arr = np.asarray(img)
        img_arr = np.clip(img_arr, self.min, self.max).astype(np.uint8)
        return Image.fromarray(img_arr)


class DropExtraBands(object):
    def __init__(self, keep_bands=3):
        self.keep_bands = keep_bands

    def __call__(self, img):
        img_arr = np.asarray(img)
        img_arr = img_arr[:, :, :self.keep_bands]
        return Image.fromarray(img_arr)


reusable_transforms = SimpleNamespace(
    normalize=normalize,
    inv_normalize=inv_normalize,
    train_transforms=t.Compose([
        ImageClip(),
        PadOut(512, 512),
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        t.RandomRotation(degrees=45),
        t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        t.ToTensor(),
        normalize,
    ]),
    train_target_transforms=t.Compose([
        PadOut(512, 512),
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        t.RandomRotation(degrees=45, fill=(0,)),
        target_to_tensor,
    ]),
    test_transforms=t.Compose([
        ImageClip(),
        PadOut(512, 512),
        t.ToTensor(),
        normalize,
    ]),
    test_target_transforms=t.Compose([
        PadOut(512, 512),
        target_to_tensor,
    ]),
    geotiff_transforms=t.Compose([
        ImageClip(),
        DropExtraBands(),
        t.ToTensor(),
        normalize,
    ]),

)
