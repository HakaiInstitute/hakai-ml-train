from types import SimpleNamespace

import torch
from torchvision import transforms as T


def _target_to_tensor_func(mask):
    return (T.functional.to_tensor(mask) * 255).long().squeeze(dim=0)


_target_to_tensor = T.Lambda(_target_to_tensor_func)
_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_inv_normalize = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            T.Normalize(mean=[-0.485, -0.456, -0.406],
                                        std=[1., 1., 1.]),
                            ])
transforms = SimpleNamespace(
    normalize=_normalize,
    inv_normalize=_inv_normalize,
    train_transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        _normalize,
    ]),
    train_target_transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=45, fill=(0,)),
        _target_to_tensor,
    ]),
    test_transforms=T.Compose([
        T.ToTensor(),
        _normalize,
    ]),
    test_target_transforms=T.Compose([
        _target_to_tensor,
    ])
)


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
