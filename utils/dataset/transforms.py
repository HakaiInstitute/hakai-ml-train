from torchvision import transforms as T
from types import SimpleNamespace


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
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=90),
        T.ToTensor(),
        _normalize,
    ]),
    train_target_transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=90, fill=(0,)),
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
