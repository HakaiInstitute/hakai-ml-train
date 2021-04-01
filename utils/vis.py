import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from dataset.transforms import reusable_transforms as T


def t2p(img):
    return ToPILImage()(img.detach().cpu())


def show_torch_img(img, label=None, pred=None):
    a = t2p(T.inv_normalize(img))
    plt.imshow(a)

    if label is not None:
        b = t2p(label.type(torch.FloatTensor))
        plt.imshow(b, alpha=0.3)

    if pred is not None:
        c = t2p(pred.max(dim=0)[1].type(torch.FloatTensor))
        plt.imshow(c, alpha=0.3)

    plt.show()


def show_torch_batch(imgs, labels, predictions):
    plt.figure(figsize=(15, 10))

    a = make_grid(torch.cat((imgs, imgs)), nrow=imgs.shape[0])
    a = t2p(T.inv_normalize(a))

    labels = labels.unsqueeze(dim=1)
    predictions = predictions.max(dim=1)[1].unsqueeze(dim=1)
    b = make_grid(torch.cat((labels, predictions)), nrow=imgs.shape[0])
    b = t2p(b.type(torch.FloatTensor))

    plt.imshow(a)
    plt.imshow(b, alpha=0.3)
    plt.show()
