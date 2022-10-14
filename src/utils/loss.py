from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.nn.functional import one_hot
from torchmetrics import Metric


# noinspection DuplicatedCode
def dice_similarity_c(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """Compute the Dice similarity index for each class for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(batch_size, num_classes, height, width)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(batch_size, height, width)
        int type ground truth labels for each sample.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    List[float]
        The calculated similarity index amount for each class.

    Examples
    --------
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> list(np.around(dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [0.777778, 0.666667]

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> list(np.around(dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [1.0, 1.0]

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[1]], [[1]], [[0]]])
    >>> list(np.around(dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [0.0, 0.0]
    """
    c = p.shape[1]
    p = p.permute(0, 2, 3, 1).reshape((-1, c))
    g = one_hot(g.flatten().long(), c)

    tp = torch.nansum(torch.mul(p, g), dim=0)
    denominator = torch.nansum(p + g, dim=0)
    return ((2 * tp) + smooth) / (denominator + smooth)


def dice_loss(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """Loss function from the paper S. R. Hashemi, et al, 2018. "Asymmetric loss functions and deep densely-connected
    networks for highly-imbalanced medical image segmentation: application to multiple sclerosis lesion detection"
    https://ieeexplore.ieee.org/abstract/document/8573779.
    Electronic ISSN: 2169-3536. DOI: 10.1109/ACCESS.2018.2886371.

    Parameters
    ----------
    p : np.ndarray shape=(batch_size, num_classes, height, width)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(batch_size, height, width)
        int type ground truth labels for each sample.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    float
        The calculated loss amount.

    Examples
    --------
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[1]], [[1]], [[0]]])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 1)
    2.0
    """
    dsc = dice_similarity_c(p, g, smooth)
    return torch.nansum(1 - dsc, dim=0)


# noinspection DuplicatedCode
def tversky_index_c(
        p: torch.Tensor,
        g: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-8,
) -> torch.Tensor:
    """Compute the Tversky similarity index for each class for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(batch_size, num_classes, height, width)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(batch_size, height, width)
        int type ground truth labels for each sample.
    alpha : Optional[float]
        The relative weight to go to false negatives.
    beta : Optional[float]
        The relative weight to go to false positives.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    List[float]
        The calculated similarity index amount for each class.

    Examples
    --------
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.allclose(dice_similarity_c(X, y).numpy(), tversky_index_c(X, y, alpha=0.5, beta=0.5).numpy())
    True

    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> list(np.around(tversky_index_c(X, y).numpy(), 6))
    [0.777778, 0.666667]

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> list(np.around(tversky_index_c(X, y).numpy(), 6))
    [1.0, 1.0]

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.LongTensor([[[1]], [[1]], [[0]]])
    >>> list(np.around(tversky_index_c(X, y).numpy(), 6))
    [0.0, 0.0]
    """
    c = p.shape[1]
    # p = p.permute(0, 2, 3, 1).reshape((-1, c))
    g = one_hot(g.flatten().long(), c)

    tp = torch.nansum(torch.mul(p, g), dim=0)
    fn = torch.nansum(torch.mul(1.0 - p, g), dim=0)
    fp = torch.nansum(torch.mul(p, 1.0 - g), dim=0)
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def tversky_loss(
        p: torch.Tensor,
        g: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-8,
) -> torch.Tensor:
    """Compute the Tversky Loss for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(batch_size, num_classes, height, width)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(batch_size, height, width)
        int type ground truth labels for each sample.
    alpha : Optional[float]
        The relative weight to go to false negatives.
    beta : Optional[float]
        The relative weight to go to false positives.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    float
        The calculated loss amount.

    Examples
    --------
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[1]], [[1]], [[0]]])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 1)
    2.0
    """
    ti = tversky_index_c(p, g, alpha, beta, smooth)
    return torch.nansum(1 - ti, dim=0)


def focal_tversky_loss(
        p: torch.Tensor,
        g: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1e-8,
) -> torch.Tensor:
    """Compute the focal Tversky Loss for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(batch_size, num_classes, height, width)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(batch_size, height, width)
        int type ground truth labels for each sample.
    alpha : Optional[float]
        The relative weight to go to false negatives.
    beta : Optional[float]
        The relative weight to go to false positives.
    gamma : Optional[float]
        Parameter controlling how much weight is given to large vs small errors in prediction.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    float
        The calculated loss amount.

    Examples
    --------
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[1]], [[1]], [[0]]])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 1)
    2.0
    """
    ti = tversky_index_c(p, g, alpha, beta, smooth)
    res = (1 - ti).pow(1 / gamma)
    return torch.nansum(res, dim=0)


def del_column(data: torch.Tensor, idx: int) -> torch.Tensor:
    """Delete the column at index."""
    return torch.cat([data[:, :idx], data[:, (idx + 1):]], 1)


class FocalTverskyLoss(Metric):
    """Computes the Focal-Tversky Loss as a torchmetrics.Metric

    Examples
    --------
    >>> loss = FocalTverskyLoss(num_classes=2, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(loss(X, y).numpy(), 6)
    0.555556

    >>> loss = FocalTverskyLoss(num_classes=2, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> np.around(loss(X, y).numpy(), 1)
    0.0

    >>> loss = FocalTverskyLoss(num_classes=2, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[1.]], [[0.]]], [[[1.]], [[0.]]], [[[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[1]], [[1]], [[0]]])
    >>> np.around(loss(X, y).numpy(), 1)
    2.0

    >>> loss = FocalTverskyLoss(num_classes=3, ignore_index=1, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[1.]], [[0.]], [[0.]]], [[[1.]], [[0.]], [[0.]]], [[[0.]], [[0.]], [[1.]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[2]]])
    >>> np.around(loss(X, y).numpy(), 1)
    0.0

    >>> loss = FocalTverskyLoss(num_classes=3, ignore_index=2, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[0.9]], [[0.1]], [[0.0]]], [[[0.5]], [[0.5]], [[0.0]]], [[[0.2]], [[0.8]], [[0.0]]], [[[0.8]], [[0.2]], [[0.0]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]], [[2]]])
    >>> np.around(loss(X, y).numpy(), 6)
    0.555556

    >>> loss = FocalTverskyLoss(num_classes=3, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> X = torch.Tensor([[[[0.9]], [[0.1]]], [[[0.5]], [[0.5]]], [[[0.2]], [[0.8]]]])
    >>> y = torch.Tensor([[[0]], [[0]], [[1]]])
    >>> loss_1 = loss(X, y).numpy()
    >>> loss_2 = focal_tversky_loss(X, y, alpha=0.5, beta=0.5, gamma=1.0, smooth=0)
    >>> np.allclose(loss_1, loss_2)
    True
    """
    full_state_update = False
    is_differentiable = True
    higher_is_better = False

    def __init__(
            self,
            num_classes: int,
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 1.0,
            smooth: float = 1e-8,
            ignore_index: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        c = self.num_classes if self.ignore_index is None else self.num_classes - 1
        self.add_state("p_g", default=torch.zeros(c), dist_reduce_fx="sum")
        self.add_state("np_g", default=torch.zeros(c), dist_reduce_fx="sum")
        self.add_state("p_ng", default=torch.zeros(c), dist_reduce_fx="sum")

    @staticmethod
    def _input_format(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = preds.shape[1]
        preds = preds.permute(0, 2, 3, 1).reshape((-1, c))
        target = one_hot(target.flatten().long(), c).float()
        return preds, target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        if self.ignore_index is not None:
            preds = del_column(preds, self.ignore_index)
            target = del_column(target, self.ignore_index)

        # Remove pixels where label is ignore class with mask
        mask = torch.nansum(target, dim=1).unsqueeze(dim=1)

        self.p_g += torch.nansum(torch.mul(preds, target) * mask, dim=0)
        self.np_g += torch.nansum(torch.mul(1.0 - preds, target) * mask, dim=0)
        self.p_ng += torch.nansum(torch.mul(preds, 1.0 - target) * mask, dim=0)

    def compute(self) -> torch.Tensor:
        ti = (self.p_g + self.smooth) / (
                self.p_g + self.alpha * self.np_g + self.beta * self.p_ng + self.smooth
        )
        res = (1 - ti).pow(1 / self.gamma)
        return torch.nansum(res, dim=0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
