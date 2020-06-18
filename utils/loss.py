import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data


def iou(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8):
    """Computes the intersection over union statistic for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(n, c, h, w)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(n, h, w)
        int type ground truth labels for each sample.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    List[float]
        The calculated IoU index for each class.
    """
    num_classes = p.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[g.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(p)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[g.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(p, dim=1)
    true_1_hot = true_1_hot.type(p.type())
    dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    return (intersection / (union + smooth))


def jaccard_loss(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8):
    """Computes the Jaccard similarity loss for predictions p and ground truth labels g.

    Parameters
    ----------
    p : np.ndarray shape=(n, c, h, w)
        Softmax or sigmoid scaled predictions.
    g : np.ndarray shape=(n, h, w)
        int type ground truth labels for each sample.
    smooth : Optional[float]
        A function smooth parameter that also provides numerical stability.

    Returns
    -------
    List[float]
        The calculated IoU index for each class.
    """
    return 1 - iou(p, g, smooth=smooth).mean()


def dice_similarity_c(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8):
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
    g = F.one_hot(g.flatten().long(), c)

    tp = torch.sum(torch.mul(p, g), dim=0)
    denom = torch.sum(p + g, dim=0)
    return ((2 * tp) + smooth) / (denom + smooth)


def dice_loss(p: torch.Tensor, g: torch.Tensor, smooth: float = 1e-8):
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
    return torch.sum(1 - dsc, dim=0)


def tversky_index_c(p: torch.Tensor, g: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-8):
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
    p = p.permute(0, 2, 3, 1).reshape((-1, c))
    g = F.one_hot(g.flatten().long(), c)

    tp = torch.sum(torch.mul(p, g), dim=0)
    fn = torch.sum(torch.mul(1. - p, g), dim=0)
    fp = torch.sum(torch.mul(p, 1. - g), dim=0)
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def tversky_loss(p: torch.Tensor, g: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-8):
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
    return torch.sum(1 - ti, dim=0)


def focal_tversky_loss(p: torch.Tensor, g: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.,
                       smooth: float = 1e-8):
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
    return torch.sum(res, dim=0)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
