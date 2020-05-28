import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data


def iou(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    return (intersection / (union + eps))


def jaccard_loss(true, logits, eps=1e-7):
    return 1 - iou(true, logits, eps=eps).mean()


def _dice_similarity_c(p, g, smooth=1e-8):
    """
    p.shape: N,C (softmax outputs)
    g.shape: N (int labels)
    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = F.one_hot(torch.LongTensor([0, 0, 1]), 2)
    >>> list(np.around(_dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [0.777778, 0.666667]

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = F.one_hot(torch.LongTensor([0, 0, 1]), 2)
    >>> list(np.around(_dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [1.0, 1.0]

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = F.one_hot(torch.LongTensor([1, 1, 0]), 2)
    >>> list(np.around(_dice_similarity_c(X, y, smooth=0).numpy(), 6))
    [0.0, 0.0]
    """
    tp = torch.sum(torch.mul(p, g), dim=0)
    denom = torch.sum(p + g, dim=0)
    return ((2 * tp) + smooth) / (denom + smooth)


def dice_loss(p, g, smooth=1e-8):
    """
    Loss function from the paper S. R. Hashemi, et al, 2018. "Asymmetric loss functions and deep densely-connected
    networks for highly-imbalanced medical image segmentation: application to multiple sclerosis lesion detection"
    https://ieeexplore.ieee.org/abstract/document/8573779.
    Electronic ISSN: 2169-3536. DOI: 10.1109/ACCESS.2018.2886371.

    p: predicted output from a sigmoid-like activation. (i.e. range is 0-1)
    g: ground truth label of pixel (0 or 1)
    beta: parameter that adjusts weight between FP and FN error importance. beta=1. simplifies to the Dice loss function
    (F1 score) and weights both FP and FNs equally. B=0 is precicion, B=2 is the F_2 score
    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([1, 1, 0])
    >>> np.around(dice_loss(X, y, smooth=0).numpy(), 1)
    2.0
    """
    num_samples, num_classes = p.shape
    g = F.one_hot(g.long(), num_classes)

    dsc = _dice_similarity_c(p, g, smooth)
    return torch.sum(1 - dsc, dim=0)


def _tversky_index_c(p, g, alpha=0.5, beta=0.5, smooth=1e-8):
    """
    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = F.one_hot(torch.LongTensor([0, 0, 1]), 2)
    >>> np.allclose(_dice_similarity_c(X, y).numpy(), _tversky_index_c(X, y, alpha=0.5, beta=0.5).numpy())
    True

    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = F.one_hot(torch.LongTensor([0, 0, 1]), 2)
    >>> list(np.around(_tversky_index_c(X, y).numpy(), 6))
    [0.777778, 0.666667]

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = F.one_hot(torch.LongTensor([0, 0, 1]), 2)
    >>> list(np.around(_tversky_index_c(X, y).numpy(), 6))
    [1.0, 1.0]

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = F.one_hot(torch.LongTensor([1, 1, 0]), 2)
    >>> list(np.around(_tversky_index_c(X, y).numpy(), 6))
    [0.0, 0.0]
    """
    tp = torch.sum(torch.mul(p, g), dim=0)
    fp = torch.sum(torch.mul(p, 1. - g), dim=0)
    fn = torch.sum(torch.mul(1. - p, g), dim=0)
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def tversky_loss(p, g, alpha=0.5, beta=0.5, smooth=1e-8):
    """
    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([1, 1, 0])
    >>> np.around(tversky_loss(X, y, smooth=0).numpy(), 1)
    2.0
    """
    num_samples, num_classes = p.shape
    g = F.one_hot(g.long(), num_classes)

    ti = _tversky_index_c(p, g, alpha, beta, smooth)
    return torch.sum(1 - ti, dim=0)


def focal_tversky_loss(p, g, alpha=0.5, beta=0.5, gamma=1., smooth=1e-8):
    """
    >>> X = torch.Tensor([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 6)
    0.555556

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([0, 0, 1])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 1)
    0.0

    >>> X = torch.Tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> y = torch.Tensor([1, 1, 0])
    >>> np.around(focal_tversky_loss(X, y, alpha=0.5, beta=0.5, smooth=0).numpy(), 1)
    2.0
    """
    num_samples, num_classes = p.shape
    g = F.one_hot(g.long(), num_classes)

    ti = _tversky_index_c(p, g, alpha, beta, smooth)
    res = (1 - ti).pow(1 / gamma)
    return torch.sum(res, dim=0)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
