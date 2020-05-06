import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable


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


def assymetric_tversky_loss(p, g, beta=1.):
    """Loss function from the paper S. R. Hashemi, et al, 2018. "Asymmetric loss functions and deep densely-connected
    networks for highly-imbalanced medical image segmentation: application to multiple sclerosis lesion detection"
    https://ieeexplore.ieee.org/abstract/document/8573779.
    Electronic ISSN: 2169-3536. DOI: 10.1109/ACCESS.2018.2886371.

    p: predicted output from a sigmoid-like activation. (i.e. range is 0-1)
    g: ground truth label of pixel (0 or 1)
    beta: parameter that adjusts weight between FP and FN error importance. beta=1. simplifies to the Dice loss function
    (F1 score) and weights both FP and FNs equally. B=0 is precicion, B=2 is the F_2 score

    >>> np.around(assymetric_tversky_loss(torch.Tensor([0.9, 0.5, 0.2]), torch.Tensor([1., 0., 1.]), beta=1.5).numpy(), 6)
    0.413934
    """
    p = p.float()
    g = g.float()
    bsq = beta * beta
    pg = torch.sum(torch.mul(p, g))  # 1.1
    similarity_coeff = ((1 + bsq) * pg) / (  # (3.25*1.1) / ((3.25*1.1)+(2.25*0.9)+(0.5))
            ((1 + bsq) * pg) + (bsq * torch.sum(torch.mul((1 - p), g))) + (torch.sum(torch.mul(p, (1 - g))))
    )
    return 1 - similarity_coeff


def dice_loss(p, g):
    """
    Loss function from the paper S. R. Hashemi, et al, 2018. "Asymmetric loss functions and deep densely-connected
    networks for highly-imbalanced medical image segmentation: application to multiple sclerosis lesion detection"
    https://ieeexplore.ieee.org/abstract/document/8573779.
    Electronic ISSN: 2169-3536. DOI: 10.1109/ACCESS.2018.2886371.

    p: predicted output from a sigmoid-like activation. (i.e. range is 0-1)
    g: ground truth label of pixel (0 or 1)
    beta: parameter that adjusts weight between FP and FN error importance. beta=1. simplifies to the Dice loss function
    (F1 score) and weights both FP and FNs equally. B=0 is precicion, B=2 is the F_2 score

    >>> np.around(dice_loss(torch.Tensor([0.9, 0.5, 0.2]), torch.Tensor([1., 0., 1.])).numpy(), 6)
    0.388889
    """
    return assymetric_tversky_loss(p, g, beta=1.)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


def make_one_hot(labels, num_classes=2):
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False)

    def forward(self, input, target):
        loss = self.nll(input, target)

        one_hot = make_one_hot(target.unsqueeze(dim=1), input.size()[1])
        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights

        return loss.mean()


if __name__ == '__main__':
    import doctest

    doctest.testmod()
