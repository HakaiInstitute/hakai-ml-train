import torch


class ConfusionMatrix(object):
    """Computer a confusion matrix or IOU scores."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes))

    def update(self, actual, predicted):
        s = actual.flatten() * self.num_classes + predicted.flatten()
        self.matrix = self.matrix.put_(s, torch.ones_like(s).float(), accumulate=True).T

    def get_matrix(self):
        return self.matrix

    def get_iou(self):
        return torch.diag(self.matrix) / self.matrix.sum(dim=0)

    def get_miou(self):
        return torch.mean(self.get_iou())
    
    def to(self, device):
        self.matrix = self.matrix.to(device)
        return self
