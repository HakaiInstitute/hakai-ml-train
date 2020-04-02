from torch import nn
from . import deeplabv3


def half_precision(model):
    for layer in model.modules():
        if not isinstance(layer, nn.BatchNorm2d):
            layer.half()
            
    return model