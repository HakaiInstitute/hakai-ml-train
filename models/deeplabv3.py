from torchvision.models.segmentation import deeplabv3_resnet101
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def create_model(num_classes=1):
    model = deeplabv3_resnet101(
        pretrained=True, progress=True)
    model.requires_grad_(False)

    # Added a Sigmoid activation after the last convolution layer
#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
#     model.classifier[1].requires_grad_(True)
#     model.classifier[2].requires_grad_(True)
#     model.classifier[3].requires_grad_(True)
#     model.classifier[4].requires_grad_(True)

    model.classifier = DeepLabHead(2048, num_classes)
    model.classifier.requires_grad_(True)

    # Set the model in training mode
    model.train()
    return model