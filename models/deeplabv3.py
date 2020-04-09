from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def create_model(num_classes=1):
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    model.requires_grad_(False)

    model.classifier = DeepLabHead(2048, num_classes)
    model.classifier.requires_grad_(True)

    # Set the model in training mode
    model.train()
    return model
