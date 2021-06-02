import torch

from lit_deeplabv3_resnet101_kelp import DeepLabv3ResNet101
from lit_lraspp_mobilenet_v3_large_kelp import LRASPPMobileNetV3Large

DEVICE = torch.device('cuda')

if __name__ == '__main__':
    x = torch.rand(1, 3, 8, 8, device=DEVICE)

    # Deeplab Presence Model
    model = DeepLabv3ResNet101({
        "aux_loss_factor": 0.3,
        "weight_decay": 0.001,
        "lr": 0.001,
        "unfreeze_backbone_epoch": 100,
        "num_classes": 2,
        "train_backbone_bn": False
    })
    model.load_state_dict(torch.load(
        "scripts/presence/train_output/checkpoints/DeepLabV3_ResNet101/version0/checkpoints/"
        "best-val_miou=0.9393-epoch=97-step=34789.pt"))
    model = model.eval()
    model = model.to(DEVICE)
    traced_model = torch.jit.trace(model, x)
    traced_model.save("torchscript_files/DeepLabV3_ResNet101_kelp_presence_jit.pt")
    # torch.onnx.export(model, x, "torchscript_files/DeepLabV3_ResNet101_kelp_presence.onnx",
    #                   opset_version=11, verbose=True)

    # Deeplab Species Model
    model = DeepLabv3ResNet101({
        "aux_loss_factor": 0.3,
        "weight_decay": 0.001,
        "lr": 0.001,
        "unfreeze_backbone_epoch": 100,
        "num_classes": 3,
        "train_backbone_bn": False
    })
    model.load_state_dict(torch.load(
        "scripts/species/train_output/checkpoints/DeepLabV3_ResNet101/version_0/checkpoints/"
        "best-val_miou=0.9198-epoch=96-step=28323_v2.pt"))
    model = model.eval()
    model = model.to(DEVICE)
    traced_model = torch.jit.trace(model, x)
    traced_model.save("torchscript_files/DeepLabV3_ResNet101_kelp_species_jit.pt")
    # torch.onnx.export(model, x, "torchscript_files/DeepLabV3_ResNet101_kelp_species.onnx",
    #                   opset_version=11, verbose=True)

    # LR-ASPP Presence Model
    model = LRASPPMobileNetV3Large({
        "weight_decay": 0.001,
        "lr": 0.001,
        "num_classes": 2,
    })
    model.load_state_dict(torch.load(
        "scripts/presence/train_output/checkpoints/LRASPP_MobileNetV3/version_0/checkpoints/"
        "best-val_miou=0.9218-epoch=196-step=69934.pt"))
    model = model.eval()
    model = model.to(DEVICE)
    traced_model = torch.jit.trace(model, x)
    traced_model.save("torchscript_files/LRASPP_MobileNetV3_kelp_presence_jit.pt")
    # torch.onnx.export(model, x, "torchscript_files/LRASPP_MobileNetV3_kelp_presence.onnx",
    #                   opset_version=11, verbose=True)

    # LR-ASPP Species Model
    model = LRASPPMobileNetV3Large({
        "weight_decay": 0.001,
        "lr": 0.001,
        "num_classes": 3,
    })
    model.load_state_dict(torch.load(
        "scripts/species/train_output/checkpoints/L_RASPP_MobileNetV3/version_0/checkpoints/"
        "best-val_miou=0.8945-epoch=198-step=58107.pt"))
    model = model.eval()
    model = model.to(DEVICE)
    traced_model = torch.jit.trace(model, x)
    traced_model.save("torchscript_files/LRASPP_MobileNetV3_kelp_species_jit.pt")
    # torch.onnx.export(model, x, "torchscript_files/LRASPP_MobileNetV3_kelp_species.onnx",
    #                   opset_version=11, verbose=True)
