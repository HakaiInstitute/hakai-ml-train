import segmentation_models_pytorch as smp
import torch

from config import TrainingConfig


def main():
    # Download .ckpt file from W&B
    # import wandb
    # run = wandb.init()
    # artifact = run.use_artifact('hakai/kom-kelp-pa-aco-rgbi/model-mfiaag3u:v0', type='model')
    # artifact_dir = artifact.download()
    # print(artifact_dir)

    DEVICE = torch.device('cpu')
    CKPT_FILE = "./UNetPlusPlus_Resnet34_kelp_presence_aco_miou=0.8340.ckpt"

    state_dict = torch.load(CKPT_FILE, map_location=DEVICE)['state_dict']
    # print(state_dict)

    # Load stripped back model
    config = TrainingConfig()
    model = smp.UnetPlusPlus('resnet34', in_channels=config.num_bands,
                             classes=config.num_classes - 1, decoder_attention_type="scse")

    # Replace keys to remove mismatches due to torch.compile
    state_dict = {k.replace('model._orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict=state_dict, strict=False)

    if len(missing) > 0:
        print(f"Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected}")

    # Export as JIT
    x = torch.rand(1, config.num_bands, config.img_shape, config.img_shape,
                               device=DEVICE, requires_grad=False)
    output_path = "../inference/UNetPlusPlus_Resnet34_kelp_presence_aco_jit_miou=0.8340.pt"
    traced_model = torch.jit.trace_module(model, {"forward": x})
    traced_model.save(output_path)


if __name__ == '__main__':
    main()
