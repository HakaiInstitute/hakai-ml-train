import os.path
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import wandb

from config import pa_training_config, sp_training_config, TrainingConfig

DEVICE = torch.device('cpu')


def convert_checkpoint(checkpoint_url: str, config: TrainingConfig):
    # Download .ckpt file from W&B
    api = wandb.Api()
    artifact = api.artifact(checkpoint_url, type='model')
    miou = round(artifact.metadata['score'], 4)
    artifact_dir = artifact.download()

    ckpt_file = Path(artifact_dir) / 'model.ckpt'.format(miou)
    output_path = '../inference/UNetPlusPlus_Resnet34_kelp_presence_aco_jit_miou={:.4f}.pt'.format(miou)

    state_dict = torch.load(ckpt_file, map_location=DEVICE)['state_dict']

    # Load stripped back model
    model = smp.UnetPlusPlus('resnet34', in_channels=config.num_bands,
                             classes=config.num_classes - 1, decoder_attention_type="scse")

    # Replace keys to remove mismatches due to torch.compile
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict=state_dict, strict=False)

    if len(missing) > 0:
        print(f"Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected}")

    # Export as JIT
    x = torch.rand(1, config.num_bands, config.tile_size, config.tile_size,
                   device=DEVICE, requires_grad=False)

    traced_model = torch.jit.trace_module(model, {"forward": x})
    traced_model.save(output_path)
    print(f"Saved JIT model to {os.path.abspath(output_path)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(("checkpoint_url"), type=str)
    parser.add_argument("model_type", type=str, choices=["pa", "sp"])
    args = parser.parse_args()

    train_config = {"pa": pa_training_config, "sp": sp_training_config}[args.model_type]
    convert_checkpoint(args.checkpoint_url, train_config)
