# Created by: Taylor Denouden
# Organization: Hakai Institute
from typing import TypeVar

from segmentation_models_pytorch import Unet
from torch.optim import Optimizer

from base_model import BaseFinetuning, BaseModel

T = TypeVar('T')


class UnetEfficientnet(BaseModel):
    def init_model(self):
        # Create model from pre-trained UNet
        self.model = Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )
        self.model.requires_grad_(True)
        self.model.encoder.requires_grad_(False)

    def freeze_before_training(self, ft_module: BaseFinetuning) -> None:
        ft_module.freeze(self.model.encoder, train_bn=False)

    def finetune_function(self, ft_module: BaseFinetuning, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if epoch == ft_module.unfreeze_at_epoch:
            ft_module.unfreeze_and_add_param_group(
                self.model.encoder,
                optimizer,
                train_bn=ft_module.train_bn)

    @staticmethod
    def drop_output_layer_weights(weights: T) -> T:
        # TODO: This is wrong, need to correct
        del weights["model.decoder.weight"]
        del weights["model.decoder.bias"]
        return weights
