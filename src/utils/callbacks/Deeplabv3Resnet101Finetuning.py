import pytorch_lightning as pl
from torch.optim import Optimizer


class Deeplabv3Resnet101Finetuning(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, train_bn=True):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._train_bn = train_bn

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.model.backbone, train_bn=self._train_bn)

    def finetune_function(
            self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer,
            opt_idx: int
    ) -> None:
        if epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=[pl_module.model.backbone.layer4, pl_module.model.backbone.layer3],
                optimizer=optimizer,
                train_bn=self._train_bn)

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("Deeplabv3Resnet101Finetuning")

        group.add_argument(
            "--unfreeze_backbone_epoch",
            type=int,
            default=0,
            help="The training epoch to unfreeze earlier layers of Deeplabv3 for fine tuning.",
        )
        group.add_argument(
            "--train_backbone_bn",
            dest="train_backbone_bn",
            action="store_true",
            help="Flag to indicate if backbone batch norm layers should be trained.",
        )
        group.add_argument(
            "--no_train_backbone_bn",
            dest="train_backbone_bn",
            action="store_false",
            help="Flag to indicate if backbone batch norm layers should not be trained.",
        )
        group.set_defaults(train_backbone_bn=True)

        return parser