import re
from typing import Any

import segmentation_models_pytorch
import torch
from lightning import pytorch as pl
from torchmetrics import classification as fm

from src import losses
from src.models import configure_optimizers as _configure_optimizers


class KomRGBSpeciesBaselineModel(pl.LightningModule):
    pa_params_file = "artifacts/model-8a1r0k6h:v0/model.ckpt"
    sp_params_file = "artifacts/model-x7m8097u:v0/model.ckpt"

    def __init__(
        self,
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.CosineAnnealingLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        self.class_names = ["bg", "macro", "nereo"]

        # metrics
        self.accuracy = fm.Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.jaccard_index = fm.JaccardIndex(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.recall = fm.Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.precision = fm.Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.f1_score = fm.F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )

        # Override parent's activation function to remove .squeeze(1) for multiclass
        self.activation_fn = lambda x: torch.softmax(x, dim=1)

        self.init_model()

    def init_model(self):
        pa_params = torch.load(
            self.pa_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        pa_params = {re.sub(r"^model\.", "", k): v for k, v in pa_params.items()}

        self.pa_model = segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="tu-tf_efficientnetv2_m",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.pa_model.load_state_dict(pa_params, strict=True)

        sp_params = torch.load(
            self.sp_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        sp_params = {re.sub(r"^model\.", "", k): v for k, v in sp_params.items()}
        self.sp_model = segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="tu-tf_efficientnetv2_m",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        )
        self.sp_model.load_state_dict(sp_params, strict=True)

        for p in self.pa_model.parameters():
            p.requires_grad = False

        for p in self.sp_model.parameters():
            p.requires_grad = False

        self.combiner = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.hparams.num_classes,
            kernel_size=1,
            bias=True,
        )
        self._initialize_combiner()

    def _initialize_combiner(self):
        """Initialize to approximate intersection logic"""
        with torch.no_grad():
            self.combiner.weight.zero_()
            self.combiner.bias.zero_()

            # Output 0 (no kelp): negative binary logit indicates no kelp
            self.combiner.weight[0, 0, 0, 0] = -2.0  # negative weight on binary logit
            self.combiner.bias[0] = (
                1.0  # positive bias to favor this when binary is negative
            )

            # Output 1 (macro): positive binary + high macro logit
            self.combiner.weight[1, 0, 0, 0] = 1.0  # positive binary logit
            self.combiner.weight[1, 1, 0, 0] = 1.0  # macro logit from species
            self.combiner.weight[1, 2, 0, 0] = -0.5  # slight negative on nereo

            # Output 2 (nereo): positive binary + high nereo logit
            self.combiner.weight[2, 0, 0, 0] = 1.0  # positive binary logit
            self.combiner.weight[2, 1, 0, 0] = -0.5  # slight negative on macro
            self.combiner.weight[2, 2, 0, 0] = 1.0  # nereo logit from species

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        cat_logits = torch.cat([presence_logits, species_logits], dim=1)
        return self.combiner(cat_logits)

    def _forward_train(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        cat_logits = torch.cat([presence_logits, species_logits], dim=1)
        logits = self.combiner(cat_logits)

        pa_label = (torch.sigmoid(presence_logits) > 0.5).to(torch.long)
        sp_label = species_logits.argmax(dim=1, keepdim=True) + 1  # shift to 1,2

        label = pa_label * sp_label  # 0 if no kelp, else species label
        return logits, label.squeeze(1)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def on_train_epoch_end(self) -> None:
        self.accuracy.reset()
        self.jaccard_index.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1_score.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("val/accuracy_epoch", self.accuracy)

        iou_per_class = self.jaccard_index.compute()
        precision_per_class = self.precision.compute()
        recall_per_class = self.recall.compute()
        f1_per_class = self.f1_score.compute()

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i])
            self.log(f"val/recall_epoch/{class_name}", recall_per_class[i])
            self.log(f"val/precision_epoch/{class_name}", precision_per_class[i])
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i])
        self.log(f"val/iou_epoch", iou_per_class[1:].mean())

        self.jaccard_index.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        if phase == "train":
            # The target for training is derived from the presence and species models
            logits, y = self._forward_train(x)
        else:
            logits = self(x)
        loss = self.loss_fn(logits, y.long())

        # Add L1 regularization on the combiner layer weights
        # This encourages sparsity in the weights, which can help with generalization
        l1 = torch.sum(torch.abs(self.combiner.weight))
        loss = loss + 0.01 * l1

        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        self.log(f"{phase}/accuracy", self.accuracy(logits, y))

        iou_per_class = self.jaccard_index(logits, y)
        precision_per_class = self.precision(logits, y)
        recall_per_class = self.recall(logits, y)
        f1_per_class = self.f1_score(logits, y)
        for i, class_name in enumerate(self.class_names):
            self.log_dict(
                {
                    f"{phase}/iou_{class_name}": iou_per_class[i],
                    f"{phase}/recall_{class_name}": recall_per_class[i],
                    f"{phase}/precision_{class_name}": precision_per_class[i],
                    f"{phase}/f1_{class_name}": f1_per_class[i],
                }
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean())

        return loss

    def configure_optimizers(self):
        return _configure_optimizers(self)


class KomRGBISpeciesBaselineModel(KomRGBSpeciesBaselineModel):
    pa_params_file = "artifacts/model-1kzli3zn:v0/model.ckpt"
    sp_params_file = "artifacts/model-nysalr7l:v0/model.ckpt"

    def init_model(self):
        pa_params = torch.load(
            self.pa_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        pa_params = {re.sub(r"^model\.", "", k): v for k, v in pa_params.items()}

        self.pa_model = segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=4,
            classes=2,
            activation=None,
            decoder_attention_type="scse",
        )
        self.pa_model.load_state_dict(pa_params, strict=True)

        sp_params = torch.load(
            self.sp_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        sp_params = {re.sub(r"^model\.", "", k): v for k, v in sp_params.items()}
        self.sp_model = segmentation_models_pytorch.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=4,
            classes=2,
            activation=None,
            decoder_attention_type="scse",
        )
        self.sp_model.load_state_dict(sp_params, strict=True)

        for p in self.pa_model.parameters():
            p.requires_grad = False

        for p in self.sp_model.parameters():
            p.requires_grad = False

        self.combiner = torch.nn.Conv2d(
            in_channels=4,
            out_channels=self.hparams.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self._initialize_combiner()

    def _initialize_combiner(self):
        # Initialize weights to respect problem structure
        with torch.no_grad():
            # Start with zeros
            self.combiner.weights.zero_()
            self.combiner.bias.zero_()

            # Channel 0 (no kelp): heavily weight binary model's "no kelp" channel
            self.combiner.weight[0, 0, 0, 0] = 2.0  # binary no_kelp

            # Channel 1 (macro): combination of binary "kelp" and species "macro"
            self.combiner.weight[1, 1, 0, 0] = 1.0  # binary kelp
            self.combiner.weight[1, 2, 0, 0] = 1.0  # species macro

            # Channel 2 (nereo): combination of binary "kelp" and species "nereo"
            self.combiner.weight[2, 1, 0, 0] = 1.0  # binary kelp
            self.combiner.weight[2, 3, 0, 0] = 1.0  # species nereo

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        cat_logits = torch.cat([presence_logits, species_logits], dim=1)
        return self.combiner(cat_logits)

    def _forward_train(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        cat_logits = torch.cat([presence_logits, species_logits], dim=1)
        logits = self.combiner(cat_logits)

        pa_label = presence_logits.argmax(dim=1, keepdim=True).to(torch.long)
        sp_label = species_logits.argmax(dim=1, keepdim=True) + 1  # shift to 1,2

        label = pa_label * sp_label  # 0 if no kelp, else species label
        return logits, label.squeeze(1)


if __name__ == "__main__":
    model = KomRGBISpeciesBaselineModel(
        loss="DiceLoss",
        loss_opts={
            "mode": "multiclass",
        },
        num_classes=3,
        ignore_index=0,
        lr=0.0003,
        wd=0.01,
        b1=0.9,
        b2=0.99,
    )

    rand = torch.rand(2, 4, 256, 256)  # Example input tensor

    output = model(rand)  # Forward pass
    print(output.shape)  # Should be [2, 3, 256, 256]
    # This will not run the model, just to check if it initializes correctly
