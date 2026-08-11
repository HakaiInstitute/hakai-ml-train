from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics as tm
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin
from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.dofa_vit import DOFAEncoderWrapper, load_dofa_weights
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from torchgeo.models import dofa
from torchvision.models._api import Weights

from hakai_ml_train import losses
from hakai_ml_train.models import LayerDecayMixin
from hakai_ml_train.models import configure_optimizers as _configure_optimizers

model_factory = EncoderDecoderFactory()


@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_base_patch16_224_custom(
    wavelengths,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = dofa.DOFABase16_Weights.DOFA_MAE,
    out_indices: list | None = None,
    pos_interpolation_mode: str = "bilinear",
    **kwargs,
):
    model = dofa.dofa_base_patch16_224(**kwargs)
    input_size = kwargs.get("img_size", 224)
    if pretrained:
        model = load_dofa_weights(
            model, pos_interpolation_mode, ckpt_data, weights, input_size
        )

    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_large_patch16_224_custom(
    wavelengths,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = dofa.DOFALarge16_Weights.DOFA_MAE,
    out_indices: list | None = None,
    pos_interpolation_mode: str = "bilinear",
    **kwargs,
):
    model = dofa.dofa_large_patch16_224(**kwargs)
    input_size = kwargs.get("img_size", 224)
    if pretrained:
        model = load_dofa_weights(
            model, pos_interpolation_mode, ckpt_data, weights, input_size
        )

    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)


class TerraTorchSegmentationModel(
    pl.LightningModule,
    PyTorchModelHubMixin,
    LayerDecayMixin,
    library_name="habitat_mapper",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/habitat-mapper",
    docs_url="https://habitat-mapper.readthedocs.io/",
):
    # DOFAEncoderWrapper holds the raw ViT at `.dofa_model`.
    backbone_module_path = "model.encoder.dofa_model"

    def __init__(
        self,
        model_opts: dict[str, Any],
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        task = "binary" if num_classes == 1 else "multiclass"

        self.model = model_factory.build_model(
            task="segmentation",
            num_classes=self.hparams.num_classes,
            **model_opts,
        )
        if ckpt_path is not None:
            ckpt = torch.load(self.hparams.ckpt_path, weights_only=False)
            self.load_state_dict(ckpt["state_dict"])

        for p in self.model.parameters():
            p.requires_grad = True
        if freeze_backbone:
            print("Freezing backbone parameters")
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        # self.model = torch.compile(self.model)

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        if task == "binary":
            self.activation_fn = lambda x: torch.sigmoid(x).squeeze(1)
        elif task == "multiclass":
            self.activation_fn = lambda x: torch.softmax(x, dim=1).squeeze(1)
        else:
            raise ValueError("task not supported. Must be 'binary' or 'multiclass'")

        # metrics
        metric_kwargs = dict(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        metrics = tm.MetricCollection(
            {
                "accuracy": fm.Accuracy(**metric_kwargs),
                "iou": fm.JaccardIndex(**metric_kwargs),
                "recall": fm.Recall(**metric_kwargs),
                "precision": fm.Precision(**metric_kwargs),
                "f1": fm.F1Score(**metric_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    @property
    def backbone(self):
        return self.model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).output

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)

        # Explicitly compute loss in f32 (not bf16, etc.)
        loss = self.loss_fn(logits.float(), y.long().unsqueeze(1))
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = self.activation_fn(logits)
        metrics = getattr(self, f"{phase}_metrics")
        self.log_dict(metrics(probs, y), sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        computed = self.train_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
            sync_dist=True,
        )
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
            sync_dist=True,
        )
        self.val_metrics.reset()

    def configure_optimizers(self):
        return _configure_optimizers(self)


if __name__ == "__main__":
    # Smoke test
    model = TerraTorchSegmentationModel(
        loss="LabelSmoothingLovasz",
        loss_opts=dict(mode="binary", ignore_index=-100),
        num_classes=1,
        ignore_index=-100,
        optimizer_class="torch.optim.AdamW",
        optimizer_opts=dict(lr=3e-4, weight_decay=0.01, betas=[0.9, 0.999]),
        lr_scheduler_class="torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts=dict(max_lr=3e-4, pct_start=0.3),
        lr_scheduler_interval="step",
        freeze_backbone=False,
        model_opts=dict(
            backbone="dofa_large_patch16_224_custom",
            backbone_pretrained=True,
            backbone_wavelengths=[
                0.442,  # Coastal
                0.49,  # Blue
                0.531,  # Green 1
                0.565,  # Green 2
                0.610,  # Yellow
                0.665,  # Red
                0.705,  # RE
                0.865,  # NIR
            ],
            backbone_out_indices=[5, 11, 17, 23],
            necks=[
                dict(name="ReshapeTokensToImage", remove_cls_token=True),
                dict(name="LearnedInterpolateToPyramidal"),
            ],
            decoder="UNetDecoder",
            decoder_channels=[512, 256, 128, 64],
            head_dropout=0.1,
        ),
    )

    x = torch.zeros(2, 8, 224, 224, dtype=torch.float32)
    out = model(x)
    print(out)

    # Smoke-test the layer-decay path via timm's optimizer factory.
    # Uses UperNetDecoder to match the actual production config; the baseline
    # smoke test above uses UNetDecoder for historical reasons.
    from timm.optim import create_optimizer_v2

    model_ld = TerraTorchSegmentationModel(
        loss="LabelSmoothingLovasz",
        loss_opts=dict(mode="binary", ignore_index=-100),
        num_classes=1,
        ignore_index=-100,
        optimizer_class="timm.optim.create_optimizer_v2",
        optimizer_opts=dict(
            opt="adamw",
            lr=1e-4,
            weight_decay=0.01,
            layer_decay=0.75,
            betas=[0.9, 0.99],
        ),
        lr_scheduler_class="torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts=dict(max_lr=1e-4, pct_start=0.3),
        lr_scheduler_interval="step",
        model_opts=dict(
            backbone="dofa_large_patch16_224_custom",
            backbone_pretrained=False,
            backbone_wavelengths=[
                0.442,
                0.49,
                0.531,
                0.565,
                0.610,
                0.665,
                0.705,
                0.865,
            ],
            backbone_out_indices=[5, 11, 17, 23],
            necks=[
                dict(name="ReshapeTokensToImage", remove_cls_token=True),
                dict(name="LearnedInterpolateToPyramidal"),
            ],
            decoder="UperNetDecoder",
            decoder_channels=512,
            head_channel_list=[512],
            head_dropout=0.1,
        ),
    )

    opt = create_optimizer_v2(model_ld, **model_ld.hparams.optimizer_opts)
    groups = opt.param_groups
    scales = [g["lr_scale"] for g in groups]
    # stem + 24 DOFA blocks (trailing norm/head folded in) + decoder/necks
    assert len(set(scales)) == 26, sorted(set(scales))
    assert max(scales) == 1.0, max(scales)
    # Every trainable param is grouped exactly once.
    n_grouped = sum(len(g["params"]) for g in groups)
    n_trainable = sum(1 for p in model_ld.parameters() if p.requires_grad)
    assert n_grouped == n_trainable, (n_grouped, n_trainable)
    print(f"layer-decay smoke test: {len(groups)} groups")
    print(f"  smallest lr_scale: {min(scales):.6f}")
    print(f"  largest  lr_scale: {max(scales):.6f}")
