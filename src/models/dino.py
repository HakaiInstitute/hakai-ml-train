from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    DINOv3ViTConfig,
    DINOv3ViTModel,
    EomtDinov3Config,
    EomtDinov3ForUniversalSegmentation,
)

from . import configure_optimizers as _configure_optimizers


def _load_dinov3_backbone_weights(
    model: EomtDinov3ForUniversalSegmentation,
    pretrained_model_name_or_path: str,
) -> None:
    """Load DINOv3ViT backbone weights into an EomtDinov3 model.

    The DINOv3ViT and EomtDinov3 have nearly identical backbone param names
    differing only by pluralization: layer.N.* -> layers.N.*, norm.* -> layernorm.*.
    """
    vit = DINOv3ViTModel.from_pretrained(pretrained_model_name_or_path)
    vit_state = vit.state_dict()

    renamed = {}
    for key, value in vit_state.items():
        if key == "embeddings.mask_token":
            continue
        new_key = key.replace("layer.", "layers.").replace("norm.", "layernorm.")
        renamed[new_key] = value

    missing, unexpected = model.load_state_dict(renamed, strict=False)
    # We expect missing keys for the segmentation head (query, class_predictor, etc.)
    # and unexpected should be empty
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading backbone weights: {unexpected}"
        )


def _convert_semantic_labels(
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert (B, H, W) class-index labels to EoMT format.

    Returns:
        mask_labels: list of B tensors, each (K, H, W) binary masks
        class_labels: list of B tensors, each (K,) class indices
    """
    B, H, W = y.shape
    mask_labels = []
    class_labels = []

    for i in range(B):
        yi = y[i]  # (H, W)
        masks = []
        classes = []

        for c in range(num_classes):
            mask = (yi == c).float()  # (H, W)
            if ignore_index is not None:
                mask[yi == ignore_index] = 0.0
            masks.append(mask)
            classes.append(c)

        mask_labels.append(torch.stack(masks))  # (K, H, W)
        class_labels.append(torch.tensor(classes, dtype=torch.long, device=y.device))

    return mask_labels, class_labels


def _get_semantic_probs(
    outputs,
    num_classes: int,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Post-process EoMT outputs into (B, C, H, W) semantic probability maps.

    Reproduces the EoMT post-processing: class_probs @ mask_probs -> semantic map.
    """
    # class_queries_logits: (B, Q, num_classes+1) — last class is no-object
    # masks_queries_logits: (B, Q, h, w) — low-res mask logits
    class_logits = outputs.class_queries_logits  # (B, Q, C+1)
    mask_logits = outputs.masks_queries_logits  # (B, Q, h, w)

    # Get class probabilities, excluding the no-object class
    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)

    # Get mask probabilities
    mask_probs = mask_logits.sigmoid()  # (B, Q, h, w)

    # Combine: (B, Q, C) x (B, Q, h, w) -> (B, C, h, w)
    semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    # Interpolate to target size
    semantic = F.interpolate(
        semantic, size=target_size, mode="bilinear", align_corners=False
    )

    return semantic


class DinoBinarySegmentationModel(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="habitat-mapper",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/habitat-mapper",
    docs_url="https://habitat-mapper.readthedocs.io/",
):
    def __init__(
        self,
        model_opts: dict[str, Any],
        num_classes: int = 2,
        num_queries: int = 100,
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

        pretrained_path = model_opts["pretrained_model_name_or_path"]

        # Build EomtDinov3Config from the DINOv3ViT pretrained config
        vit_config = DINOv3ViTConfig.from_pretrained(pretrained_path)
        eomt_config = EomtDinov3Config(
            hidden_size=vit_config.hidden_size,
            intermediate_size=vit_config.intermediate_size,
            num_hidden_layers=vit_config.num_hidden_layers,
            num_attention_heads=vit_config.num_attention_heads,
            patch_size=vit_config.patch_size,
            num_channels=vit_config.num_channels,
            num_register_tokens=vit_config.num_register_tokens,
            hidden_act=vit_config.hidden_act,
            layer_norm_eps=vit_config.layer_norm_eps,
            layerscale_value=vit_config.layerscale_value,
            drop_path_rate=vit_config.drop_path_rate,
            num_labels=num_classes,
            num_queries=num_queries,
        )

        # Create model with random init, then load backbone weights
        self.model = EomtDinov3ForUniversalSegmentation(eomt_config)
        _load_dinov3_backbone_weights(self.model, pretrained_path)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.load_state_dict(ckpt["state_dict"])

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if name.startswith(("embeddings.", "layers.", "layernorm.")):
                    param.requires_grad = False

        self.model = torch.compile(self.model)

        # Metrics
        self.accuracy = fm.Accuracy(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.jaccard_index = fm.JaccardIndex(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.recall = fm.Recall(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.precision = fm.Precision(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.f1_score = fm.F1Score(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(self, x: torch.Tensor):
        return self.model(pixel_values=x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        _, _, H, W = x.shape

        mask_labels, class_labels = _convert_semantic_labels(
            y, self.hparams.num_classes, self.hparams.ignore_index
        )

        outputs = self.model(
            pixel_values=x,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = _get_semantic_probs(outputs, self.hparams.num_classes, (H, W))

        self.log_dict(
            {
                f"{phase}/accuracy": self.accuracy(probs, y),
                f"{phase}/iou": self.jaccard_index(probs, y),
                f"{phase}/recall": self.recall(probs, y),
                f"{phase}/precision": self.precision(probs, y),
                f"{phase}/f1": self.f1_score(probs, y),
            },
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                "val/accuracy_epoch": self.accuracy,
                "val/iou_epoch": self.jaccard_index,
                "val/recall_epoch": self.recall,
                "val/precision_epoch": self.precision,
                "val/f1_epoch": self.f1_score,
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        return _configure_optimizers(self)


class DinoMulticlassSegmentationModel(DinoBinarySegmentationModel):
    def __init__(
        self, *args, class_names: tuple[str, ...] = ("bg", "macro", "nereo"), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

        # Override metrics with per-class (average="none") variants
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

    def on_validation_epoch_end(self) -> None:
        self.log("val/accuracy_epoch", self.accuracy, sync_dist=True)

        iou_per_class = self.jaccard_index.compute()
        precision_per_class = self.precision.compute()
        recall_per_class = self.recall.compute()
        f1_per_class = self.f1_score.compute()

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i], sync_dist=True)
            self.log(
                f"val/recall_epoch/{class_name}", recall_per_class[i], sync_dist=True
            )
            self.log(
                f"val/precision_epoch/{class_name}",
                precision_per_class[i],
                sync_dist=True,
            )
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i], sync_dist=True)
        self.log("val/iou_epoch", iou_per_class[1:].mean(), sync_dist=True)

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        _, _, H, W = x.shape

        mask_labels, class_labels = _convert_semantic_labels(
            y, self.hparams.num_classes, self.hparams.ignore_index
        )

        outputs = self.model(
            pixel_values=x,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = _get_semantic_probs(outputs, self.hparams.num_classes, (H, W))

        self.log(f"{phase}/accuracy", self.accuracy(probs, y), sync_dist=True)

        iou_per_class = self.jaccard_index(probs, y)
        precision_per_class = self.precision(probs, y)
        recall_per_class = self.recall(probs, y)
        f1_per_class = self.f1_score(probs, y)
        for i, class_name in enumerate(self.class_names):
            self.log_dict(
                {
                    f"{phase}/iou_{class_name}": iou_per_class[i],
                    f"{phase}/recall_{class_name}": recall_per_class[i],
                    f"{phase}/precision_{class_name}": precision_per_class[i],
                    f"{phase}/f1_{class_name}": f1_per_class[i],
                },
                sync_dist=True,
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean(), sync_dist=True)

        return loss
