import math
import os
from argparse import Namespace
from pathlib import Path

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t
from utils.eval import predict_tiff
from utils.loss import iou, focal_tversky_loss


class DeepLabv3Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Create model from pre-trained DeepLabv3
        self.model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.model.requires_grad_(False)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)
        self.model.classifier.requires_grad_(True)

        self.model.aux_classifier = FCNHead(1024, self.hparams.num_classes)
        self.model.aux_classifier.requires_grad_(True)

        if self.hparams.unfreeze_backbone_epoch == 0:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        steps_per_epoch = math.floor(len(self.train_dataloader()))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.hparams.epochs, steps_per_epoch=steps_per_epoch)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        ds_train = SegmentationDataset(self.hparams.train_data_dir, transform=t.train_transforms,
                                       target_transform=t.train_target_transforms)
        return DataLoader(ds_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True,
                          drop_last=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        ds_val = SegmentationDataset(self.hparams.val_data_dir, transform=t.test_transforms,
                                     target_transform=t.test_target_transforms)
        return DataLoader(ds_val, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=os.cpu_count())

    def calc_loss(self, p, g):
        return focal_tversky_loss(p.permute(0, 2, 3, 1).reshape((-1, self.hparams.num_classes)), g.flatten(),
                                  alpha=0.3, beta=0.7, gamma=4. / 3.)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        logits = y_hat['out']
        scores = torch.softmax(logits, dim=1)
        loss = self.calc_loss(scores, y)

        aux_logits = y_hat['aux']
        aux_scores = torch.softmax(aux_logits, dim=1)
        aux_loss = self.calc_loss(aux_scores, y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        ious = iou(logits.float(), y)
        return {'loss': loss, 'ious': ious}

    def training_epoch_end(self, outputs):
        # Allow fine-tuning of backbone layers after 150 epochs
        if self.current_epoch == self.hparams.unfreeze_backbone_epoch - 1:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_ious = torch.stack([x['ious'] for x in outputs]).mean(dim=0)
        avg_mious = avg_ious.mean()

        tensorboard_logs = {
            'loss': avg_loss,
            'miou': avg_mious,
            'macro_iou': avg_ious[0],
            'nereo_iou': avg_ious[1],
            'bg_iou': avg_ious[2]
        }
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        logits = y_hat['out']
        scores = torch.softmax(logits, dim=1)
        loss = self.calc_loss(scores, y)

        ious = iou(logits.float(), y)
        return {'val_loss': loss, 'val_ious': ious}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ious = torch.stack([x['val_ious'] for x in outputs]).mean(dim=0)
        avg_mious = avg_ious.mean()

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_miou': avg_mious,
            'val_macro_iou': avg_ious[0],
            'val_nereo_iou': avg_ious[1],
            'val_bg_iou': avg_ious[2]
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def _get_checkpoint(checkpoint_dir, name=""):
    versions_dir = Path(checkpoint_dir).joinpath(name)
    checkpoint_files = list(versions_dir.glob("*/checkpoints/best_val_miou_*.ckpt"))
    if len(checkpoint_files) > 0:
        return str(sorted(checkpoint_files)[-1])
    else:
        return False


def train(train_data_dir, val_data_dir, checkpoint_dir,
          num_classes=3, batch_size=4, lr=0.001, weight_decay=1e-4, epochs=310, aux_loss_factor=0.3,
          accumulate_grad_batches=1, gradient_clip_val=0, precision=32, amp_level='O1', auto_lr_find=False,
          unfreeze_backbone_epoch=0, auto_scale_batch_size=False, name=""):
    """
    Train the DeepLabV3 Kelp Detection model.
    Args:
        train_data_dir: Path to the directory containing subdirectories x and y containing the training dataset.
        val_data_dir: Path to the directory containing subdirectories x and y containing the validation dataset.
        checkpoint_dir: Path to the directory where tensorboard logging and model checkpoint outputs should go.
        num_classes: The number of classes in the dataset. Defaults to 2.
        batch_size: The batch size for training. Multiplied by # GPUs for DDP. Defaults to 4.
        lr: The learning rate. Defaults to 0.001.
        weight_decay: The amount of L2 regularization on model parameters. Defaults to 1e-4.
        epochs: The number of training epochs. Defaults to 310.
        aux_loss_factor: The weight for the auxiliary loss to encourage could features in early layers. Default 0.3.
        accumulate_grad_batches: The number of gradients batches to accumulate before backprop. Defaults to 1 (No accumulation).
        gradient_clip_val: The value of the gradient norm at which backprop should be skipped if over that value. Defaults to 0 (None).
        precision: The floating point precision of model weights. Defaults to 32. Set to 16 for AMP training with Nvidia Apex.
        amp_level: The AMP level in NVidia Apex. Defaults to "O1" (i.e. letter O, number 1). See Apex docs to details.
        auto_lr_find: Flag on wether or not to run the LR finder at the beginning of training. Defaults to False.
        unfreeze_backbone_epoch: The epoch at which blocks 3 and 4 of the backbone network should start adjusting parameters. Defaults to 150.
        auto_scale_batch_size: Run a heuristic to maximize batch size per GPU. Defaults to False.
        name: The name of the model. Creates a subdirectory in the checkpoint dir with this name. Defaults to "".

    Returns: None. Side effects include logging and checkpointing models to the checkpoint directory.
    
    """
    os.environ['TORCH_HOME'] = str(Path(checkpoint_dir).parent)

    logger = TensorBoardLogger(Path(checkpoint_dir), name=name)

    lr_logger_callback = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_miou',
        mode='max',
        prefix='best_val_miou_'
    )

    hparams = Namespace(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        aux_loss_factor=aux_loss_factor,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        amp_level=amp_level,
        unfreeze_backbone_epoch=unfreeze_backbone_epoch,
    )

    trainer_kwargs = {
        'gpus': torch.cuda.device_count() if torch.cuda.is_available() else None,
        'distributed_backend': 'ddp' if torch.cuda.is_available() else None,
        'checkpoint_callback': checkpoint_callback,
        'logger': logger,
        'early_stop_callback': False,
        'deterministic': True,
        'default_root_dir': checkpoint_dir,
        'accumulate_grad_batches': accumulate_grad_batches,
        'gradient_clip_val': gradient_clip_val,
        'max_epochs': epochs,
        'precision': precision,
        'amp_level': amp_level,
        'auto_scale_batch_size': auto_scale_batch_size,
        'callbacks': [lr_logger_callback],
    }

    # If checkpoint exists, resume
    checkpoint = _get_checkpoint(checkpoint_dir, name)
    if checkpoint:
        print("Loading checkpoint:", checkpoint)
        model = DeepLabv3Model.load_from_checkpoint(checkpoint, hparams=hparams)
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint, **trainer_kwargs)
    else:
        model = DeepLabv3Model(hparams)
        trainer = pl.Trainer(auto_lr_find=auto_lr_find, **trainer_kwargs)

    trainer.fit(model)


def predict(seg_in, seg_out, weights, batch_size=4, crop_size=300, crop_pad=150):
    """
    Segment the FG class using DeepLabv3 Segmentation model of an input .tif image.
    Args:
        weights: Path to a model weights file (*.pt). Required for eval and pred mode.
        seg_in: Path to a *.tif image to do segmentation on in pred mode.
        seg_out: Path to desired output *.tif created by the model in pred mode.
        batch_size: The batch size per GPU. Defaults to 4.
        crop_size: The crop size in pixels for processing the image. Defines the length and width of the individual
            sections the input .tif image is cropped to for processing. Defaults to 300.
        crop_pad: The amount of padding added for classification context to each image crop. The output classification
            on this crop area is not output by the model but will influence the classification of the area in the
            (crop_size x crop_size) window. Defaults to 150.

    Note: Changing the crop_size and crop_pad parameters may require adjusting the batch_size such that the image
        batches fit into the GPU memory.

    Returns: None. Outputs various files as a side effect depending on the script mode.
    """
    seg_in, seg_out, weights = Path(seg_in), Path(seg_out), Path(weights)
    os.environ['TORCH_HOME'] = str(weights.parent)

    # Create output directory as required
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    # Load model and weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepLabv3Model.load_from_checkpoint(weights)
    model.freeze()
    model = model.to(device)

    print("Processing:", seg_in)
    batch_size = batch_size * max(torch.cuda.device_count(), 1)
    predict_tiff(model, device, seg_in, seg_out,
                 transform=t.test_transforms,
                 crop_size=crop_size, pad=crop_pad,
                 batch_size=batch_size)


if __name__ == '__main__':
    seed_everything(0)
    fire.Fire({
        'train': train,
        'pred': predict
    })
