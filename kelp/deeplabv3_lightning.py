import os
from argparse import Namespace
from pathlib import Path

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t
from utils.loss import dice_loss, iou

PRED_CROP_SIZE = 300
PRED_CROP_PAD = 150


class DeepLabv3Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Create model from pre-trained DeepLabv3
        self.model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.model.requires_grad_(False)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)
        self.model.classifier.requires_grad_(True)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        ds_train = SegmentationDataset(self.hparams.train_data_dir, transform=t.train_transforms,
                                       target_transform=t.train_target_transforms)
        return DataLoader(ds_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True,
                          drop_last=True,
                          num_workers=os.cpu_count())

    def val_dataloader(self):
        ds_val = SegmentationDataset(self.hparams.val_data_dir, transform=t.test_transforms,
                                     target_transform=t.test_target_transforms)
        return DataLoader(ds_val, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=os.cpu_count())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        logits = y_hat['out']
        scores = torch.softmax(logits, dim=1)
        loss = dice_loss(scores[:, 1], y)

        aux_logits = y_hat['aux']
        aux_scores = torch.softmax(aux_logits, dim=1)
        aux_loss = dice_loss(aux_scores[:, 1], y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        ious = iou(y, logits.float())
        return {'loss': loss, 'ious': ious}

    def training_epoch_end(self, outputs):
        # Allow fine-tuning of backbone layers after 150 epochs
        if self.current_epoch == self.hparams.unfreeze_backbone_epoch - 1:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_ious = torch.stack([x['ious'] for x in outputs]).mean(dim=0)
        avg_mious = avg_ious.mean()
        avg_fg_ious = avg_ious[0]
        avg_bg_ious = avg_ious[1]

        tensorboard_logs = {
            'loss': avg_loss,
            'miou': avg_mious,
            'fg_iou': avg_fg_ious,
            'bg_iou': avg_bg_ious
        }
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        logits = y_hat['out']
        scores = torch.softmax(logits, dim=1)
        loss = dice_loss(scores[:, 1], y)

        ious = iou(y, logits.float())
        return {'val_loss': loss, 'val_ious': ious}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ious = torch.stack([x['val_ious'] for x in outputs]).mean(dim=0)
        avg_mious = avg_ious.mean()
        avg_fg_ious = avg_ious[0]
        avg_bg_ious = avg_ious[1]

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_miou': avg_mious,
            'val_fg_iou': avg_fg_ious,
            'val_bg_iou': avg_bg_ious
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def _get_checkpoint(checkpoint_dir):
    checkpoint_files = list(Path(checkpoint_dir).glob("best_val_miou_*.ckpt"))
    if len(checkpoint_files) > 0:
        return str(checkpoint_files[0])
    else:
        return False
        print("Loading checkpoint:", checkpoint)


def train(train_data_dir, val_data_dir, checkpoint_dir,
          num_classes=2, batch_size=4, lr=0.001, weight_decay=1e-4, epochs=310, aux_loss_factor=0.3,
          accumulate_grad_batches=1, precision=32, amp_level='O1', auto_lr_find=False, unfreeze_backbone_epoch=150,
          auto_scale_batch_size=False, restart=True):
    os.environ['TORCH_HOME'] = str(Path(checkpoint_dir).parent)
    logger = TensorBoardLogger(Path(checkpoint_dir).joinpath('runs'), name="")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir,
        save_top_k=True,
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
        precision=precision,
        amp_level=amp_level,
        unfreeze_backbone_epoch=unfreeze_backbone_epoch,
    )

    trainer_kwargs = {
        'gpus': list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None,
        'checkpoint_callback': checkpoint_callback,
        'logger': logger,
        'early_stop_callback': False,
        'deterministic': True,
        'default_root_dir': checkpoint_dir,
        'accumulate_grad_batches': accumulate_grad_batches,
        'max_epochs': epochs,
        'precision': precision,
        'amp_level': amp_level,
        'auto_scale_batch_size': auto_scale_batch_size,
    }

    # If checkpoint exists, resume
    checkpoint = _get_checkpoint(checkpoint_dir)
    if restart and checkpoint:
        print("Loading checkpoint:", checkpoint)
        model = DeepLabv3Model.load_from_checkpoint(checkpoint)
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint, **trainer_kwargs)
    else:
        model = DeepLabv3Model(hparams)
        trainer = pl.Trainer(auto_lr_find=auto_lr_find, **trainer_kwargs)

    trainer.fit(model)


def evaluate():
    # TODO: Implement eval method
    pass


def predict():
    # TODO: Implement pred method
    pass


if __name__ == '__main__':
    seed_everything(0)
    print("GPUS available:", torch.cuda.is_available())
    print("# GPUs:", torch.cuda.device_count())

    fire.Fire({
        'train': train,
        'eval': evaluate,
        'pred': predict
    })
