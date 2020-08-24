import os
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from kelp.presence.model import KelpModel
from utils.checkpoint import get_checkpoint

pl.seed_everything(0)


def train(train_data_dir, val_data_dir, checkpoint_dir,
          num_classes=2, batch_size=4, lr=0.001, weight_decay=1e-4, epochs=310, aux_loss_factor=0.3,
          accumulate_grad_batches=1, gradient_clip_val=0, precision=32, amp_level='O2', auto_lr_find=False,
          unfreeze_backbone_epoch=0, auto_scale_batch_size=False, overfit_batches=None, name=""):
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
        overfit_batches: The number or percentage of batches to train on. Useful for debugging.
        name: The name of the model. Creates a subdirectory in the checkpoint dir with this name. Defaults to "".

    Returns: None. Side effects include logging and checkpointing models to the checkpoint directory.

    """
    os.environ['TORCH_HOME'] = str(Path(checkpoint_dir).parent)

    logger = TensorBoardLogger(Path(checkpoint_dir), name=name)

    lr_logger_callback = pl.callbacks.LearningRateLogger()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor='val_miou',
        mode='max',
        prefix='best_val_miou_',
        save_top_k=1,
        save_last=True,
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
        'amp_level': amp_level,
        'precision': precision,
        'checkpoint_callback': checkpoint_callback,
        'logger': logger,
        'early_stop_callback': False,
        'deterministic': True,
        'default_root_dir': checkpoint_dir,
        'accumulate_grad_batches': accumulate_grad_batches,
        'gradient_clip_val': gradient_clip_val,
        'max_epochs': epochs,
        'auto_scale_batch_size': auto_scale_batch_size,
        'callbacks': [lr_logger_callback]
    }

    if overfit_batches is not None:
        trainer_kwargs['overfit_batches'] = overfit_batches

    # If checkpoint exists, resume
    checkpoint = get_checkpoint(checkpoint_dir, name)
    if checkpoint:
        print("Loading checkpoint:", checkpoint)
        model = KelpModel.load_from_checkpoint(checkpoint, hparams=hparams)
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint, **trainer_kwargs)
    else:
        model = KelpModel(hparams)
        trainer = pl.Trainer(auto_lr_find=auto_lr_find, **trainer_kwargs)

    trainer.fit(model)
