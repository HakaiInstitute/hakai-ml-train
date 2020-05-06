#!/usr/bin/env python
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import deeplabv3
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t
from utils.eval import predict_tiff
from utils.loss import dice_loss
from utils.loss import iou

# Hyper-parameter defaults
NUM_CLASSES = 2
NUM_EPOCHS = 200
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 0.01
PRED_CROP_SIZE = 300
PRED_CROP_PAD = 150
RESTART_TRAINING = True
AUX_LOSS_FACTOR = 0.3

DOCKER = bool(os.environ.get('DOCKER', False))
DISABLE_CUDA = False
MODEL_NAME = "deeplabv3"

if not DISABLE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Make results reproducible
torch.manual_seed(0)
# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def get_dataloaders(mode, train_data_dir, eval_data_dir, batch_size=BATCH_SIZE):
    if mode == "train":
        transforms = t.train_transforms
        target_transforms = t.train_target_transforms
    elif mode == "eval":
        transforms = t.test_transforms
        target_transforms = t.test_target_transforms
    else:
        raise ValueError("mode must be 'train' or 'eval'")

    ds_train = SegmentationDataset(train_data_dir, transform=transforms,
                                   target_transform=target_transforms)
    ds_val = SegmentationDataset(eval_data_dir, transform=t.test_transforms,
                                 target_transform=t.test_target_transforms)

    dataloader_opts = {
        "batch_size": batch_size * max(torch.cuda.device_count(), 1),
        "pin_memory": True,
        "drop_last": mode == "train",
        "num_workers": os.cpu_count()
    }
    return {
        'train': DataLoader(ds_train, shuffle=(mode == "train"), **dataloader_opts),
        'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
    }


class TensorboardWriters(object):
    def __init__(self, device, log_dir):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.device = device
        self.best_miou = {'train': 0, 'eval': 0}

    def __enter__(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(self.log_dir, current_time + '_' + socket.gethostname())

        self.writers = {
            'train': SummaryWriter(log_dir=log_dir + '_train'),
            'eval': SummaryWriter(log_dir=log_dir + '_eval')
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for phase in ['train', 'eval']:
            self.writers[phase].flush()
            self.writers[phase].close()

    def update(self, phase, epoch, mloss, miou, iou_bg, iou_fg, x, y, pred):
        writer = self.writers[phase]
        writer.add_scalar('Loss', mloss, epoch)
        writer.add_scalar('IoU/Mean', miou, epoch)
        writer.add_scalar('IoU/BG', iou_bg, epoch)
        writer.add_scalar('IoU/FG', iou_fg, epoch)

        # Show images for best miou models only to save space
        if miou > self.best_miou[phase]:
            self.best_miou[phase] = miou

            x = x[:, -3:, :, :]
            img_grid = torchvision.utils.make_grid(x, nrow=8)
            img_grid = t.inv_normalize(img_grid)

            y = y.unsqueeze(dim=1)
            label_grid = torchvision.utils.make_grid(y, nrow=8).to(self.device)
            label_grid = alpha_blend(img_grid, label_grid)
            writer.add_image('Labels/True', label_grid.detach().cpu(), epoch)

            pred = pred.max(dim=1)[1].unsqueeze(dim=1)
            pred_grid = torchvision.utils.make_grid(pred, nrow=8).to(self.device)
            pred_grid = alpha_blend(img_grid, pred_grid)
            writer.add_image('Labels/Pred', pred_grid.detach().cpu(), epoch)


def train_one_epoch(model, device, optimizer, lr_scheduler, dataloader, epoch, writers, checkpoint_dir):
    model.train()
    sum_loss = 0.
    sum_iou = np.zeros(NUM_CLASSES)
    iters = len(dataloader)

    for i, (x, y) in enumerate(tqdm(iter(dataloader), desc=f"train epoch {epoch}", file=sys.stdout)):
        y = y.to(device)
        x = x.to(device)

        optimizer.zero_grad()

        pred = model(x)
        logits = pred['out']
        aux_logits = pred['aux']
        scores = torch.softmax(logits, dim=1)
        loss = dice_loss(scores[:, 1], y)

        aux_scores = torch.softmax(aux_logits, dim=1)
        aux_loss = dice_loss(aux_scores[:, 1], y)
        loss = loss + AUX_LOSS_FACTOR * aux_loss

        loss.backward()
        optimizer.step()

        # Compute metrics
        sum_loss += loss.detach().cpu().item()
        sum_iou += iou(y, logits.float()).detach().cpu().numpy()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + i / iters)

    mloss = sum_loss / len(dataloader)
    ious = np.around(sum_iou / len(dataloader), 4)
    miou = np.mean(ious)
    iou_bg = sum_iou[0] / len(dataloader)
    iou_fg = sum_iou[1] / len(dataloader)

    print(f'train-loss={mloss}; train-miou={miou}; train-iou-bg={iou_bg}; train-iou-fg={iou_fg};')
    if len(dataloader):
        # noinspection PyUnboundLocalVariable
        writers.update('train', epoch, mloss, miou, iou_bg, iou_fg, x, y, scores)
    else:
        raise RuntimeWarning("No data in train dataloader")

    # Save model checkpoints
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mean_eval_loss': mloss,
    }, Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_checkpoint.pt'))

    return mloss, miou, iou_bg, iou_fg


def validate_one_epoch(model, device, dataloader, epoch, writers=None):
    model.eval()
    sum_loss = 0.
    sum_iou = np.zeros(NUM_CLASSES)

    for x, y in tqdm(iter(dataloader), desc=f"eval epoch {epoch}", file=sys.stdout):
        with torch.no_grad():
            y = y.to(device)
            x = x.to(device)

            logits = model(x)['out']
            scores = torch.softmax(logits, dim=1)
            loss = dice_loss(scores[:, 1], y)

        # Compute metrics
        sum_loss += loss.detach().cpu().item()
        sum_iou += iou(y, logits.float()).detach().cpu().numpy()

    mloss = sum_loss / len(dataloader)
    ious = np.around(sum_iou / len(dataloader), 4)
    miou = np.mean(ious)
    iou_bg = sum_iou[0] / len(dataloader)
    iou_fg = sum_iou[1] / len(dataloader)

    print(f'eval-loss={mloss}; eval-miou={miou}; eval-iou-bg={iou_bg}; eval-iou-fg={iou_fg};')
    if not len(dataloader):
        raise RuntimeWarning("No data in eval dataloader")
    elif writers is not None:
        # noinspection PyUnboundLocalVariable
        writers.update('eval', epoch, mloss, miou, iou_bg, iou_fg, x, y, scores)

    return mloss, miou, iou_bg, iou_fg


def train(train_data_dir, eval_data_dir, checkpoint_dir,
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY):
    """
    Train the DeepLabv3 Segmentation model.
    Args:
        train_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            training images.
        eval_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            eval images.
        checkpoint_dir: Path to save model checkpoints so training can be resumed. Training will automatically search
            this directory for checkpoints on start.
        epochs: Number of training epochs to do. Defaults to 200.
        batch_size: The batch size for each GPU. Defaults to 4.
        lr: The learning rate. Defaults to 1e-4.
        weight_decay: The amount of L2 regularization. Defaults to 0.01.

    Returns: None.
    """
    train_data_dir, eval_data_dir, checkpoint_dir = Path(train_data_dir), Path(eval_data_dir), Path(checkpoint_dir)

    # Load model
    os.environ['TORCH_HOME'] = str(checkpoint_dir.parents[0])
    model = deeplabv3.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)

    # Get dataset loaders, optimizer, lr_scheduler
    dataloaders = get_dataloaders("train", train_data_dir, eval_data_dir, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Restart at checkpoint if it exists
    checkpoint_path = checkpoint_dir.joinpath(f'{MODEL_NAME}_checkpoint.pt')
    if Path(checkpoint_path).exists() and RESTART_TRAINING:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # Train loop
    best_val_loss = None
    best_val_miou = 0
    best_train_loss = None
    best_train_miou = 0

    with TensorboardWriters(DEVICE, log_dir=checkpoint_dir.joinpath('runs')) as writers:
        for epoch in range(start_epoch, epochs):
            # Fine Tune backbone after 150 epochs
            if epoch >= 150:
                model.module.backbone.layer3.requires_grad_(True)
                model.module.backbone.layer4.requires_grad_(True)

            mloss, miou, iou_bg, iou_fg = train_one_epoch(model, DEVICE, optimizer, lr_scheduler, dataloaders['train'],
                                                          epoch, writers, checkpoint_dir)
            # Save best models
            if best_train_loss is None or mloss < best_train_loss:
                best_train_loss = mloss
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_train_loss.pt'))
            if miou > best_train_miou:
                best_train_miou = miou
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_train_miou.pt'))

            mloss, miou, iou_bg, iou_fg = validate_one_epoch(model, DEVICE, dataloaders['eval'], epoch, writers)
            # Save best models
            if best_val_loss is None or mloss < best_val_loss:
                best_val_loss = mloss
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_val_loss.pt'))
            if miou > best_val_miou:
                best_val_miou = miou
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_val_miou.pt'))

    # Save final model params
    torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f"{MODEL_NAME}_final.pt"))


def evaluate(train_data_dir, eval_data_dir, weights, batch_size=BATCH_SIZE):
    """
    Evaluate DeepLabv3 Segmentation model performance.
    Args:
        train_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            training images. Required for train and eval mode.
        eval_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            eval images. Required for train and eval mode.
        weights: Path to a model weights file (*.pt). Required for eval and pred mode.
        batch_size: The data batch size per GPU. Defaults to 4.

    Returns: None.
    """
    train_data_dir, eval_data_dir, weights = Path(train_data_dir), Path(eval_data_dir), Path(weights)
    os.environ['TORCH_HOME'] = str(weights.parents[0])

    model = deeplabv3.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    dataloaders = get_dataloaders("eval", train_data_dir, eval_data_dir, batch_size=batch_size)
    for phase in ['train', 'eval']:
        print(phase.capitalize())
        validate_one_epoch(model, DEVICE, dataloaders[phase], 0)


def predict(seg_in, seg_out, weights, batch_size=BATCH_SIZE, crop_size=PRED_CROP_SIZE, crop_pad=PRED_CROP_PAD):
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
    # Model
    seg_in, seg_out, weights = Path(seg_in), Path(seg_out), Path(weights)
    os.environ['TORCH_HOME'] = str(weights.parents[0])

    model = deeplabv3.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    print("Processing:", seg_in)
    batch_size = batch_size * max(torch.cuda.device_count(), 1)
    predict_tiff(model, DEVICE, seg_in, seg_out,
                 transform=t.test_transforms,
                 crop_size=crop_size, pad=crop_pad,
                 batch_size=batch_size)


if __name__ == '__main__':
    print("Using device:", DEVICE)
    print(torch.cuda.device_count(), "GPU(s) detected")

    fire.Fire({
        "train": train,
        "eval": evaluate,
        "pred": predict
    })
