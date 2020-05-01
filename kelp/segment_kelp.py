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
from utils.eval import eval_model, predict_tiff
from utils.loss import assymetric_tversky_loss
from utils.loss import iou

# Hyper-parameters
NUM_CLASSES = 2
NUM_EPOCHS = 200
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 0.01
PRED_CROP_SIZE = 300
PRED_CROP_PAD = 150
RESTART_TRAINING = True

DOCKER = bool(os.environ.get('DOCKER', False))
DISABLE_CUDA = False
MODEL_NAME = "deeplabv3"

if not DISABLE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# TRAIN_DATA_DIR = Path("/opt/ml/input/data/train")
# EVAL_DATA_DIR = Path("/opt/ml/input/data/eval")
# TRAINED_WEIGHTS = Path("/opt/ml/input/weights.pt")
# SEGMENTATION_INPUT_IMG = Path("/opt/segmentation/input/image.tif")
# SEGMENTATION_OUT_DIR = Path("/opt/segmentation/output")
# CHECKPOINT_DIR = Path('/opt/ml/output/checkpoints')
# WEIGHTS_OUT_DIR = Path('/opt/ml/output/model')

# Make results reproducible
torch.manual_seed(0)
# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def get_dataloaders(mode, train_data_dir, eval_data_dir):
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
        "batch_size": BATCH_SIZE * max(torch.cuda.device_count(), 1),
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

    def __enter__(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(self.log_dir.joinpath('runs', current_time + '_' + socket.gethostname()))

        self.writers = {
            'train': SummaryWriter(log_dir=log_dir + '_train'),
            'eval': SummaryWriter(log_dir=log_dir + '_eval')
        }
        return self

    def __exit__(self):
        for phase in ['train', 'eval']:
            self.writers[phase].flush()
            self.writers[phase].close()

    def update(self, phase, epoch, mloss, miou, iou_bg, iou_fg, x, y, pred):
        writer = self.writers[phase]
        writer.add_scalar('Loss', mloss, epoch)
        writer.add_scalar('IoU/Mean', miou, epoch)
        writer.add_scalar('IoU/BG', iou_bg, epoch)
        writer.add_scalar('IoU/FG', iou_fg, epoch)

        # Show images
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

    for x, y in tqdm(iter(dataloader), desc=f"train epoch {epoch}", file=sys.stdout):
        y = y.to(device)
        x = x.to(device)

        optimizer.zero_grad()

        pred = model(x)['out']
        scores = torch.softmax(pred, dim=1)
        loss = assymetric_tversky_loss(scores[:, 1], y, beta=1.)

        loss.backward()
        optimizer.step()

        # Compute metrics
        sum_loss += loss.detach().cpu().item()
        sum_iou += iou(y, pred.float()).detach().cpu().numpy()

    if lr_scheduler is not None:
        lr_scheduler.step(epoch)

    mloss = sum_loss / len(dataloader)
    ious = np.around(sum_iou / len(dataloader), 4)
    miou = np.mean(ious)
    iou_bg = sum_iou[0] / len(dataloader)
    iou_fg = sum_iou[1] / len(dataloader)

    print(f'train-loss={mloss}; train-miou={miou}; train-iou-bg={iou_bg}; train-iou-fg={iou_fg};')
    if len(dataloader):
        # noinspection PyUnboundLocalVariable
        writers.update('train', epoch, mloss, miou, iou_bg, iou_fg, x, y, pred)
    else:
        raise RuntimeWarning("No data in train dataloader")

    # Save model checkpoints
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mean_eval_loss': mloss,
    }, Path(checkpoint_dir).joinpath(f'{MODEL_NAME}.pt'))

    return model


def validate_one_epoch(model, device, dataloader, epoch, writers):
    model.eval()
    sum_loss = 0.
    sum_iou = np.zeros(NUM_CLASSES)

    for x, y in tqdm(iter(dataloader), desc=f"eval epoch {epoch}", file=sys.stdout):
        with torch.no_grad():
            y = y.to(device)
            x = x.to(device)

            pred = model(x)['out']
            sig = torch.sigmoid(pred[:, 1])
            loss = assymetric_tversky_loss(sig, y, beta=1.)

        # Compute metrics
        sum_loss += loss.detach().cpu().item()
        sum_iou += iou(y, pred.float()).detach().cpu().numpy()

    mloss = sum_loss / len(dataloader)
    ious = np.around(sum_iou / len(dataloader), 4)
    miou = np.mean(ious)
    iou_bg = sum_iou[0] / len(dataloader)
    iou_fg = sum_iou[1] / len(dataloader)

    print(f'eval-loss={mloss}; eval-miou={miou}; eval-iou-bg={iou_bg}; eval-iou-fg={iou_fg};')
    if len(dataloader):
        writers.update('train', epoch, mloss, miou, iou_bg, iou_fg, x, y, pred)
    else:
        raise RuntimeWarning("No data in eval dataloader")
    # Save best models for eval set
    return mloss, miou, iou_bg, iou_fg


def train(train_data_dir, eval_data_dir, checkpoint_dir):
    """
    Train the DeepLabv3 Segmentation model.
    Args:
        train_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            training images. Required for train and eval mode.
        eval_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            eval images. Required for train and eval mode.
        checkpoint_dir: Path to save model checkpoints so training can be resumed. Training will automatically search
            this directory for checkpoints on start. Required for train mode.

    Returns: None.
    """
    train_data_dir, eval_data_dir, checkpoint_dir = Path(train_data_dir), Path(eval_data_dir), Path(checkpoint_dir)

    os.environ['TORCH_HOME'] = str(checkpoint_dir.parents[0])
    model = deeplabv3.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)

    dataloaders = get_dataloaders("train", train_data_dir, eval_data_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    checkpoint_path = checkpoint_dir.joinpath(f'{MODEL_NAME}.pt')

    # Restart at checkpoint if it exists
    if Path(checkpoint_path).exists() and RESTART_TRAINING:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_log_dir = os.path.join(checkpoint_dir.joinpath('runs', current_time + '_' + socket.gethostname()))

    best_val_loss = None
    best_val_iou_fg = None
    best_val_miou = None

    with TensorboardWriters(DEVICE, tensorboard_log_dir) as writers:
        for epoch in range(start_epoch, NUM_EPOCHS):
            model = train_one_epoch(model, DEVICE, optimizer, lr_scheduler, dataloaders['train'], epoch, writers,
                                    checkpoint_dir)
            mloss, miou, iou_bg, iou_fg = validate_one_epoch(model, DEVICE, dataloaders['eval'], epoch, writers)

            # Save best models
            if best_val_loss is None or mloss < best_val_loss:
                best_val_loss = mloss
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_val_loss.pt'))
            if best_val_iou_fg is None or iou_fg < best_val_iou_fg:
                best_val_iou_fg = iou_fg
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_val_fg_iou.pt'))
            if best_val_miou is None or miou < best_val_miou:
                best_val_miou = miou
                torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f'{MODEL_NAME}_best_val_miou.pt'))

    torch.save(model.state_dict(), Path(checkpoint_dir).joinpath(f"{MODEL_NAME}_final.pt"))


def evaluate(train_data_dir, eval_data_dir, weights):
    """
    Evaluate DeepLabv3 Segmentation model performance.
    Args:
        train_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            training images. Required for train and eval mode.
        eval_data_dir: Path to the training data directory that contains subdirectories x/ and y/ with all .png
            eval images. Required for train and eval mode.
        weights: Path to a model weights file (*.pt). Required for eval and pred mode.

    Returns: None.
    """
    train_data_dir, eval_data_dir, weights = Path(train_data_dir), Path(eval_data_dir), Path(weights)
    os.environ['TORCH_HOME'] = str(weights.parents[0])

    model = deeplabv3.create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    dataloaders = get_dataloaders("eval", train_data_dir, eval_data_dir)
    eval_model(model, DEVICE, dataloaders, NUM_CLASSES)


def predict(seg_in, seg_out, weights):
    """
    Segment the FG class using DeepLabv3 Segmentation model.
    Args:
        weights: Path to a model weights file (*.pt). Required for eval and pred mode.
        seg_in: Path to a *.tif image to do segmentation on in pred mode.
        seg_out: Path to desired output *.tif created by the model in pred mode.

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
    batch_size = BATCH_SIZE * max(torch.cuda.device_count(), 1)
    predict_tiff(model, DEVICE, seg_in, seg_out,
                 transform=t.test_transforms,
                 crop_size=PRED_CROP_SIZE, pad=PRED_CROP_PAD,
                 batch_size=batch_size)


if __name__ == '__main__':
    print("Using device:", DEVICE)
    print(torch.cuda.device_count(), "GPU(s) detected")

    fire.Fire({
        "train": train,
        "eval": evaluate,
        "pred": predict
    })
