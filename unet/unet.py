#!/usr/bin/env python
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.UNet import UNet
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as T, add_veg_indices
from utils.loss import assymetric_tversky_loss
from utils.loss import iou


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def update_tensorboard(writer, device, epoch, mloss, miou, iou_bg, iou_seagrass, x, y, pred):
    writer.add_scalar('Loss', mloss, epoch)
    writer.add_scalar('IoU/Mean', miou, epoch)
    writer.add_scalar('IoU/BG', iou_bg, epoch)
    writer.add_scalar('IoU/Seagrass', iou_seagrass, epoch)

    # Show images
    x = x[:, -3:, :, :]
    img_grid = torchvision.utils.make_grid(x, nrow=8)
    img_grid = T.inv_normalize(img_grid)

    y = y.unsqueeze(dim=1)
    label_grid = torchvision.utils.make_grid(y, nrow=8).to(device)
    label_grid = alpha_blend(img_grid, label_grid)
    writer.add_image('Labels/True', label_grid.detach().cpu(), epoch)

    pred = pred.max(dim=1)[1].unsqueeze(dim=1)
    pred_grid = torchvision.utils.make_grid(pred, nrow=8).to(device)
    pred_grid = alpha_blend(img_grid, pred_grid)
    writer.add_image('Labels/Pred', pred_grid.detach().cpu(), epoch)


def train_one_epoch(model, device, optimizer, lr_scheduler, dataloader, epoch, writer, checkpoint_dir):
    model.train()
    sum_loss = 0.
    sum_iou = np.zeros(num_classes)

    for x, y in tqdm(iter(dataloader), desc=f"train epoch {epoch}", file=sys.stdout):
        y = y.to(device)
        x = x.to(device)
        x = add_veg_indices(x)

        optimizer.zero_grad()

        pred = model(x)
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
    iou_seagrass = sum_iou[1] / len(dataloader)

    print(f'train-loss={mloss}; train-miou={miou}; train-iou-bg={iou_bg}; train-iou-seagrass={iou_seagrass};')
    update_tensorboard(writer, device, epoch, mloss, miou, iou_bg, iou_seagrass, x, y, pred)

    # Save model checkpoints
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mean_eval_loss': mloss,
    }, Path(checkpoint_dir).joinpath('unet.pt'))

    return model

def validate_one_epoch(model, device, dataloader, epoch, writer):
    model.eval()
    sum_loss = 0.
    sum_iou = np.zeros(num_classes)

    for x, y in tqdm(iter(dataloader), desc=f"eval epoch {epoch}", file=sys.stdout):
        with torch.no_grad():
            y = y.to(device)
            x = x.to(device)
            x = add_veg_indices(x)

            pred = model(x)
            sig = torch.sigmoid(pred[:, 1])
            loss = assymetric_tversky_loss(sig, y, beta=1.5)

        # Compute metrics
        sum_loss += loss.detach().cpu().item()
        sum_iou += iou(y, pred.float()).detach().cpu().numpy()

    mloss = sum_loss / len(dataloader)
    ious = np.around(sum_iou / len(dataloader), 4)
    miou = np.mean(ious)
    iou_bg = sum_iou[0] / len(dataloader)
    iou_seagrass = sum_iou[1] / len(dataloader)

    print(f'eval-loss={mloss}; eval-miou={miou}; eval-iou-bg={iou_bg}; eval-iou-seagrass={iou_seagrass};')
    update_tensorboard(writer, device, epoch, mloss, miou, iou_bg, iou_seagrass, x, y, pred)

    # Save best models for eval set
    return mloss, miou, iou_bg, iou_seagrass


def train_model(model, device, dataloaders, num_epochs, lr, weight_decay, checkpoint_dir, weights_dir,
                restart_training):
    # Pass lr, weight_decay rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    checkpoint_path = checkpoint_dir.joinpath('unet.pt')
    # Restart at checkpoint if it exists
    if Path(checkpoint_path).exists() and restart_training:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(checkpoint_dir.joinpath('runs', current_time + '_' + socket.gethostname()))

    train_writer = SummaryWriter(log_dir=log_dir + '_train')
    eval_writer = SummaryWriter(log_dir=log_dir + '_eval')

    best_val_loss = None
    best_val_iou_seagrass = None
    best_val_miou = None

    for epoch in range(start_epoch, num_epochs):
        model = train_one_epoch(model, device, optimizer, lr_scheduler, dataloaders['train'], epoch, train_writer, checkpoint_dir)

        # Validation loop
        mloss, miou, iou_bg, iou_seagrass = validate_one_epoch(model, device, dataloaders['eval'], epoch, eval_writer)

        # Save best models
        if best_val_loss is None or mloss < best_val_loss:
            best_val_loss = mloss
            torch.save(model.state_dict(), Path(weights_dir).joinpath('unet_best_val_loss.pt'))
        if best_val_iou_seagrass is None or iou_seagrass < best_val_iou_seagrass:
            best_val_iou_seagrass = iou_seagrass
            torch.save(model.state_dict(), Path(weights_dir).joinpath('unet_best_val_seagrass_iou.pt'))
        if best_val_miou is None or miou < best_val_miou:
            best_val_miou = miou
            torch.save(model.state_dict(), Path(weights_dir).joinpath('unet_best_val_miou.pt'))

    train_writer.flush()
    train_writer.close()
    eval_writer.flush()
    eval_writer.close()

    torch.save(model.state_dict(), Path(weights_dir).joinpath("unet_final.pt"))


if __name__ == '__main__':
    script_mode = sys.argv[1]
    DOCKER = bool(os.environ.get('DOCKER', False))
    DISABLE_CUDA = False

    if DOCKER:
        # These locations are those required by Amazon Sagemaker
        hparams_path = Path('/opt/ml/input/config/hyperparameters.json')
        train_data_dir = "/opt/ml/input/data/train"
        eval_data_dir = "/opt/ml/input/data/eval"
        seg_in_dir = Path("/opt/ml/input/data/segmentation")
        checkpoint_dir = Path('/opt/ml/output/checkpoints')
        weights_dir = Path('/opt/ml/output/model')
        seg_out_dir = Path("/opt/ml/output/segmentation")

        # Redirect stderr to file
        sys.stderr = open('/opt/ml/output/failure', 'w')
    else:
        # For running script locally without Docker use these for e.g
        hparams_path = Path('seagrass/train_input/config/hyperparameters.json')
        train_data_dir = "seagrass/train_input/data/train"
        eval_data_dir = "seagrass/train_input/data/eval"
        seg_in_dir = Path("seagrass/train_input/data/segmentation")
        checkpoint_dir = Path('seagrass/train_output/checkpoints')
        weights_dir = Path('seagrass/train_output/model')
        seg_out_dir = Path("seagrass/train_output/segmentation")

    # Load hyper-parameters dictionary
    hparams = json.load(open(hparams_path))
    num_classes = int(hparams["num_classes"])  # "2",
    num_epochs = int(hparams["num_epochs"])  # "200",
    batch_size = int(hparams["batch_size"])  # "8",
    lr = float(hparams["lr"])  # "0.001",
    weight_decay = float(hparams["weight_decay"])  # "0.01",
    restart_training = hparams["restart_training"] == "true"  # "true

    # Make results reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if not DISABLE_CUDA and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Model
    os.environ['TORCH_HOME'] = str(checkpoint_dir.parents[0])
    model = UNet(n_channels=7, n_classes=num_classes, bilinear=True)
    model = model.to(device)
    model = nn.DataParallel(model)

    # Datasets
    num_gpus = torch.cuda.device_count()
    print(num_gpus, "GPU(s) detected")
    if num_gpus > 1:
        batch_size *= num_gpus

    if script_mode == "train":
        # Create dataloaders
        ds_train = SegmentationDataset(train_data_dir, transform=T.train_transforms,
                                       target_transform=T.train_target_transforms)
        ds_val = SegmentationDataset(eval_data_dir, transform=T.test_transforms,
                                     target_transform=T.test_target_transforms)
        dataloader_opts = {
            "batch_size": batch_size,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": os.cpu_count()
        }
        data_loaders = {
            'train': DataLoader(ds_train, shuffle=True, **dataloader_opts),
            'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
        }

        train_model(model, device, data_loaders, num_epochs, lr, weight_decay, checkpoint_dir,
                    weights_dir, restart_training)

    elif script_mode == "eval":
        pass
    # ds_train = SegmentationDataset(train_data_dir, transform=T.train_transforms,
    #                                target_transform=T.train_target_transforms)
    # ds_val = SegmentationDataset(eval_data_dir, transform=T.test_transforms,
    #                              target_transform=T.test_target_transforms)
    #
    # dataloader_opts = {
    #     "batch_size": batch_size,
    #     "pin_memory": True,
    #     "num_workers": os.cpu_count()
    # }
    # data_loaders = {
    #     'train': DataLoader(ds_train, shuffle=False, **dataloader_opts),
    #     'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
    # }
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Trained model
    # model_weights_path = weights_dir.joinpath(hparams['eval_weights'])
    # checkpoint = torch.load(model_weights_path, map_location=device)
    # model.load_state_dict(checkpoint)
    #
    # eval_model(model, device, data_loaders, num_classes, criterion)

    elif script_mode == "pred":
        pass
    # # Trained model
    # model_weights_path = weights_dir.joinpath(hparams['eval_weights'])
    # checkpoint = torch.load(model_weights_path, map_location=device)
    # model.load_state_dict(checkpoint)
    #
    # # Process all .tif images in segmentation in directory
    # for img_path in seg_in_dir.glob("*.tif"):
    #     print("Processing:", img_path)
    #     out_path = str(seg_out_dir.joinpath(img_path.stem + "_seagrass_seg" + img_path.suffix))
    #
    #     predict_tiff(model, device, img_path, out_path, T.test_transforms, crop_size=300, pad=150, batch_size=4)
    #
    #     # Move input file to output directory
    #     rasterio.shutil.copyfiles(str(img_path), str(seg_out_dir.joinpath(img_path.name)))

    else:
        raise RuntimeError("Must specify train|eval|pred as script arg")

    if DOCKER:
        sys.stderr.close()
