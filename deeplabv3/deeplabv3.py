#!/usr/bin/env python
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio.shutil
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import deeplabv3
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as T
from utils.eval import predict_tiff, eval_model
from utils.loss import iou


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def train_model(model, device, dataloaders, num_classes, optimizer, criterion, num_epochs, checkpoint_dir, output_dir,
                lr_scheduler=None, start_epoch=0):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(checkpoint_dir.joinpath('runs', current_time + '_' + socket.gethostname()))

    writers = {
        'train': SummaryWriter(log_dir=log_dir + '_train'),
        'eval': SummaryWriter(log_dir=log_dir + '_eval')
    }

    best_val_loss = None
    best_val_iou_kelp = None
    best_val_miou = None

    for epoch in range(start_epoch, num_epochs):
        for phase in ['train', 'eval']:
            sum_loss = 0.
            sum_iou = np.zeros(num_classes)

            for x, y in tqdm(iter(dataloaders[phase]), desc=f"{phase} epoch {epoch}", file=sys.stdout):
                y = y.to(device)
                x = x.to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                pred = F.softmax(model(x)['out'], dim=1)
                loss = criterion(pred.float(), y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Compute metrics
                sum_loss += loss.detach().cpu().item()
                sum_iou += iou(y, pred.float()).detach().cpu().numpy()

            mloss = sum_loss / len(dataloaders[phase])
            ious = np.around(sum_iou / len(dataloaders[phase]), 4)
            miou = np.mean(ious)
            iou_bg = sum_iou[0] / len(dataloaders[phase])
            iou_kelp = sum_iou[1] / len(dataloaders[phase])

            print(f'{phase}-loss={mloss}; {phase}-miou={miou}; {phase}-iou-bg={iou_bg}; {phase}-iou-kelp={iou_kelp};')

            writers[phase].add_scalar('Loss', mloss, epoch)
            writers[phase].add_scalar('IoU/Mean', miou, epoch)
            writers[phase].add_scalar('IoU/BG', iou_bg, epoch)
            writers[phase].add_scalar('IoU/Kelp', iou_kelp, epoch)

            # Show images
            img_grid = torchvision.utils.make_grid(x, nrow=8)
            img_grid = T.inv_normalize(img_grid)

            y = y.unsqueeze(dim=1)
            label_grid = torchvision.utils.make_grid(y, nrow=8).cuda()
            label_grid = alpha_blend(img_grid, label_grid)
            writers[phase].add_image('Labels/True', label_grid.detach().cpu(), epoch)

            pred = pred.max(dim=1)[1].unsqueeze(dim=1)
            pred_grid = torchvision.utils.make_grid(pred, nrow=8).cuda()
            pred_grid = alpha_blend(img_grid, pred_grid)
            writers[phase].add_image('Labels/Pred', pred_grid.detach().cpu(), epoch)

            # Save model checkpoints
            if phase == 'train':
                # Model checkpoint after every train phase every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mean_eval_loss': mloss,
                }, Path(checkpoint_dir).joinpath('deeplabv3.pt'))
            else:
                # Save best models for eval set
                if best_val_loss is None or mloss < best_val_loss:
                    best_val_loss = mloss
                    torch.save(model.state_dict(), Path(output_dir).joinpath('deeplabv3_best_val_loss.pt'))
                if best_val_iou_kelp is None or iou_kelp < best_val_iou_kelp:
                    best_val_iou_kelp = iou_kelp
                    torch.save(model.state_dict(), Path(output_dir).joinpath('deeplabv3_best_val_kelp_iou.pt'))
                if best_val_miou is None or miou < best_val_miou:
                    best_val_miou = miou
                    torch.save(model.state_dict(), Path(output_dir).joinpath('deeplabv3_best_val_miou.pt'))

        if lr_scheduler is not None:
            lr_scheduler.step()

    writers['train'].flush()
    writers['train'].close()
    writers['eval'].flush()
    writers['eval'].close()

    torch.save(model.state_dict(), Path(output_dir).joinpath("deeplabv3_final.pt"))


if __name__ == '__main__':
    script_mode = sys.argv[1]
    DOCKER = bool(os.environ.get('DOCKER', False))
    DISABLE_CUDA = False

    if DOCKER:
        # These locations are those required by Amazon Sagemaker
        checkpoint_dir = Path('/opt/ml/checkpoints')
        weights_dir = Path('/opt/ml/model')
        hparams_path = Path('/opt/ml/input/config/hyperparameters.json')
        train_data_dir = "/opt/ml/input/data/train"
        eval_data_dir = "/opt/ml/input/data/eval"
        seg_in_dir = Path("/opt/ml/input/data/segmentation")
        seg_out_dir = Path("/opt/ml/segmentation")

        # Redirect stderr to file
        sys.stderr = open('/opt/ml/output/failure', 'w')
    else:
        # For running script locally without Docker use these for e.g
        checkpoint_dir = Path('kelp/train_output/checkpoints')
        weights_dir = Path('kelp/train_output/model_weights')
        hparams_path = Path('kelp/train_input/config/hyperparameters.json')
        train_data_dir = "kelp/train_input/data/train"
        eval_data_dir = "kelp/train_input/data/eval"
        seg_in_dir = Path("kelp/train_input/data/segmentation")
        seg_out_dir = Path("kelp/train_output/segmentation")

    # Load hyper-parameters dictionary
    hparams = json.load(open(hparams_path))
    num_classes = int(hparams["num_classes"])  # "2",
    num_epochs = int(hparams["num_epochs"])  # "200",
    batch_size = int(hparams["batch_size"])  # "8",
    lr = float(hparams["lr"])  # "0.001",
    momentum = float(hparams["momentum"])  # "0.9",
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
    model = deeplabv3.create_model(num_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    # Datasets
    num_gpus = torch.cuda.device_count()
    print(num_gpus, "GPU(s) detected")
    if num_gpus > 1:
        batch_size *= num_gpus

    if script_mode == "train":
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

        # Optimizer, Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        checkpoint_path = checkpoint_dir.joinpath('deeplabv3.pt')

        # Restart at checkpoint if it exists
        if Path(checkpoint_path).exists() and restart_training:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            cur_epoch = checkpoint['epoch']
        else:
            cur_epoch = 0

        train_model(model, device, data_loaders, num_classes, optimizer, criterion, num_epochs, checkpoint_dir,
                    weights_dir, start_epoch=cur_epoch)

    elif script_mode == "eval":
        ds_train = SegmentationDataset(train_data_dir, transform=T.test_transforms,
                                       target_transform=T.test_target_transforms)
        ds_val = SegmentationDataset(eval_data_dir, transform=T.test_transforms,
                                     target_transform=T.test_target_transforms)

        dataloader_opts = {
            "batch_size": batch_size,
            "pin_memory": True,
            "num_workers": os.cpu_count()
        }
        data_loaders = {
            'train': DataLoader(ds_train, shuffle=False, **dataloader_opts),
            'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
        }

        criterion = nn.CrossEntropyLoss()

        # Trained model
        model_weights_path = weights_dir.joinpath(hparams['eval_weights'])
        checkpoint = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(checkpoint)

        eval_model(model, device, data_loaders, num_classes, criterion)

    elif script_mode == "pred":
        # Trained model
        model_weights_path = weights_dir.joinpath(hparams['eval_weights'])
        checkpoint = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Process all .tif images in segmentation in directory
        for img_path in seg_in_dir.glob("*.tif"):
            print("Processing:", img_path)
            out_path = str(seg_out_dir.joinpath(img_path.stem + "_kelp_seg" + img_path.suffix))

            predict_tiff(model, device, img_path, out_path, T.test_transforms, crop_size=300, pad=150, batch_size=4)

            # Move input file to output directory
            rasterio.shutil.copyfiles(str(img_path), str(seg_out_dir.joinpath(img_path.name)))

    else:
        raise RuntimeError("Must specify train|eval|pred as script arg")

    if DOCKER:
        sys.stderr.close()
