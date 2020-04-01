#!/usr/bin/env python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from tqdm import tqdm, trange
from collections import OrderedDict
import numpy as np
from pathlib import Path
import socket
from datetime import datetime

from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.TransformDataset import TransformDataset
from utils.dataset.transforms import transforms as T
from models import deeplabv3
from utils.loss import iou

disable_cuda = False
num_classes = 2
num_epochs = 200
batch_size = 32
ignore_index = 100  # Value of labels that we ignore in loss and other logic (e.g. kelp with unknown species)

prep_datasets = False
torch_dataset_pickle = Path('checkpoints/datasets_010420.pt')
torch_dataset_pickle.parents[0].mkdir(parents=True, exist_ok=True)
ds_paths = [
    "data/datasets/kelp/nw_calvert_2012",
    "data/datasets/kelp/nw_calvert_2015",
    "data/datasets/kelp/choked_pass_2016",
    "data/datasets/kelp/west_beach_2016"
]
dataloader_opts = {
    "batch_size": batch_size,
    "pin_memory": True,
    "drop_last": True,
    "num_workers": os.cpu_count()
}


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def ds_pixel_stats(dataloader):
    pixel_counts = torch.zeros((num_classes)).to(device)

    for _, y in tqdm(iter(dataloader)):
        try:
            un, counts = torch.unique(y.to(device), return_counts=True)
            mask = un != ignore_index
            pixel_counts.index_add_(0, un[mask], counts[mask].float())
        except Exception:
            import pdb;
            pdb.set_trace()

    pixel_ratio = pixel_counts / pixel_counts.sum(dim=0)
    pixel_counts = pixel_counts.detach().cpu().numpy()
    pixel_ratio = np.around(pixel_ratio.detach().cpu().numpy(), 4)

    return pixel_ratio, pixel_counts


def get_indices_of_kelp_images(dataset):
    dl = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    indices = []
    for i, (_, y) in enumerate(tqdm(iter(dl))):
        if torch.any(y > 0):
            indices.append(i)
    return indices


def train_model(model, dataloaders, num_classes, optimizer, criterion, num_epochs, save_path, lr_scheduler=None,
                start_epoch=0):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('checkpoints/runs', current_time + '_' + socket.gethostname())

    writers = {
        'train': SummaryWriter(log_dir=log_dir + '_train'),
        'eval': SummaryWriter(log_dir=log_dir + '_eval')
    }
    info = OrderedDict()

    best_val_loss = None
    best_val_iou_kelp = None
    best_val_miou = None

    pbar_epoch = trange(num_epochs, desc="epoch")
    if start_epoch != 0:
        pbar_epoch.update(start_epoch)
        pbar_epoch.refresh()

    for epoch in pbar_epoch:
        for phase in ['train', 'eval']:
            sum_loss = 0.
            sum_iou = np.zeros(num_classes)

            with tqdm(iter(dataloaders[phase]), desc=phase) as pbar:
                for i, (x, y) in enumerate(pbar):
                    y = y.to(device)
                    x = x.to(device)

                    optimizer.zero_grad()

                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    pred = model(x)['out']
                    loss = criterion(pred.float(), y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Compute metrics
                    sum_loss += loss.detach().cpu().item()
                    sum_iou += iou(y, pred.float()).detach().cpu().numpy()

                    mloss = sum_loss / (i + 1)
                    ious = np.around(sum_iou / (i + 1), 4)
                    miou = np.mean(ious)
                    iou_bg = sum_iou[0] / (i + 1)
                    iou_kelp = sum_iou[1] / (i + 1)

                    info['mean_loss'] = mloss
                    info['mIoUs'] = ious
                    pbar.set_postfix(info)

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
            writers[phase].add_image('Labels/True', label_grid, epoch)

            pred = pred.max(dim=1)[1].unsqueeze(dim=1)
            pred_grid = torchvision.utils.make_grid(pred, nrow=8).cuda()
            pred_grid = alpha_blend(img_grid, pred_grid)
            writers[phase].add_image('Labels/Pred', pred_grid, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save model checkpoints
        if phase == 'train':
            # Model checkpoint after every train phase every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_eval_loss': info['mean_loss'],
            }, Path(save_path).joinpath('checkpoint.pt'))
        else:
            # Save best models for eval set
            if best_val_loss is None or mloss < best_val_loss:
                best_val_loss = mloss
                torch.save(model.state_dict(), Path(save_path).joinpath('deeplabv3_best_val_loss.pt'))
            if best_val_iou_kelp is None or iou_kelp < best_val_iou_kelp:
                best_val_iou_kelp = iou_kelp
                torch.save(model.state_dict(), Path(save_path).joinpath('deeplabv3_best_val_kelp_iou.pt'))
            if best_val_miou is None or miou < best_val_miou:
                best_val_miou = miou
                torch.save(model.state_dict(), Path(save_path).joinpath('deeplabv3_best_val_miou.pt'))

    pbar_epoch.close()
    writers['train'].flush()
    writers['train'].close()
    writers['eval'].flush()
    writers['eval'].close()
    return model


if __name__ == '__main__':
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using device:", device)

    # Make results reproducable
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Datasets
    if prep_datasets:
        ds = torch.utils.data.ConcatDataset([SegmentationDataset(path) for path in ds_paths])

        train_num = int(len(ds) * 0.8)
        val_num = len(ds) - train_num
        ds_train, ds_val = torch.utils.data.random_split(ds, [train_num, val_num])

        # Bind data transforms to the subset datasets
        ds_train = TransformDataset(ds_train, T.train_transforms, T.train_target_transforms)
        ds_val = TransformDataset(ds_val, T.test_transforms, T.test_target_transforms)

        train_indices = get_indices_of_kelp_images(ds_train)
        val_indices = get_indices_of_kelp_images(ds_val)

        ds_train = torch.utils.data.Subset(ds_train, train_indices)
        ds_val = torch.utils.data.Subset(ds_val, val_indices)

        torch.save({
            'train': ds_train,
            'val': ds_val,
        }, torch_dataset_pickle)

    else:
        dataset_saved = torch.load(torch_dataset_pickle)
        ds_train = dataset_saved['train']
        ds_val = dataset_saved['val']

    # Net, opt, loss, dataloaders
    model = deeplabv3.create_model(num_classes)
    model = model.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        batch_size *= num_gpus
        model = nn.DataParallel(model)

    data_loaders = {
        'train': DataLoader(ds_train, shuffle=True, **dataloader_opts),
        'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
    }

    # Train the model
    save_path = Path('checkpoints/deeplabv3/')
    save_path.parents[0].mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path.joinpath('checkpoint.pt')

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    poly_lambda = lambda i: (1 - i / num_epochs) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Restart at checkpoint if it exists
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    model = train_model(model, data_loaders, num_classes, optimizer, criterion, num_epochs, save_path,
                        lr_scheduler, start_epoch=epoch)

    # Save the final model
    save_path = Path(f'checkpoints/deeplabv3/deeplabv3_epoch{num_epochs}.pt')
    save_path.parents[0].mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
