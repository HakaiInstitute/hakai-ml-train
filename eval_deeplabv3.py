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

from utils.dataset import SegmentationDataset, TransformDataset, transforms as T
from models import deeplabv3, half_precision
from utils.loss import iou

torch_dataset_pickle = 'checkpoints/datasets_010420.pt'
model_save_path = 'checkpoints/deeplabv3/deeplabv3_310320.pt'

disable_cuda = False
num_classes = 2
num_epochs = 200
batch_size = 16

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


def eval_model(model, dataloaders, num_classes, criterion):
    info = OrderedDict()
    model.eval()

    for phase in ['train', 'eval']:
        sum_loss = 0.
        sum_iou = np.zeros(num_classes)

        for i, (x, y) in enumerate(tqdm(iter(dataloaders[phase]), desc=phase)):
            y = y.to(device)
            x = x.to(device)

            pred = model(x)['out']
            loss = criterion(pred.float(), y)

            # Compute metrics
            sum_loss += loss.detach().cpu().item()
            info['mean_loss'] = sum_loss / (i + 1)
            sum_iou += iou(y, pred.float()).detach().cpu().numpy()
            info['mIoUs'] = np.around(sum_iou / (i + 1), 4)

        print(phase.capitalize())
        print('Loss', info['mean_loss'])
        print('IoU/Mean', np.mean(info['mIoUs']))
        print('IoU/BG', sum_iou[0] / (i + 1))
        print('IoU/Kelp', sum_iou[1] / (i + 1))


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
    dataset_saved = torch.load(torch_dataset_pickle)
    data_loaders = {
        'train': DataLoader(dataset_saved['train'], shuffle=True, **dataloader_opts),
        'eval': DataLoader(dataset_saved['val'], shuffle=False, **dataloader_opts),
    }

    # Net, loss
    model = deeplabv3.create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Eval the model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint)

    eval_model(model, data_loaders, num_classes, criterion)
