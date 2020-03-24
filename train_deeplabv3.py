#!/usr/bin/env python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import numpy as np
from pathlib import Path

from utils.dataset import SegmentationDataset, TransformDataset, transforms as T
from utils.metrics import ConfusionMatrix
from models import deeplabv3, half_precision


disable_cuda = False
num_classes = 2
num_epochs = 200
batch_size = 4
ignore_index = 100  # Value of labels that we ignore in loss and other logic (e.g. kelp with unknown species)

prep_datasets = True
torch_dataset_pickle = Path('checkpoints/datasets.pt')
torch_dataset_pickle.parents[0].mkdir(parents=True, exist_ok=True)
ds_paths = [
    "data/datasets/RPAS/Calvert_2012",
    "data/datasets/RPAS/Calvert_2015",
    "data/datasets/RPAS/Calvert_ChokedNorthBeach_2016",
    "data/datasets/RPAS/Calvert_WestBeach_2016"
]
dataloader_opts = {
    "batch_size": batch_size,
    "pin_memory": True,
    "drop_last": True,
    "num_workers": os.cpu_count()
}


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


def train_model(model, dataloaders, num_classes, optimizer, criterion, num_epochs, save_path, start_epoch=0):
    writer = SummaryWriter()
    info = OrderedDict()

    best_loss = None

    for epoch in trange(start_epoch, num_epochs, desc="epoch"):
        sum_loss = 0.
        sum_iou = np.zeros(num_classes)
        cm = ConfusionMatrix(num_classes).to(device)

        for phase in ['train', 'eval']:
            with tqdm(iter(dataloaders[phase]), desc=phase) as pbar:
                for i, (x, y) in enumerate(pbar):
                    global_step = (epoch + 1) * (i + 1)

                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()

                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    pred = model(x)['out']
                    loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Compute metrics
                    sum_loss += loss.detach().cpu().item()
                    info['mean_loss'] = sum_loss / (i + 1)

                    mask = y != ignore_index
                    cm.update(y[mask], pred.max(dim=1)[1][mask])
                    info['IoUs'] = np.around(np.nan_to_num(cm.get_iou().detach().cpu().numpy()), 4)

                    pbar.set_postfix(info)

                    #                     if global_step == 1:
                    #                         writer.add_graph(model, x)

                    writer.add_scalar(f'Loss/{phase}', info['mean_loss'], global_step)
                    writer.add_scalar(f'Mean IoU/{phase}', np.mean(info['IoUs']), global_step)
                    writer.add_histogram(f'IoUs/{phase}', info['IoUs'], global_step, bins=num_classes)

                    if global_step % 500 == 0:
                        # Show images
                        grid = torchvision.utils.make_grid(x, nrow=x.shape[0])
                        grid = T.inv_normalize(grid)
                        writer.add_image(f'{phase}/images', grid, global_step)

                        # Show labels and predictions
                        y = y.unsqueeze(dim=1)
                        grid = torchvision.utils.make_grid(y, nrow=y.shape[0]).type(torch.FloatTensor)
                        writer.add_image(f'{phase}/labels', grid, global_step)

                        # Show predictions
                        pred = pred.max(dim=1)[1].unsqueeze(dim=1)
                        grid = torchvision.utils.make_grid(pred, nrow=pred.shape[0]).type(torch.FloatTensor)
                        writer.add_image(f'{phase}/preds', grid, global_step)

        # Model checkpointing after eval stage
        if best_loss is None or info['mean_loss'] < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_eval_loss': info['mean_loss'],
            }, save_path)

    writer.close()
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
    # model = half_precision(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

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
    save_path = Path('checkpoints/deeplabv3/checkpoint.pt')
    save_path.parents[0].mkdir(parents=True, exist_ok=True)

    # Restart at checkpoint if it exists
    if Path(save_path).exists():
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    model = train_model(model, data_loaders, num_classes, optimizer, criterion, num_epochs, save_path, start_epoch=epoch)

    # Save the final model
    save_path = Path(f'checkpoints/deeplabv3/deeplabv3_epoch{num_epochs}.pt')
    save_path.parents[0].mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
