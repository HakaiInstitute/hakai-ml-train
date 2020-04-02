import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import numpy as np
from pathlib import Path
import socket
from datetime import datetime
import json
from tqdm import tqdm


from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as T
from models import deeplabv3
from utils.loss import iou


def alpha_blend(bg, fg, alpha=0.5):
    return fg * alpha + bg * (1 - alpha)


def train_model(model, dataloaders, num_classes, optimizer, criterion, num_epochs, save_path,
                lr_scheduler=None,
                start_epoch=0):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_path.joinpath('runs', current_time + '_' + socket.gethostname()))

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

            for i, (x, y) in enumerate(tqdm(iter(dataloaders[phase]), desc=f"Epoch {epoch} - {phase}")):
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

            # Save model checkpoints
            if phase == 'train':
                # Model checkpoint after every train phase every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mean_eval_loss': mloss,
                }, Path(save_path).joinpath('deeplabv3.pt'))
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

        if lr_scheduler is not None:
            lr_scheduler.step()

    writers['train'].flush()
    writers['train'].close()
    writers['eval'].flush()
    writers['eval'].close()
    return model


if __name__ == '__main__':
    checkpoint_dir = Path('/opt/ml/checkpoints')
    hparams_path = Path('/opt/ml/config/hyperparameters.json')
    train_data_dir = "/opt/ml/train"
    eval_data_dir = "/opt/ml/eval"
    # checkpoint_dir = Path('../data/checkpoints')
    # hparams_path = Path('../data/config/hyperparameters.json')
    # train_data_dir = "../data/train"
    # eval_data_dir = "../data/eval"

    # Load hyperparameters dictionary
    hparams = json.load(open(hparams_path))

    # Make results reproducible
    torch.manual_seed(hparams['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(hparams['random_seed'])

    if not hparams['disable_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Model
    os.environ['TORCH_HOME'] = str(checkpoint_dir.parents[0])
    model = deeplabv3.create_model(hparams['num_classes'])
    model = model.to(device)

    # Datasets
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        hparams['batch_size'] *= num_gpus
        model = nn.DataParallel(model)

    ds_train = SegmentationDataset(train_data_dir, transform=T.train_transforms, target_transform=T.train_target_transforms)
    ds_val = SegmentationDataset(eval_data_dir, transform=T.test_transforms, target_transform=T.test_target_transforms)

    dataloader_opts = {
        "batch_size": hparams['batch_size'],
        "pin_memory": hparams['dataloader_pin_memory'],
        "drop_last": hparams['dataloader_drop_last'],
        "num_workers": (os.cpu_count() if hparams['dataloader_num_workers'] < 0 else hparams['dataloader_num_workers'])
    }
    data_loaders = {
        'train': DataLoader(ds_train, shuffle=True, **dataloader_opts),
        'eval': DataLoader(ds_val, shuffle=False, **dataloader_opts),
    }

    # Optimizer, Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'], momentum=hparams['momentum'],
                                weight_decay=hparams['weight_decay'])
    poly_lambda = lambda i: (1 - i / hparams['num_epochs']) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lambda)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    checkpoint_path = checkpoint_dir.joinpath('deeplabv3.pt')

    # Restart at checkpoint if it exists
    if Path(checkpoint_path).exists() and hparams['restart_training']:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
    else:
        cur_epoch = 0

    model = train_model(model, data_loaders, hparams['num_classes'], optimizer, criterion, hparams['num_epochs'],
                        checkpoint_dir,
                        lr_scheduler, start_epoch=cur_epoch)

    # Save the final model
    final_save_path = checkpoint_dir.joinpath(f"deeplabv3_epoch{hparams['num_epochs']}.pt")
    torch.save(model.state_dict(), final_save_path)
