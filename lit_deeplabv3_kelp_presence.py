import math
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.deeplabv3 import DeepLabv3
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t


def cli_main():
    pl.seed_everything(0)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('train_data_dir', type=str)
    parser.add_argument('val_data_dir', type=str)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    ds_train = SegmentationDataset(args.train_data_dir, transform=t.train_transforms,
                                   target_transform=t.train_target_transforms)
    ds_val = SegmentationDataset(args.val_data_dir, transform=t.test_transforms,
                                 target_transform=t.test_target_transforms)

    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args.batch_size, pin_memory=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    model = DeepLabv3(num_classes=2,
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      unfreeze_backbone_epoch=100,
                      aux_loss_factor=0.3,
                      max_epochs=args.max_epochs,
                      steps_per_epoch=math.floor(len(train_loader) / max(args.gpus, 1)),
                      )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=TensorBoardLogger("./lightning_logs", name=args.name),
                                            callbacks=[
                                                pl.callbacks.LearningRateMonitor(),
                                                pl.callbacks.GPUStatsMonitor(),
                                                pl.callbacks.ModelCheckpoint(
                                                    verbose=True,
                                                    monitor='val_miou',
                                                    mode='max',
                                                    prefix='best-val-miou',
                                                    save_top_k=1,
                                                    save_last=True,
                                                )
                                            ])
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    # TODO
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
