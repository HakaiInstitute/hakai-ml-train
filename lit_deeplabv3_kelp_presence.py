import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.deeplabv3 import DeepLabv3
from utils.checkpoint import get_checkpoint
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('train_data_dir', type=str)
    parser.add_argument('val_data_dir', type=str)
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('--name', type=str, default="")

    parser = DeepLabv3.add_argparse_args(parser)
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
    pl.seed_everything(0)
    os.environ['TORCH_HOME'] = str(Path(args.checkpoint_dir).parent)

    if checkpoint := get_checkpoint(args.checkpoint_dir, args.name):
        print("Loading checkpoint:", checkpoint)
        model = DeepLabv3.load_from_checkpoint(checkpoint)
    else:
        model = DeepLabv3(args)

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger(args.checkpoint_dir, name=args.name)

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            verbose=True,
            monitor='val_miou',
            mode='max',
            prefix='best-val-miou',
            save_top_k=1,
            save_last=True,
        ),
    ]
    if args.gpus >= 1:
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,
                                            resume_from_checkpoint=checkpoint)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    # TODO
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
