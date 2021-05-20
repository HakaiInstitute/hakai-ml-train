import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from models.unet import UNet
from pytorch_lightning.loggers import TestTubeLogger

from kelp_data_module import KelpDataModule
from utils.checkpoint import get_checkpoint


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'pred'])

    args = parser.parse_args()
    if args.mode == 'train':
        train(parser)
    elif args.mode == 'pred':
        pred(parser)


def pred(parent_parser):
    print("TODO: prediction")


def train(parent_parser):
    # ------------
    # args
    # ------------
    parser = ArgumentParser(parents=parent_parser)

    parser.add_argument('data_dir', type=str)
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('--name', type=str, default="")

    parser = UNet.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(0)

    # ------------
    # data
    # ------------
    kelp_presence_data = KelpDataModule(args.data_dir, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    os.environ['TORCH_HOME'] = str(Path(args.checkpoint_dir).parent)
    if checkpoint := get_checkpoint(args.checkpoint_dir, args.name):
        print("Loading checkpoint:", checkpoint)
        model = UNet.load_from_checkpoint(checkpoint)
    else:
        model = UNet(args)

    # ------------
    # training
    # ------------
    logger = TestTubeLogger(args.checkpoint_dir, name=args.name)

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
    if isinstance(args.gpus, int):
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,
                                            resume_from_checkpoint=checkpoint)
    # Tune params
    # trainer.tune(model, datamodule=kelp_presence_data)

    # Training
    trainer.fit(model, datamodule=kelp_presence_data)

    # Testing
    trainer.test(datamodule=kelp_presence_data)


if __name__ == '__main__':
    cli_main()
