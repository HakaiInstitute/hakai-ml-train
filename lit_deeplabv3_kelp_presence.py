import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from kelp_presence_data_module import KelpPresenceDataModule
from models.deeplabv3 import DeepLabv3
from utils.checkpoint import get_checkpoint


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('data_dir', type=str)
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('--name', type=str, default="")

    parser = DeepLabv3.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(0)

    # ------------
    # data
    # ------------
    kelp_presence_data = KelpPresenceDataModule(args.data_dir, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
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
