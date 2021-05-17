import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.loggers import TestTubeLogger

from kelp_presence_data_module import KelpPresenceDataModule
from models.deeplabv3 import DeepLabv3
from utils.checkpoint import get_checkpoint


def cli_main(argv=None):
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(name='train', help="Train the model.")
    parser_train.add_argument('data_dir', type=str,
                              help="The path to a data directory with subdirectories 'train' and 'eval', each with 'x' "
                                   "and 'y' subdirectories containing image crops and labels, respectively.")
    parser_train.add_argument('checkpoint_dir', type=str, help="The path to save training outputs")
    parser_train.add_argument('--initial_weights_ckpt', type=str, help="Path to checkpoint file to load as initial model weights")
    parser_train.add_argument('--initial_weights', type=str, help="Path to pytorch weights to load as initial model weights")
    parser_train.add_argument('--name', type=str, default="",
                              help="Identifier used when creating files and directories for this training run.")

    parser_train = KelpPresenceDataModule.add_argparse_args(parser_train)
    parser_train = DeepLabv3.add_argparse_args(parser_train)
    # parser_train = DeepLabv3FineTuningCallback.add_argparse_args(parser_train)
    parser_train = pl.Trainer.add_argparse_args(parser_train)
    parser_train.set_defaults(func=train)

    parser_pred = subparsers.add_parser(name='pred', help='Predict kelp presence in an image.')
    parser_pred.add_argument('seg_in', type=str,
                             help="Path to a *.tif image to do segmentation on in pred mode.")
    parser_pred.add_argument('seg_out', type=str,
                             help="Path to desired output *.tif created by the model in pred mode.")
    parser_pred.add_argument('weights', type=str,
                             help="Path to a model weights file (*.pt). Required for eval and pred mode.")
    parser_pred.add_argument('--batch_size', type=int, default=2,
                             help="The batch size per GPU (default 2).")
    parser_pred.add_argument('--crop_pad', type=int, default=128,
                             help="The amount of padding added for classification context to each image crop. The "
                                  "output classification on this crop area is not output by the model but will "
                                  "influence the classification of the area in the (crop_size x crop_size) window "
                                  "(defaults to 128).")
    parser_pred.add_argument('--crop_size', type=int, default=512,
                             help="The crop size in pixels for processing the image. Defines the length and width of "
                                  "the individual sections the input .tif image is cropped to for processing "
                                  "(default 512).")
    parser_pred.set_defaults(func=pred)

    args = parser.parse_args(argv)
    args.func(args)


def pred(args):
    seg_in, seg_out = Path(args.seg_in), Path(args.seg_out)
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logger.warning("Could not find GPU device")

    # ------------
    # model
    # ------------
    model = DeepLabv3.load_from_checkpoint(args.weights, batch_size=args.batch_size,
                                           crop_size=args.crop_size,
                                           padding=args.crop_pad)
    model.freeze()
    model = model.to(device)

    # ------------
    # inference
    # ------------
    model.predict_geotiff(str(seg_in), str(seg_out))


def train(args):
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
    elif args.initial_weights_ckpt:
        print("Loading initial weights ckpt:", args.initial_weights_ckpt)
        model = DeepLabv3.load_from_checkpoint(args.initial_weights_ckpt)
    else:
        model = DeepLabv3(args)

    if args.initial_weights:
        print("Loading initial weights:", args.initial_weights)
        model.load_state_dict(torch.load(args.initial_weights))

    # ------------
    # callbacks
    # ------------
    logger = TestTubeLogger(args.checkpoint_dir, name=args.name)
    lr_monitor_cb = pl.callbacks.LearningRateMonitor()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
            verbose=True,
            monitor='val_miou',
            mode='max',
            filename='best-{val_miou:.4f}-{epoch}-{step}',
            save_top_k=1,
            save_last=True,
        )

    callbacks = [
        lr_monitor_cb,
        checkpoint_cb,
        # DeepLabv3FineTuningCallback(args.unfreeze_backbone_epoch, args.train_backbone_bn)
    ]
    if isinstance(args.gpus, int):
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,
                                            resume_from_checkpoint=checkpoint)
    # Tune params
    # trainer.tune(model, datamodule=kelp_presence_data)

    # Training
    trainer.fit(model, datamodule=kelp_presence_data)

    # Testing
    model.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(datamodule=kelp_presence_data)

    # Save final weights only
    torch.save(model.state_dict(), Path(checkpoint_cb.best_model_path).with_suffix('.pt'))


if __name__ == '__main__':
    debug = os.getenv('DEBUG', False)
    if debug:
        # cli_main([
        #     'pred',
        #     'kelp/presence/train_input/Triquet_kelp_U0653.tif',
        #     'kelp/presence/train_output/checkpoints/NewKelpDataset/version_2/Triquet_kelp_U0653_kelp.tif',
        #     'kelp/presence/train_output/checkpoints/NewKelpDataset/version_2/checkpoints/best-val-miou-epoch=89-step=10259.ckpt',
        #     # '--batch_size=2',
        #     # '--crop_size=64',
        #     # '--crop_pad=0'
        # ])
        cli_main([
            'train',
            'kelp/presence/train_input/data',
            'kelp/presence/train_output/checkpoints',
            '--name=TEST', '--num_classes=2', '--lr=0.001', '--backbone_lr=0.00001',
            '--weight_decay=0.001', '--gradient_clip_val=0.5', '--auto_select_gpus', '--gpus=-1',
            '--benchmark', '--max_epochs=10', '--batch_size=2', "--unfreeze_backbone_epoch=100",
            '--log_every_n_steps=5', '--overfit_batches=1', '--no_train_backbone_bn',
            '--initial_weights=kelp/presence/train_output/checkpoints/2021-05-17-2218_Hist_Equalization/version_1/checkpoints/best-val_miou=0.8154-epoch=97-step=11171.pt'
        ])
    else:
        cli_main()
