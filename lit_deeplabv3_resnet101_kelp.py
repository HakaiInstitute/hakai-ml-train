import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.loggers import TestTubeLogger

from kelp_data_module import KelpDataModule
from models.deeplabv3_resnet101 import DeepLabv3ResNet101
from utils.checkpoint import get_checkpoint


def cli_main(argv=None):
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(name='train', help="Train the model.")
    parser_train.add_argument('data_dir', type=str,
                              help="The path to a data directory with subdirectories 'train' and "
                                   "'eval', each with 'x' and 'y' subdirectories containing image "
                                   "crops and labels, respectively.")
    parser_train.add_argument('checkpoint_dir', type=str, help="The path to save training outputs")
    parser_train.add_argument('--initial_weights_ckpt', type=str,
                              help="Path to checkpoint file to load as initial model weights")
    parser_train.add_argument('--initial_weights', type=str,
                              help="Path to pytorch weights to load as initial model weights")
    parser_train.add_argument('--pa_weights', type=str,
                              help="Presence/Absence model weights to use as initial model weights")
    parser_train.add_argument('--name', type=str, default="",
                              help="Identifier used when creating files and directories for this "
                                   "training run.")

    parser_train = KelpDataModule.add_argparse_args(parser_train)
    parser_train = DeepLabv3ResNet101.add_argparse_args(parser_train)
    # parser_train = DeepLabv3FineTuningCallback.add_argparse_args(parser_train)
    parser_train = pl.Trainer.add_argparse_args(parser_train)
    parser_train.set_defaults(func=train)

    parser_pred = subparsers.add_parser(name='pred', help='Predict kelp presence in an image.')
    parser_pred.add_argument('seg_in', type=str,
                             help="Path to a *.tif image to do segmentation on in pred mode.")
    parser_pred.add_argument('seg_out', type=str,
                             help="Path to desired output *.tif created by the model in pred mode.")
    parser_pred.add_argument('weights', type=str,
                             help="Path to a model weights file (*.pt). "
                                  "Required for eval and pred mode.")
    parser_pred.add_argument('--batch_size', type=int, default=2,
                             help="The batch size per GPU (default 2).")
    parser_pred.add_argument('--crop_pad', type=int, default=128,
                             help="The amount of padding added for classification context to each "
                                  "image crop. The output classification on this crop area is not "
                                  "output by the model but will influence the classification of "
                                  "the area in the (crop_size x crop_size) window "
                                  "(defaults to 128).")
    parser_pred.add_argument('--crop_size', type=int, default=256,
                             help="The crop size in pixels for processing the image. Defines the "
                                  "length and width of the individual sections the input .tif "
                                  "image is cropped to for processing (defaults 256).")
    parser_pred = DeepLabv3ResNet101.add_argparse_args(parser_pred)
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
    print("Loading model:", args.weights)
    if Path(args.weights).suffix == "ckpt":
        model = DeepLabv3ResNet101.load_from_checkpoint(args.weights, batch_size=args.batch_size,
                                                        crop_size=args.crop_size,
                                                        padding=args.crop_pad)
    else:  # Assumes .pt
        model = DeepLabv3ResNet101(args)
        model.load_state_dict(torch.load(args.weights), strict=False)

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
    kelp_presence_data = KelpDataModule(args.data_dir, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    os.environ['TORCH_HOME'] = str(Path(args.checkpoint_dir).parent)
    if checkpoint := get_checkpoint(args.checkpoint_dir, args.name):
        print("Loading checkpoint:", checkpoint)
        model = DeepLabv3ResNet101.load_from_checkpoint(checkpoint)
    elif args.initial_weights_ckpt:
        print("Loading initial weights ckpt:", args.initial_weights_ckpt)
        model = DeepLabv3ResNet101.load_from_checkpoint(args.initial_weights_ckpt)
    elif args.pa_weights:
        print("Loading presence/absence ckpt:", args.pa_weights)
        model = DeepLabv3ResNet101.from_presence_absence_weights(args.pa_weights, args)
    else:
        model = DeepLabv3ResNet101(args)

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

    # Validation and Test stats
    trainer.validate(datamodule=kelp_presence_data, ckpt_path=checkpoint_cb.best_model_path)
    trainer.test(datamodule=kelp_presence_data, ckpt_path=checkpoint_cb.best_model_path)

    # Save final weights only
    model.load_from_checkpoint(checkpoint_cb.best_model_path)
    torch.save(model.state_dict(), Path(checkpoint_cb.best_model_path).with_suffix('.pt'))


if __name__ == '__main__':
    if os.getenv('DEBUG', False):
        # cli_main([
        #     'pred',
        #     'scripts/presence/train_input/Triquet_kelp_U0653.tif',
        #     'scripts/presence/train_output/Triquet_kelp_U0653_kelp.tif',
        #     'species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt',
        #     # '--batch_size=2',
        #     # '--crop_size=256',
        #     # '--crop_pad=128'
        # ])
        cli_main([
            'train',
            'scripts/presence/train_input/data',
            'scripts/presence/train_output/checkpoints',
            '--name=TEST', '--num_classes=2', '--lr=0.001', '--backbone_lr=0.00001',
            '--weight_decay=0.001', '--gradient_clip_val=0.5', '--auto_select_gpus', '--gpus=-1',
            '--benchmark', '--max_epochs=100', '--batch_size=2', "--unfreeze_backbone_epoch=100",
            '--log_every_n_steps=5', '--overfit_batches=1', '--no_train_backbone_bn'
        ])
        # cli_main([
        #     'train',
        #     'scripts/species/train_input/data',
        #     'scripts/species/train_output/checkpoints',
        #     '--name=TEST', '--num_classes=3', '--lr=0.001', '--backbone_lr=0.00001',
        #     '--weight_decay=0.001', '--gradient_clip_val=0.5', '--auto_select_gpus', '--gpus=-1',
        #     '--benchmark', '--max_epochs=10', '--batch_size=2', "--unfreeze_backbone_epoch=100",
        #     '--log_every_n_steps=5', '--overfit_batches=1', '--no_train_backbone_bn',
        #     '--pa_weights=species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt'
        # ])
    else:
        cli_main()
