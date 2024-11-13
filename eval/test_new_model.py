from pathlib import Path

import torch
import torchmetrics.classification as fm
import wandb
from einops import rearrange
from tqdm.auto import tqdm

from train.configs.config import load_yml_config
from train.datamodule import DataModule
from train.model import SMPSegmentationModel

# from train.losses import DiceLoss
from train.transforms import get_train_transforms, get_test_transforms, extra_transforms

DEVICE = torch.device("cuda")
BATCH_SIZE = 6
TILE_SIZE = 1024
IGNORE_INDEX = 2
NUM_CLASSES = 2


def main(
    run_id: str = "8a1r0k6h",
    config_file: str = "/home/taylor/PycharmProjects/hakai-ml-train/train/configs/kelp-rgb/pa-unetpp-efficientnetv2-m.yml",
):
    # Setup WandB logging
    run = wandb.init(project="kom-kelp-sp-rgb", id=run_id, resume="must")

    # Download and load model
    artifact = run.use_artifact(f"hakai/kom-kelp-sp-rgb/model-{run_id}:best")
    checkpoint_path = artifact.get_path("model.ckpt").download()
    state_dict = torch.load(checkpoint_path)["state_dict"]

    config = load_yml_config(Path(config_file))
    model = SMPSegmentationModel(**config.segmentation_config.dict())
    model.load_state_dict(state_dict)
    model = model.model.eval().to(DEVICE)
    model.requires_grad_(False)

    data_module = DataModule(
        data_dir="/home/taylor/data/KS-RGB-Jan2024/",
        tile_size=config.tile_size,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        train_transforms=get_train_transforms(
            tile_size=TILE_SIZE,
        ),
        test_transforms=get_test_transforms(
            tile_size=TILE_SIZE,
        ),
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Setup metrics
    # loss_fn = DiceLoss(mode="binary", from_logits=True, smooth=1.0)
    accuracy = fm.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    jaccard_index = fm.JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(
        DEVICE
    )
    recall = fm.Recall(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    precision = fm.Precision(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    f1_score = fm.F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    dice = fm.Dice(num_classes=NUM_CLASSES, multiclass=True).to(DEVICE)

    for phase, dataloader in [("test", test_dataloader)]:
        # Test loop
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=phase)):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)

            # Flatten and eliminate ignore class instances
            y = rearrange(y, "b h w -> (b h w)").long()
            logits = rearrange(logits, "b c h w -> (b h w) c")

            mask = y != IGNORE_INDEX
            logits, y = logits[mask], y[mask]

            if len(y) == 0:
                # print("0 length y!")
                continue

            probs = torch.softmax(logits, dim=1)
            # probs = torch.sigmoid(logits).squeeze(1)

            # Log metrics
            wandb.log(
                {
                    # f"{phase}/loss": loss,
                    f"{phase}/accuracy": accuracy(probs, y),
                    f"{phase}/iou": jaccard_index(probs, y),
                    f"{phase}/recall": recall(probs, y),
                    f"{phase}/precision": precision(probs, y),
                    f"{phase}/f1": f1_score(probs, y),
                    f"{phase}/dice": dice(probs, y),
                    "epoch": 19,
                },
            )

        # Log epoch metrics
        wandb.log(
            {
                f"{phase}/iou_epoch": jaccard_index.compute(),
                f"{phase}/recall_epoch": recall.compute(),
                f"{phase}/accuracy_epoch": accuracy.compute(),
                f"{phase}/precision_epoch": precision.compute(),
                f"{phase}/f1_epoch": f1_score.compute(),
                f"{phase}/dice_epoch": dice.compute(),
                "epoch": 19,
            },
        )

        # Reset metrics
        accuracy.reset()
        jaccard_index.reset()
        recall.reset()
        precision.reset()
        f1_score.reset()
        dice.reset()

    # End run
    run.finish()


if __name__ == "__main__":
    main()
