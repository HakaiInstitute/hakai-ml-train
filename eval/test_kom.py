import torch
import torchmetrics.classification as fm
import wandb
from einops import rearrange
from kelp_o_matic.models import KelpRGBPresenceSegmentationModel
from tqdm.auto import tqdm

from train.datamodule import DataModule
# from train.losses import DiceLoss
from train.transforms import get_train_transforms, get_test_transforms

DEVICE = torch.device("cuda")
BATCH_SIZE = 6
TILE_SIZE = 1024
IGNORE_INDEX = 2


def main():
    model = KelpRGBPresenceSegmentationModel()

    data_module = DataModule(
        data_dir="/home/taylor/data/KP-RGB-Jan2024/",
        tile_size=TILE_SIZE,
        num_classes=1,
        batch_size=BATCH_SIZE,
        train_transforms=get_train_transforms(tile_size=TILE_SIZE),
        test_transforms=get_test_transforms(tile_size=TILE_SIZE),
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Setup WandB logging
    run = wandb.init(
        project="kom-kelp-pa-rgb",
        name="KOM-LRASPP_MobileNetV3 ",
        notes="KOM production RGB model. LRASPP_MobileNetV3",
        tags=["kp-rgb", "Jan2024"],
    )

    # Setup metrics
    # loss_fn = DiceLoss(mode="binary", from_logits=True, smooth=1.0)
    accuracy = fm.BinaryAccuracy().to(DEVICE)
    jaccard_index = fm.BinaryJaccardIndex().to(DEVICE)
    recall = fm.BinaryRecall().to(DEVICE)
    precision = fm.BinaryPrecision().to(DEVICE)
    f1_score = fm.BinaryF1Score().to(DEVICE)
    dice = fm.Dice().to(DEVICE)

    for phase, dataloader in [("test", test_dataloader), ("val", val_dataloader), ("train", train_dataloader)]:
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
                print("0 length y!")
                return 0

            # loss = loss_fn(logits, y.long().unsqueeze(1))
            probs = torch.softmax(logits, dim=1)[:, 1]

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
                    "epoch": 19
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
                "epoch": 19
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
