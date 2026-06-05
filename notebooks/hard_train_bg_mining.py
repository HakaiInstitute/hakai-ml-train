import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from pathlib import Path

    import polars as pl
    from hakai_ml_train.models.smp import SMPBinarySegmentationModel
    from hakai_ml_train.data import NpzSegmentationDataset
    import albumentations as A
    from torch.utils.data import DataLoader
    from tqdm.notebook import tqdm
    import torch
    import torch.nn.functional as F
    from pathlib import Path


@app.cell
def _():
    FULL_DATA_DIR = "/home/taylor/data/PlanetScope/pre-chipped-8b/1024_512_20260515_ss2023_full/train"
    return (FULL_DATA_DIR,)


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SMPBinarySegmentationModel.load_from_checkpoint(
        "./kelp-ps8b/u6nrh89o/checkpoints/kelp_ps8b_segformer_b3_epoch-61_val-iou-0.7434.ckpt"
    ).to(device)

    model.eval()
    device
    return device, model


@app.cell
def _(FULL_DATA_DIR):
    test_trans = A.Compose(
        [
            A.PadIfNeeded(
                border_mode=0,
                fill_mask=0.0,
                fill=0.0,
                min_height=1024,
                min_width=1024,
                p=1,
                position="center",
            ),
            A.Normalize(
                max_pixel_value=4000.0,
                mean=[
                    0.43,
                    0.42875,
                    0.47825,
                    0.522,
                    0.5685,
                    0.5725,
                    0.65325,
                    0.9925,
                ],
                std=[
                    0.18675,
                    0.1745,
                    0.18475,
                    0.192,
                    0.21225,
                    0.217,
                    0.21225,
                    0.2285,
                ],
                normalization="standard",
                p=1.0,
            ),
            A.ToTensorV2(p=1.0, transpose_mask=False),
        ]
    )

    dset = NpzSegmentationDataset(FULL_DATA_DIR, transforms=test_trans)
    len(dset)
    return (dset,)


@app.cell
def _(dset):
    BATCH_SIZE = 16
    dataloader = DataLoader(
        dataset=dset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    len(dataloader)
    return BATCH_SIZE, dataloader


@app.cell
def _(BATCH_SIZE, dataloader, device, dset, model):
    records: list[dict] = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            images, masks = batch[0].to(device), batch[1].to(device)
            logits = model(images)

            # Squeeze to [B, H, W]
            logits = logits.squeeze(1)
            masks = masks.squeeze(1).float()

            losses = []
            for j, (x, y) in enumerate(zip(logits, masks)):
                loss = model.loss_fn(logits, masks).detach().cpu().item()

                file_path = dset.chips[idx * BATCH_SIZE + j]
                records.append(
                    {
                        "file": str(file_path),
                        "name": Path(file_path).name,
                        "loss": loss,
                    }
                )

    loss_df = pl.DataFrame(records).sort("loss", descending=True)
    loss_df
    return (loss_df,)


@app.cell
def _(loss_df):
    i = 0
    for r in loss_df.filter(pl.col("loss") > 0.8).iter_rows(named=True):
        p = Path(r["file"])
        dest = Path(f"./data/kelp-ps8b/1024x512/train/{p.name}")
        if not dest.exists():
            dest.hardlink_to(p)
            i += 1

    print(i, "examples added")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
