import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import sys
    sys.path.append("src")
    return


@app.cell
def _():
    from data import DataModule

    return (DataModule,)


@app.cell
def _(DataModule):
    dm = DataModule(
        train_chip_dir="/home/taylor/data/kelp-ps8b/224x224/cali_bc/full/train/",
        val_chip_dir="/home/taylor/data/kelp-ps8b/1024x1024/cali_bc/full/val/",
        test_chip_dir="/home/taylor/data/kelp-ps8b/1024x1024/cali_bc/full/test/",
        batch_size=1,
        num_workers=2,
    )
    return (dm,)


@app.cell
def _(dm):
    dm.setup()
    return


@app.cell
def _(dm):
    import torch
    import tqdm

    _band_sum = None
    _band_sum_sq = None
    _pixel_count: int = 0

    for _x_batch, _y_batch in tqdm.tqdm(dm.train_dataloader(), total=len(dm.train_dataloader())):
        _x_batch = _x_batch.float() / 10000.0
        # shape: (batch, h, w, bands)
        _batch_size, _h, _w, _num_bands = _x_batch.shape

        if _band_sum is None:
            _band_sum = torch.zeros(_num_bands)
            _band_sum_sq = torch.zeros(_num_bands)

        # mask out nodata pixels: pixels where all bands are 0
        # valid_mask shape: (batch, h, w)
        _valid_mask = (_x_batch.sum(dim=-1) != 0)

        # sum over batch, h, w dimensions, excluding nodata pixels
        # expand mask to (batch, h, w, bands) for broadcasting
        _valid_mask_expanded = _valid_mask.unsqueeze(-1).expand_as(_x_batch)
        _band_sum += (_x_batch * _valid_mask_expanded).sum(dim=(0, 1, 2))
        _band_sum_sq += ((_x_batch ** 2) * _valid_mask_expanded).sum(dim=(0, 1, 2))
        _pixel_count += _valid_mask.sum().item()

    _band_mean = _band_sum / _pixel_count
    _band_std = (_band_sum_sq / _pixel_count - _band_mean ** 2).sqrt()

    print("Band-wise mean:", _band_mean)
    print("Band-wise std: ", _band_std)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
