"""Train directly from GeoTIFF/COG mosaics with torchgeo, without pre-chipping.

Expects ``<root>/{train,val,test}/{images,labels}/<name>_<YYYYMMDD>.tif``, with each
label raster named like its image. Every scene is its own torchgeo dataset
(``SceneImagery(image) & SceneLabels(label)``), so an image is only ever paired with
its own label, overlapping multi-date mosaics are never mixed, and torchgeo's
samplers only have to cover one scene's bounds at a time.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path
from typing import Any

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import rasterio
import rasterio.features
import shapely
import torch
from affine import Affine
from rasterio.crs import CRS as RasterioCRS
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Sampler
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.samplers import GriddedPatchSampler, RandomPatchSampler

from hakai_ml_train.data import DataModule

FILENAME_REGEX = r"^(?P<name>.+)_(?P<date>\d{8})\.tif$"


class SceneImagery(RasterDataset):
    filename_regex = FILENAME_REGEX
    date_format = "%Y%m%d"
    is_image = True
    all_bands = ("red", "green", "blue")
    rgb_bands = all_bands

    def footprint_from_datasource(self, src):
        # Outline of the valid pixels from a coarse (~2048 px) read of the overviews,
        # so samplers stay inside imagery instead of the file's bounding box.
        f = max(1, max(src.width, src.height) // 2048)
        shape = (max(1, src.height // f), max(1, src.width // f))
        valid = src.dataset_mask(out_shape=shape) > 0
        transform = src.transform * Affine.scale(
            src.width / shape[1], src.height / shape[0]
        )
        # Fill no-imagery specks smaller than 1000 coarse pixels: thousands of tiny
        # holes make the samplers' geometry operations very slow.
        min_hole = 1000 * abs(transform.a * transform.e)
        polys = []
        for geom, _ in rasterio.features.shapes(
            valid.astype(np.uint8), mask=valid, transform=transform
        ):
            p = shapely.geometry.shape(geom)
            holes = [h for h in p.interiors if shapely.Polygon(h).area >= min_hole]
            polys.append(shapely.Polygon(p.exterior, holes))
        if not polys:
            return None
        return shapely.make_valid(shapely.union_all(polys).simplify(abs(transform.a)))


class SceneLabels(RasterDataset):
    filename_regex = FILENAME_REGEX
    date_format = "%Y%m%d"
    is_image = False


class AlbumentationsSample:
    """Remaps labels, ignores no-imagery pixels, then runs an albumentations pipeline.

    Returns an ``(image, mask)`` tuple, the same batch format as ``DataModule``.
    """

    def __init__(
        self,
        transform: A.BasicTransform | None,
        label_remap: dict[int, int] | None = None,
        ignore_index: int = -100,
    ):
        self.transform = transform
        self.label_remap = label_remap
        self.ignore_index = ignore_index
        self._pid = None

    def __call__(self, sample: dict[str, torch.Tensor]):
        if self.transform is not None and self._pid != os.getpid():
            # albumentations doesn't reseed per DataLoader worker; torch's worker
            # seed does, and derives from seed_everything.
            self.transform.set_random_seed(torch.initial_seed() % 2**32)
            self._pid = os.getpid()
        image = sample["image"]
        label = sample["mask"].squeeze(0).long()
        mask = label.clone()
        for src, dst in (self.label_remap or {}).items():
            mask[label == src] = dst
        mask[(image == 0).all(dim=0)] = self.ignore_index

        image = image.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        mask = mask.numpy().astype(np.int16)
        if self.transform is None:
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask)
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


class SceneCollection(Dataset):
    """One torchgeo dataset per scene, indexed with ``(scene index, geoslice)`` keys."""

    def __init__(self, datasets, names, image_paths, label_paths):
        self.datasets, self.names = datasets, names
        self.image_paths, self.label_paths = image_paths, label_paths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, key):
        i, query = key
        return self.datasets[i][query]


def foreground_roi(label_path, dataset, size, overview_factor=8):
    """Outline of pixels with label > 0, grown by half a patch diagonal.

    Returned in the dataset's CRS, or None if the scene has no foreground.
    ``RandomPatchSampler`` shrinks its roi back in by the same distance, so patch
    centres land on or next to foreground.
    """
    grow = math.hypot(size * dataset.res[0] / 2, size * dataset.res[1] / 2)
    with rasterio.open(label_path) as full:
        factors = full.overviews(1)
    level = max(
        (i for i, f in enumerate(factors) if f <= overview_factor), default=None
    )
    opts = {} if level is None else {"overview_level": level}
    with (
        rasterio.open(label_path, **opts) as src,
        WarpedVRT(src, crs=RasterioCRS.from_wkt(dataset.crs.to_wkt())) as vrt,
    ):
        fg = vrt.read(1) > 0
        transform = vrt.transform
    # Max-pool to cells of about 1/8 of the growth distance, so even single plants survive.
    k = max(1, int(grow / 8 / abs(transform.a)))
    h, w = fg.shape
    fg = np.pad(fg, ((0, -h % k), (0, -w % k)))
    fg = fg.reshape(fg.shape[0] // k, k, fg.shape[1] // k, k).any(axis=(1, 3))
    if not fg.any():
        return None
    transform = transform * Affine.scale(k)
    cell = abs(transform.a)
    grown = ndimage.distance_transform_edt(~fg) * cell <= grow
    polys = []
    for geom, _ in rasterio.features.shapes(
        grown.astype(np.uint8), mask=grown, transform=transform
    ):
        p = shapely.geometry.shape(geom)
        holes = [r for r in p.interiors if shapely.Polygon(r).area >= grow**2]
        polys.append(shapely.Polygon(p.exterior, holes))
    return shapely.make_valid(shapely.union_all(polys).simplify(cell))


def native_valid_pixels(path) -> float:
    """Number of valid (imaged) pixels at the file's native resolution, from a coarse mask read."""
    with rasterio.open(path) as src:
        f = max(1, max(src.width, src.height) // 2048)
        shape = (max(1, src.height // f), max(1, src.width // f))
        valid = src.dataset_mask(out_shape=shape) > 0
        return float(valid.mean()) * src.width * src.height


class SceneRandomSampler(Sampler):
    """Random patches from every scene, with new positions each epoch.

    Patches are split across scenes in proportion to each scene's valid-pixel count
    at its native resolution (the same weighting a chip dataset has), then
    ``foreground_fraction`` of each scene's patches are centred on foreground and
    the rest come from anywhere in the imagery.
    """

    def __init__(self, collection, size, length, seed, foreground_fraction=0.0):
        self.rng = np.random.default_rng(seed)
        self.native_pixels = np.array(
            [native_valid_pixels(p) for p in collection.image_paths]
        )
        share = length * self.native_pixels / self.native_pixels.sum()
        self.counts = np.floor(share).astype(int)
        # Largest remainders get the leftover patches.
        self.counts[np.argsort(self.counts - share)[: length - self.counts.sum()]] += 1
        # Built once: RandomPatchSampler draws new positions from the shared generator
        # on every pass.
        self.samplers, self.foreground_counts = [], np.zeros_like(self.counts)
        for i, (ds, n) in enumerate(zip(collection.datasets, self.counts, strict=True)):
            n_fg = int(round(foreground_fraction * n))
            if n_fg:
                roi = foreground_roi(collection.label_paths[i], ds, size)
                fg_sampler = (
                    None
                    if roi is None
                    else RandomPatchSampler(
                        ds, size=size, length=n_fg, roi=roi, generator=self.rng
                    )
                )
                if fg_sampler is None or fg_sampler.series.is_empty.all():
                    n_fg = 0  # no room for a patch on foreground
                else:
                    self.samplers.append((i, fg_sampler))
            if n - n_fg:
                self.samplers.append(
                    (
                        i,
                        RandomPatchSampler(
                            ds, size=size, length=int(n - n_fg), generator=self.rng
                        ),
                    )
                )
            self.foreground_counts[i] = n_fg

    def __len__(self):
        return int(self.counts.sum())

    def __iter__(self):
        keys = [(i, query) for i, sampler in self.samplers for query in sampler]
        for j in self.rng.permutation(len(keys)):
            yield keys[j]


class SceneGridSampler(Sampler):
    """A fixed, non-overlapping grid over every scene, chained; each pixel is visited once."""

    def __init__(self, collection, size):
        self.keys = [
            (i, query)
            for i, ds in enumerate(collection.datasets)
            for query in GriddedPatchSampler(ds, size=size)
        ]

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        return iter(self.keys)


# noinspection PyAbstractClass
class SceneDataModule(pl.LightningDataModule):
    """Samples patches on the fly from per-scene GeoTIFF/COG mosaics with torchgeo.

    Args:
        root: Directory holding ``{train,val,test}/{images,labels}/*.tif``.
        crs: CRS to read every scene in, e.g. ``EPSG:3156``. With ``res``, scenes
            in other CRSs are reprojected on the fly. ``None`` keeps each scene's
            native grid.
        res: Resolution in ``crs`` units, or ``None`` for each scene's native resolution.
        patch_size: Training patch size in pixels.
        train_samples_per_epoch: Random training patches per epoch.
        eval_patch_size: Non-overlapping grid patch size for validation and test.
        foreground_patch_fraction: Share of training patches centred on pixels with
            stored label > 0; the rest come from anywhere in the imagery.
        label_remap: Maps stored label values to training targets, as in ``DataModule``.
        ignore_index: Target value for no-imagery pixels (all bands 0).
        seed: Seed for the training patch positions.
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        crs: str | None = None,
        res: float | None = None,
        patch_size: int = 1024,
        train_samples_per_epoch: int = 1000,
        eval_patch_size: int = 1024,
        foreground_patch_fraction: float = 0.0,
        label_remap: dict[int, int] | None = None,
        ignore_index: int = -100,
        seed: int = 42,
        num_workers: int = os.cpu_count() or 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_transforms: Any | None = None,
        test_transforms: Any | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.crs, self.res = crs, res
        self.patch_size = patch_size
        self.train_samples_per_epoch = train_samples_per_epoch
        self.eval_patch_size = eval_patch_size
        self.foreground_patch_fraction = foreground_patch_fraction
        self.label_remap = label_remap
        self.ignore_index = ignore_index
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_trans = (
            A.from_dict(train_transforms) if train_transforms is not None else None
        )
        self.test_trans = (
            A.from_dict(test_transforms) if test_transforms is not None else None
        )

        self.datasets: dict[str, SceneCollection] = {}
        self.samplers: dict[str, Sampler] = {}

    def _make(self, split: str, transforms):
        transform = AlbumentationsSample(
            transforms, self.label_remap, self.ignore_index
        )
        names, datasets, image_paths, label_paths = [], [], [], []
        # torchgeo prints a "Converting ... res/CRS" line for every file it reprojects.
        with contextlib.redirect_stdout(io.StringIO()):
            for image in sorted((self.root / split / "images").glob("*.tif")):
                label = self.root / split / "labels" / image.name
                images = SceneImagery(image, crs=self.crs, res=self.res)
                labels = SceneLabels(label)
                datasets.append(
                    IntersectionDataset(images, labels, transforms=transform)
                )
                names.append(image.stem)
                image_paths.append(image)
                label_paths.append(label)
        if not datasets:
            raise FileNotFoundError(
                f"No scenes found in {self.root / split / 'images'}"
            )

        collection = SceneCollection(datasets, names, image_paths, label_paths)
        self.datasets[split] = collection
        if split == "train":
            self.samplers[split] = SceneRandomSampler(
                collection,
                self.patch_size,
                self.train_samples_per_epoch,
                self.seed,
                foreground_fraction=self.foreground_patch_fraction,
            )
        else:
            self.samplers[split] = SceneGridSampler(collection, self.eval_patch_size)

    def setup(self, stage: str | None = None):
        splits = {"fit": ["train", "val"], "validate": ["val"], "test": ["test"]}
        for split in splits.get(stage, ["train", "val", "test"]):
            if split not in self.datasets:
                trans = self.train_trans if split == "train" else self.test_trans
                self._make(split, trans)

    def _loader(self, split: str) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            sampler=self.samplers[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")

    # Logs the albumentations pipelines to W&B, as DataModule does.
    on_after_batch_transfer = DataModule.on_after_batch_transfer
