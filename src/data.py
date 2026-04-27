from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import albumentations as A
import datasets as hf_datasets
import lightning.pytorch as pl
import numpy as np
import torch
from albumentations import ToTensorV2, to_dict
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset


class NpzSegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during model train and validation phases."""

    def __init__(
        self,
        root: str,
        *args,
        **kwargs,
    ):
        super().__init__(root, *args, **kwargs)
        self.chips = sorted(Path(root).glob(f"*.npz"))

    def __len__(self):
        return len(self.chips)

    # noinspection DuplicatedCode
    def __getitem__(self, idx):
        chip_name = self.chips[idx]
        data = np.load(chip_name)
        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=data["image"], mask=data["label"])
                return augmented["image"], augmented["mask"]

        return data["image"], data["label"]


class WebDataset(Dataset):
    """Load preprocessed image chips from a HuggingFace dataset.

    Expects the dataset to have "image.tif" and "label.tif" columns containing
    PIL images or numpy arrays.
    """

    def __init__(self, root: str, split: str = "train", transforms=None):
        super().__init__()
        self.transforms = transforms
        self._dataset = hf_datasets.load_dataset(root, split=split)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        image = np.array(sample["image.tif"])
        mask = np.array(sample["label.tif"])

        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=image, mask=mask)
                return augmented["image"], augmented["mask"]

        return image, mask


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_chip_dir: str,
        val_chip_dir: str,
        test_chip_dir: str,
        batch_size: int,
        num_workers: int = os.cpu_count() or 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_transforms: Any | None = None,
        test_transforms: Any | None = None,
    ):
        super().__init__()
        self.train_data_dir = train_chip_dir
        self.val_data_dir = val_chip_dir
        self.test_data_dir = test_chip_dir

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

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.ds_train = NpzSegmentationDataset(
                self.train_data_dir,
                transforms=self.train_trans,
            )
        if stage in ["fit", "validate"] or stage is None:
            self.ds_val = NpzSegmentationDataset(
                self.val_data_dir,
                transforms=self.test_trans,
            )
        if stage == "test":
            self.ds_test = NpzSegmentationDataset(
                self.test_data_dir,
                transforms=self.test_trans,
            )

    def teardown(self, stage: str | None = None) -> None:
        del self.ds_train
        del self.ds_val
        del self.ds_test

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.ds_train,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.ds_val,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # This runs once at the start
        if not hasattr(self, "_logged_transforms"):
            if (
                self.trainer.logger
                and self.train_trans
                and hasattr(self.trainer.logger.experiment.config, "update")
            ):
                self.trainer.logger.experiment.config.update(
                    {
                        "train_transforms": to_dict(self.train_trans),
                        "test_transforms": to_dict(self.test_trans),
                    },
                    allow_val_change=True,
                )
            self._logged_transforms = True
        return batch


class WebDataModule(DataModule):
    def __init__(
        self,
        *args,
        fold_idx: int | None = None,
        num_folds: int = 5,
        pool_splits: tuple[str, ...] = ("train", "validation"),
        fold_seed: int = 42,
        fold_assignments_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.pool_splits = tuple(pool_splits)
        self.fold_seed = fold_seed
        self.fold_assignments_path = fold_assignments_path
        # Set by orchestrator before setup. "train" => train_dataset = pool minus
        # held-out, val_dataset = held-out. "predict" => predict_dataset = held-out.
        self.fold_mode: str = "train"
        self.ds_predict = None
        # Cached pool (concatenation of pool splits) and the assignments df.
        self._pool = None
        self._assignments = None
        self._split_offsets = None

    def _ensure_pool(self):
        if self._pool is not None:
            return
        # Use train_data_dir as the HF dataset root; pool_splits are the splits
        # within that dataset to concatenate.
        loaded = [
            hf_datasets.load_dataset(self.train_data_dir, split=s)
            for s in self.pool_splits
        ]
        self._pool = hf_datasets.concatenate_datasets(loaded)
        # Track the offset of each split inside the concatenated pool so we can
        # recover (original_split, original_index) from a pool index.
        self._split_offsets = []
        offset = 0
        for name, d in zip(self.pool_splits, loaded, strict=True):
            self._split_offsets.append((name, offset, offset + len(d)))
            offset += len(d)

    def _ensure_assignments(self):
        if self._assignments is not None:
            return
        from src.audit.folds import (
            compute_fold_assignments,
            load_fold_assignments,
        )

        if (
            self.fold_assignments_path is not None
            and Path(self.fold_assignments_path).exists()
        ):
            self._assignments = load_fold_assignments(Path(self.fold_assignments_path))
        else:
            self._ensure_pool()
            split_sizes = {
                name: end - start for name, start, end in self._split_offsets
            }
            self._assignments = compute_fold_assignments(
                split_sizes, num_folds=self.num_folds, seed=self.fold_seed
            )

    def _pool_indices_for_fold(
        self, fold_idx: int, holdout: bool, ordered: bool
    ) -> list[int]:
        """Return indices into the concatenated pool corresponding to the given fold.

        If holdout is True, returns indices in the held-out fold; otherwise
        returns the complement (training portion). When ordered=True, the
        result is ordered by row_index (used for predict so the fold's
        probs.zarr lines up with the aggregator's expectations).
        """
        self._ensure_pool()
        self._ensure_assignments()
        if holdout:
            sub = self._assignments[self._assignments["fold_idx"] == fold_idx]
        else:
            sub = self._assignments[self._assignments["fold_idx"] != fold_idx]
        if ordered:
            sub = sub.iloc[np.argsort(sub["row_index"].to_numpy())]

        offset_lookup = {name: start for name, start, _end in self._split_offsets}
        return [
            offset_lookup[row.original_split] + int(row.original_index)
            for row in sub.itertuples()
        ]

    def setup(self, stage: str | None = None):
        # Default behavior preserved when fold_idx is None.
        if self.fold_idx is None:
            if stage == "fit" or stage is None:
                self.ds_train = WebDataset(
                    self.train_data_dir,
                    split="train",
                    transforms=self.train_trans,
                )
            if stage in ["fit", "validate"] or stage is None:
                self.ds_val = WebDataset(
                    self.val_data_dir,
                    split="validation",
                    transforms=self.test_trans,
                )
            if stage == "test":
                self.ds_test = WebDataset(
                    self.test_data_dir,
                    split="test",
                    transforms=self.test_trans,
                )
            return

        # Fold mode.
        self._ensure_pool()
        self._ensure_assignments()

        if self.fold_mode == "train":
            train_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=False, ordered=False
            )
            holdout_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=True, ordered=False
            )
            self.ds_train = _PoolSubset(
                self._pool, train_indices, transforms=self.train_trans
            )
            self.ds_val = _PoolSubset(
                self._pool, holdout_indices, transforms=self.test_trans
            )
        elif self.fold_mode == "predict":
            ordered_pool_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=True, ordered=True
            )
            self.ds_predict = _PoolSubset(
                self._pool, ordered_pool_indices, transforms=self.test_trans
            )
        else:
            raise ValueError(f"Unknown fold_mode: {self.fold_mode}")

    def predict_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.ds_predict,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


class _PoolSubset(Dataset):
    """Thin wrapper that selects rows from a concatenated HF dataset and
    applies Albumentations transforms."""

    def __init__(self, pool, indices: list[int], transforms=None):
        self._pool = pool
        self._indices = indices
        self.transforms = transforms

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample = self._pool[self._indices[idx]]
        image = np.array(sample["image.tif"])
        mask = np.array(sample["label.tif"])
        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=image, mask=mask)
                return augmented["image"], augmented["mask"]
        return image, mask


class MAEDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        kwargs.pop("train_transforms", None)
        kwargs.pop("test_transforms", None)
        super().__init__(*args, **kwargs)
        self.train_trans = A.Compose(
            [
                A.D4(),
                # A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )
        self.test_trans = self.train_trans


class KOMBaselineRGBIDataModule(DataModule):
    @staticmethod
    def _rgbi_kelp_transform(img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # to float
        x = img.to(torch.float)
        # min-max scale
        x_unique = x.flatten().unique()
        min_ = x_unique[0]
        if len(x_unique) > 1:
            min_, _ = torch.kthvalue(x_unique, 2)
        max_ = x.flatten().max()
        return torch.clamp((x - min_) / (max_ - min_ + 1e-8), 0, 1)

    def __init__(self, *args, **kwargs):
        kwargs.pop("test_transforms", None)
        super().__init__(*args, **kwargs)
        self.test_trans = A.Compose(
            [
                ToTensorV2(),
                A.Lambda(name="normalize", image=self._rgbi_kelp_transform),
            ],
            p=1,
        )
