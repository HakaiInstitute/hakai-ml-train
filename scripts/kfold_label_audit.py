"""K-fold label audit orchestrator.

Usage:
    uv run python scripts/kfold_label_audit.py \\
        --config configs/mussels-goosenecks-rgb/dpt_dinov3_vitl16.yaml \\
        --output-dir audits/mussel_gooseneck_v1

See docs/superpowers/specs/2026-04-27-kfold-label-audit-design.md for the design.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import shutil
import sys
from pathlib import Path

import datasets as hf_datasets
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

# Make `src` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.aggregate import aggregate_fold_probs
from src.audit.folds import (
    compute_fold_assignments,
    load_fold_assignments,
    save_fold_assignments,
)
from src.audit.scoring import per_pixel_scores, per_tile_metrics_batch
from src.data import WebDataModule
from src.models.smp import SMPMulticlassSegmentationModel

NUM_FOLDS = 5
FOLD_SEED = 42
POOL_SPLITS = ("train", "validation")
CONFIDENCE_THRESHOLD = 0.9


def _import_class(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _instantiate_model(model_cfg: dict) -> pl.LightningModule:
    cls = _import_class(model_cfg["class_path"])
    return cls(**model_cfg.get("init_args", {}))


def _instantiate_data(
    data_cfg: dict, fold_idx: int, fold_assignments_path: Path
) -> WebDataModule:
    cls = _import_class(data_cfg["class_path"])
    init_args = dict(data_cfg.get("init_args", {}))
    init_args["fold_idx"] = fold_idx
    init_args["num_folds"] = NUM_FOLDS
    init_args["pool_splits"] = list(POOL_SPLITS)
    init_args["fold_seed"] = FOLD_SEED
    init_args["fold_assignments_path"] = str(fold_assignments_path)
    return cls(**init_args)


def _build_callbacks(fold_dir: Path) -> list:
    return [
        ModelCheckpoint(
            dirpath=fold_dir / "checkpoint",
            filename="best-{epoch:02d}-{val/iou_epoch:.4f}",
            auto_insert_metric_name=False,
            monitor="val/iou_epoch",
            mode="max",
            save_last=True,
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/iou_epoch",
            mode="max",
            patience=30,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def _build_trainer(
    base_trainer_cfg: dict, fold_dir: Path, run_name: str, group: str, callbacks: list
) -> Trainer:
    cfg = copy.deepcopy(base_trainer_cfg)
    cfg["default_root_dir"] = str(fold_dir)
    cfg.pop("callbacks", None)  # We replace these per-fold.
    cfg.pop("logger", None)

    logger = WandbLogger(
        entity="hakai",
        project="kom-mussel-gooseneck-rgb",
        name=run_name,
        group=group,
        log_model=False,  # Per-fold checkpoints are kept locally; don't bloat W&B.
        tags=["kfold_audit"],
    )
    return Trainer(**cfg, callbacks=callbacks, logger=logger)


def _run_fold_training(
    config: dict, fold_idx: int, output_dir: Path, run_group: str
) -> Path:
    """Run fit on one fold. Returns the path to the best checkpoint."""
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = _instantiate_model(config["model"])
    dm = _instantiate_data(
        config["data"],
        fold_idx=fold_idx,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "train"

    callbacks = _build_callbacks(fold_dir)
    trainer = _build_trainer(
        config["trainer"],
        fold_dir,
        run_name=f"kfold_audit_fold_{fold_idx}",
        group=run_group,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)
    ckpt_cb = trainer.checkpoint_callback
    best_path = getattr(ckpt_cb, "best_model_path", None)
    if not best_path:
        raise RuntimeError(f"Fold {fold_idx} did not produce a best checkpoint")
    return Path(best_path)


def _run_fold_prediction(
    config: dict,
    fold_idx: int,
    output_dir: Path,
    best_ckpt: Path,
    n_classes: int,
    height: int,
    width: int,
) -> None:
    """Run predict on the held-out fold and write probs.zarr."""
    fold_dir = output_dir / f"fold_{fold_idx}"

    model = SMPMulticlassSegmentationModel.load_from_checkpoint(str(best_ckpt))

    dm = _instantiate_data(
        config["data"],
        fold_idx=fold_idx,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "predict"

    # Predict on a clean trainer with no callbacks/loggers.
    trainer_cfg = copy.deepcopy(config["trainer"])
    trainer_cfg.pop("callbacks", None)
    trainer_cfg.pop("logger", None)
    trainer_cfg["default_root_dir"] = str(fold_dir)
    pred_trainer = Trainer(**trainer_cfg, enable_progress_bar=True)

    batch_outputs = pred_trainer.predict(model, datamodule=dm)
    # batch_outputs: list of (B, C, H, W) float16 tensors
    probs = torch.cat(batch_outputs, dim=0).numpy()
    if (
        probs.shape[1] != n_classes
        or probs.shape[2] != height
        or probs.shape[3] != width
    ):
        raise RuntimeError(
            f"Fold {fold_idx} prediction shape {probs.shape} disagrees with expected "
            f"(N, {n_classes}, {height}, {width})"
        )

    n = probs.shape[0]
    arr = zarr.open(
        str(fold_dir / "probs.zarr"),
        mode="w",
        shape=(n, n_classes, height, width),
        chunks=(min(16, n), n_classes, height, width),
        dtype=np.float16,
    )
    arr[:] = probs

    (fold_dir / "_SUCCESS").touch()


def _peek_dataset_dimensions(config: dict, output_dir: Path) -> tuple[int, int, int]:
    """Load a single sample to determine (num_classes, H, W) without iterating the full dataset."""
    n_classes = config["model"]["init_args"]["num_classes"]
    dm = _instantiate_data(
        config["data"],
        fold_idx=0,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "predict"
    dm.setup()
    sample_img, _ = dm.ds_predict[0]
    h, w = sample_img.shape[-2:]
    return n_classes, int(h), int(w)


def _aggregate_and_score(
    output_dir: Path,
    fold_assignments: pd.DataFrame,
    pool_root: str,
    n_classes: int,
    height: int,
    width: int,
    ignore_index: int,
) -> None:
    fold_dirs = {k: output_dir / f"fold_{k}" for k in range(NUM_FOLDS)}
    oof_path = output_dir / "oof_probs.zarr"
    aggregate_fold_probs(
        fold_assignments, fold_dirs, oof_path, n_classes, height, width
    )

    oof = zarr.open(str(oof_path), mode="r")
    n_total = oof.shape[0]

    score_arr = zarr.open(
        str(output_dir / "per_pixel_scores.zarr"),
        mode="w",
        shape=(n_total, height, width),
        chunks=(min(16, n_total), height, width),
        dtype=np.float16,
    )

    # Build the pool once for label lookup (matches WebDataModule's pool construction).
    loaded = [hf_datasets.load_dataset(pool_root, split=s) for s in POOL_SPLITS]
    pool = hf_datasets.concatenate_datasets(loaded)
    split_offsets = {}
    offset = 0
    for name, d in zip(POOL_SPLITS, loaded, strict=True):
        split_offsets[name] = offset
        offset += len(d)

    rows = []
    chunk = 16
    fa_sorted = fold_assignments.iloc[
        np.argsort(fold_assignments["row_index"].to_numpy())
    ].reset_index(drop=True)
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        probs_chunk = oof[start:end].astype(np.float32)

        labels_chunk = np.zeros((end - start, height, width), dtype=np.int64)
        for i, row in enumerate(fa_sorted.iloc[start:end].itertuples()):
            pool_idx = split_offsets[row.original_split] + int(row.original_index)
            sample = pool[pool_idx]
            label = np.array(sample["label.tif"]).astype(np.int64)
            if label.shape != (height, width):
                raise RuntimeError(
                    f"Label shape mismatch at row {start + i}: got {label.shape}, "
                    f"expected ({height}, {width})"
                )
            labels_chunk[i] = label

        scores = per_pixel_scores(probs_chunk, labels_chunk, ignore_index=ignore_index)
        score_arr[start:end] = scores.astype(np.float16)

        tile_metrics = per_tile_metrics_batch(
            probs_chunk,
            labels_chunk,
            ignore_index=ignore_index,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        for i, m in enumerate(tile_metrics.to_dict("records")):
            row = fa_sorted.iloc[start + i]
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "original_split": row["original_split"],
                    "original_index": int(row["original_index"]),
                    "fold_idx": int(row["fold_idx"]),
                    "row_index": int(row["row_index"]),
                    **m,
                }
            )

    df = pd.DataFrame(rows).sort_values(by="conf_disagree_pct", ascending=False)
    df.to_csv(output_dir / "tile_scores.csv", index=False)


def _ignore_index_from_config(config: dict) -> int:
    return int(config["model"]["init_args"].get("ignore_index", -100))


def _pool_root_from_config(config: dict) -> str:
    return config["data"]["init_args"]["train_chip_dir"]


def _save_config_snapshot(config: dict, output_dir: Path) -> None:
    with open(output_dir / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(config, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to base YAML config."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to put audits/<run-name>/...",
    )
    parser.add_argument(
        "--force", action="store_true", help="Wipe output-dir before running."
    )
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    _save_config_snapshot(config, args.output_dir)

    # Compute or load fold assignments.
    fa_path = args.output_dir / "fold_assignments.parquet"
    if fa_path.exists():
        fold_assignments = load_fold_assignments(fa_path)
        print(f"Loaded existing fold assignments from {fa_path}")
    else:
        pool_root = _pool_root_from_config(config)
        loaded = [hf_datasets.load_dataset(pool_root, split=s) for s in POOL_SPLITS]
        split_sizes = {
            name: len(d) for name, d in zip(POOL_SPLITS, loaded, strict=True)
        }
        fold_assignments = compute_fold_assignments(
            split_sizes, num_folds=NUM_FOLDS, seed=FOLD_SEED
        )
        save_fold_assignments(fold_assignments, fa_path)
        print(f"Wrote new fold assignments to {fa_path}")

    run_group = f"kfold_audit_{args.output_dir.name}"

    for fold_idx in range(NUM_FOLDS):
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        if (fold_dir / "_SUCCESS").exists():
            print(f"Fold {fold_idx}: _SUCCESS present, skipping.")
            continue
        print(f"=== Fold {fold_idx}: training ===")
        best_ckpt = _run_fold_training(config, fold_idx, args.output_dir, run_group)
        print(f"=== Fold {fold_idx}: predicting (ckpt={best_ckpt}) ===")
        dim_cache = args.output_dir / "dimensions.yaml"
        if dim_cache.exists():
            with open(dim_cache) as f:
                d = yaml.safe_load(f)
            n_classes, h, w = d["num_classes"], d["height"], d["width"]
        else:
            n_classes, h, w = _peek_dataset_dimensions(config, args.output_dir)
            with open(dim_cache, "w") as f:
                yaml.safe_dump({"num_classes": n_classes, "height": h, "width": w}, f)

        _run_fold_prediction(
            config,
            fold_idx,
            args.output_dir,
            best_ckpt,
            n_classes=n_classes,
            height=h,
            width=w,
        )
        print(f"Fold {fold_idx} complete.")

    print("=== Aggregating OOF predictions ===")
    with open(args.output_dir / "dimensions.yaml") as f:
        d = yaml.safe_load(f)
    n_classes, h, w = d["num_classes"], d["height"], d["width"]
    _aggregate_and_score(
        args.output_dir,
        fold_assignments,
        pool_root=_pool_root_from_config(config),
        n_classes=n_classes,
        height=h,
        width=w,
        ignore_index=_ignore_index_from_config(config),
    )
    print(f"Done. tile_scores.csv at {args.output_dir / 'tile_scores.csv'}")


if __name__ == "__main__":
    main()
