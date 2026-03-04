"""Visualize augmentation pipeline from a training config.

Usage:
    python visualize_augmentations.py configs/mussels-goosenecks-rgb/segformer_b3.yaml
    python visualize_augmentations.py configs/mussels-goosenecks-rgb/segformer_b3.yaml --n 16 --cols 4
    python visualize_augmentations.py configs/mussels-goosenecks-rgb/segformer_b3.yaml --output augmentations.png
"""

import argparse
import importlib
import random
import sys
from pathlib import Path

import albumentations as A
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Colour palette for mask classes (index 0 = background)
CLASS_COLOURS = [
    (0.15, 0.15, 0.15),  # bg  – dark grey
    (0.20, 0.60, 1.00),  # class 1 – blue
    (1.00, 0.50, 0.10),  # class 2 – orange
    (0.20, 0.80, 0.20),  # class 3 – green
    (0.90, 0.20, 0.20),  # class 4 – red
    (0.80, 0.20, 0.90),  # class 5 – purple
]
IGNORE_COLOUR = (1.0, 0.0, 1.0)  # magenta for ignore_index pixels


def _import_class(class_path: str):
    module_path, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _denormalize(tensor, mean, std):
    """Reverse ImageNet-style normalisation; returns HWC uint8 numpy array."""
    img = tensor.numpy().astype(np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _mask_to_rgb(mask_np, n_classes, ignore_index=-100):
    """Convert integer mask to an RGB image using CLASS_COLOURS."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(n_classes):
        colour = CLASS_COLOURS[c % len(CLASS_COLOURS)]
        rgb[mask_np == c] = colour
    rgb[mask_np == ignore_index] = IGNORE_COLOUR
    return rgb


def _extract_norm_params(transforms_dict):
    """Walk the serialised Albumentations dict to find Normalize params."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if transforms_dict is None:
        return mean, std
    queue = [transforms_dict]
    while queue:
        node = queue.pop()
        if isinstance(node, dict):
            if node.get("__class_fullname__") == "Normalize":
                mean = node.get("mean", mean)
                std = node.get("std", std)
                return mean, std
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return mean, std


def build_datamodule(cfg: dict):
    """Instantiate the LightningDataModule described in the config."""
    data_cfg = cfg["data"]
    cls = _import_class(data_cfg["class_path"])
    init_args = data_cfg.get("init_args", {})
    return cls(**init_args)


def get_raw_dataset(dm):
    """Return the underlying dataset with NO transforms applied."""
    import copy

    dm_raw = copy.copy(dm)
    dm_raw.train_trans = None
    dm_raw.setup(stage="fit")
    return dm_raw.ds_train


def get_aug_dataset(dm):
    dm.setup(stage="fit")
    return dm.ds_train


def main():
    parser = argparse.ArgumentParser(description="Visualise training augmentations")
    parser.add_argument("config", help="Path to Lightning YAML config")
    parser.add_argument(
        "--n", type=int, default=8, help="Number of samples to show (default 8)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Samples per row in the augmented grid (default 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to file instead of showing",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to visualise (default train)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Pull class names and ignore_index from model config if present
    model_args = cfg.get("model", {}).get("init_args", {})
    class_names = model_args.get("class_names", None)
    ignore_index = model_args.get("ignore_index", -100)
    n_classes = model_args.get("num_classes", len(class_names) if class_names else 2)

    # Detect normalisation params so we can denormalise for display
    data_args = cfg.get("data", {}).get("init_args", {})
    train_transforms_dict = data_args.get("train_transforms")
    mean, std = _extract_norm_params(train_transforms_dict)

    print(f"Loading data module from config: {cfg_path}")
    dm = build_datamodule(cfg)

    # Raw dataset (no augmentations, no normalisation – raw numpy arrays)
    raw_ds = get_raw_dataset(dm)

    # Augmented dataset
    aug_ds = get_aug_dataset(dm)

    n = min(args.n, len(raw_ds))
    indices = random.sample(range(len(raw_ds)), n)

    cols = args.cols
    rows = (n + cols - 1) // cols

    # Layout: each "cell" shows [raw | augmented | mask] side-by-side
    fig_cols = cols * 3
    fig_rows = rows
    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(fig_cols * 2.5, fig_rows * 2.5),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for plot_idx, ds_idx in enumerate(indices):
        row = plot_idx // cols
        col = plot_idx % cols
        base_col = col * 3

        # --- Raw image ---
        raw_image, raw_mask = raw_ds[ds_idx]
        if isinstance(raw_image, np.ndarray):
            raw_img_disp = raw_image
        else:
            raw_img_disp = raw_image.numpy()
        if raw_img_disp.ndim == 3 and raw_img_disp.shape[0] in (1, 3, 4):
            raw_img_disp = raw_img_disp.transpose(1, 2, 0)
        if raw_img_disp.dtype != np.uint8:
            raw_img_disp = np.clip(raw_img_disp, 0, 255).astype(np.uint8)

        # --- Augmented image + mask ---
        aug_image, aug_mask = aug_ds[ds_idx]
        aug_img_disp = _denormalize(aug_image, mean, std)

        mask_np = aug_mask.numpy() if hasattr(aug_mask, "numpy") else np.array(aug_mask)
        mask_rgb = _mask_to_rgb(mask_np, n_classes, ignore_index)

        ax_raw = axes[row, base_col]
        ax_aug = axes[row, base_col + 1]
        ax_mask = axes[row, base_col + 2]

        ax_raw.imshow(raw_img_disp[:, :, :3])
        ax_aug.imshow(aug_img_disp[:, :, :3])
        ax_mask.imshow(mask_rgb)

        if row == 0:
            ax_raw.set_title("Original", fontsize=8, pad=2)
            ax_aug.set_title("Augmented", fontsize=8, pad=2)
            ax_mask.set_title("Mask", fontsize=8, pad=2)

    # Legend for mask colours
    legend_handles = []
    names = class_names if class_names else [f"class_{i}" for i in range(n_classes)]
    for i, name in enumerate(names):
        colour = CLASS_COLOURS[i % len(CLASS_COLOURS)]
        legend_handles.append(mpatches.Patch(color=colour, label=f"{i}: {name}"))
    legend_handles.append(
        mpatches.Patch(color=IGNORE_COLOUR, label=f"ignore ({ignore_index})")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    split_label = args.split
    fig.suptitle(
        f"Augmentation preview – {cfg_path.name}  ({split_label}, n={n})",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if args.output:
        out_path = Path(args.output)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
