r"""
We will create chips of size `224 x 224` to feed them to the model, feel
free to experiment with other chip sizes as well.
   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir> <chip_size> <chip_stride? (defaults to chip_size)>

   Example:
   python preprocess_data.py data/ps_8b/ data/ps_8b_tiled 224
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm.auto import tqdm

LABELS = {
    "bg": 0,
    "unknown_kelp_species": 1,
    "macro": 2,
    "nereo": 3,
    "noise": 4,
}


def remap_label(labels, device=None):
    if device is None:
        device = labels.device

    # Create a lookup tensor
    # Index 0->0, 1->-100, 2->1, 3->2
    lookup = torch.tensor([0, -100, 1, 2], dtype=labels.dtype, device=device)

    # Handle out-of-bounds values
    mask = (labels >= 0) & (labels < len(lookup))
    result = torch.full_like(labels, -100)  # default value
    result[mask] = lookup[labels[mask]]

    return result


def noise_pixels_count(label) -> int:
    return (label == LABELS["noise"]).sum()


def interesting_pixels_count(label) -> int:
    return ((label == LABELS["macro"]) | (label == LABELS["nereo"])).sum()


class RasterMosaicDataset(RasterDataset):
    is_image = True
    separate_files = False
    dtype = torch.uint8

    def __init__(self, *args, img_name, **kwargs):
        self.img_name = img_name
        super().__init__(*args, **kwargs)

    @property
    def filename_glob(self):
        return f"**/images/{self.img_name}"


class KomLabelsDataset(RasterDataset):
    is_image = False

    def __init__(self, *args, img_name, **kwargs):
        self.img_name = img_name
        super().__init__(*args, **kwargs)

    @property
    def filename_glob(self):
        return f"**/labels/{self.img_name}"


def create_chips(
    out_root, name, dset, img_path, chip_size=224, chip_stride=224, num_bands=3
):
    out_dir = out_root / name
    out_dir.mkdir(exist_ok=True, parents=True)

    sampler = GridGeoSampler(dset, size=chip_size, stride=chip_stride)
    dataloader = DataLoader(
        dset,
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=stack_samples,
    )

    for i, batch in enumerate(tqdm(dataloader, desc=img_path.stem)):
        img = batch["image"]
        label = batch["mask"]

        interesting_pixels = interesting_pixels_count(label)
        noise_pixels = noise_pixels_count(label)
        height = img.shape[2]
        width = img.shape[3]
        # total_pixels = height * width

        if noise_pixels > 0:
            continue

        if (
            height < chip_size
            or width < chip_size
            or (name in ["train", "val"] and interesting_pixels == 0)
            # or (name in ["train", "val"] and interesting_pixels / total_pixels < 0.002)
        ):
            continue

        # Convert to numpy arrays
        img_array = img[0, :num_bands].numpy()

        assert img_array.max() <= 255 and img_array.min() >= 0, (
            "Image values should be in [0, 255]"
        )
        img_array = img_array.astype(np.uint8)
        label_array = remap_label(label).numpy().astype(np.int64)

        # Save the image as npz
        np.savez_compressed(
            out_dir / f"{img_path.stem}_{i}.npz",
            image=np.moveaxis(
                img_array, 0, -1
            ),  # Move channel dimension to last position
            label=label_array[0],  # Ensure label is 2D
        )


def load_dataset(data_dir, img_name):
    images = RasterMosaicDataset(paths=[str(data_dir)], img_name=img_name)
    labels = KomLabelsDataset(paths=[str(data_dir)], img_name=img_name)

    return images & labels


def main():
    """
    Main function to process files and create chips.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=Path, help="Directory containing the input GeoTIFF files."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the output chips."
    )
    parser.add_argument(
        "--size", type=int, default=244, help="Size of the square chips."
    )
    parser.add_argument(
        "--num_bands", type=int, default=3, help="Number of bands in the chips."
    )
    parser.add_argument("--stride", type=int, default=244, help="Stride for the chips.")

    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    imgs = sorted(train_dir.glob("images/*.tif"))
    for i, x in enumerate(imgs):
        print(i, x.name)
    for img in tqdm(imgs, desc="Training chips"):
        print(img.name)
        train_ds = load_dataset(args.data_dir / "train", img_name=img.name)
        create_chips(
            args.output_dir,
            "train",
            train_ds,
            img_path=img,
            chip_size=args.size,
            chip_stride=args.stride,
            num_bands=args.num_bands,
        )

    val_dir = args.data_dir / "val"
    imgs = sorted(val_dir.glob("images/*.tif"))
    for i, x in enumerate(imgs):
        print(i, x.name)
    for img in tqdm(imgs, desc="Validation chips"):
        print(img.name)
        val_ds = load_dataset(args.data_dir / "val", img_name=img.name)
        create_chips(
            args.output_dir,
            "val",
            val_ds,
            img_path=img,
            chip_size=args.size,
            chip_stride=args.stride,
            num_bands=args.num_bands,
        )

    test_dir = args.data_dir / "test"
    imgs = sorted(test_dir.glob("images/*.tif"))
    for i, x in enumerate(imgs):
        print(i, x.name)
    for img in tqdm(imgs, desc="Test chips"):
        print(img.name)
        test_ds = load_dataset(args.data_dir / "test", img_name=img.name)
        create_chips(
            args.output_dir,
            "test",
            test_ds,
            img_path=img,
            chip_size=args.size,
            chip_stride=args.stride,
            num_bands=args.num_bands,
        )


if __name__ == "__main__":
    main()
