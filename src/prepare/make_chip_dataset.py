r"""
We will create chips of size `224 x 224` to feed them to the model, feel
free to experiment with other chip sizes as well.
   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir>

   Example:
   python preprocess_data.py data/ps_8b/ data/ps_8b_tiled_224_full
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

# HAKAI_LABELS = {
#     "bg": 0,
#     "unknown_kelp_species": 1,
#     "macro": 2,
#     "nereo": 3,
#     "noise": 4,
# }

# KATE_LABELS = {
#     "water": 0,
#     "kelp": 1,
#     "land": 2,
#     "nodata": 3,
#     "noise": 4,
# }


def remap_label(labels, band_remapping, device=None):
    if device is None:
        device = labels.device

    # Create a lookup tensor
    # Index 0->0, 1->-100, 2->1, 3->2
    lookup = torch.tensor(band_remapping, dtype=labels.dtype, device=device)

    # Handle out-of-bounds values
    mask = (labels >= 0) & (labels < len(lookup))
    result = torch.full_like(labels, -100)  # default value
    result[mask] = lookup[labels[mask]]

    return result


class RasterMosaicDataset(RasterDataset):
    is_image = True
    separate_files = False

    def __init__(self, *args, img_name, **kwargs):
        self.filename_glob = img_name
        super().__init__(*args, **kwargs)


class KomLabelsDataset(RasterDataset):
    is_image = False

    def __init__(self, *args, img_name, **kwargs):
        self.filename_glob = img_name
        super().__init__(*args, **kwargs)


def create_chips(
    out_root,
    name,
    dset,
    img_path,
    chip_size=224,
    chip_stride=224,
    num_bands=3,
    band_remapping=(0, 1),
    dtype=np.uint8,
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

        height = img.shape[2]
        width = img.shape[3]

        if height < chip_size or width < chip_size:
            # Toss chips that are smaller than the chip size
            continue

        # Convert to numpy arrays
        img_array = img[0, :num_bands].numpy()

        assert (
            img_array.max() <= np.iinfo(dtype).max
            and img_array.min() >= np.iinfo(dtype).min
        ), f"Image values should be in [{np.iinfo(dtype).min}, {np.iinfo(dtype).max}]"
        img_array = img_array.astype(dtype)
        label_array = remap_label(label, band_remapping).numpy().astype(np.int64)[0]

        # Skip all black images
        empty_img = np.all(img_array == 0)
        if empty_img:
            continue

        # Set black areas in image to 0 in label
        is_nodata = np.all(img_array == 0, axis=0)
        if np.any(is_nodata):
            label_array[is_nodata] = 0

        # Save the image as npz
        np.savez_compressed(
            out_dir / f"{img_path.stem}_{i}.npz",
            image=np.moveaxis(
                img_array, 0, -1
            ),  # Move channel dimension to last position
            label=label_array,  # Ensure label is 2D
        )


def load_dataset(data_dir, img_name):
    images = RasterMosaicDataset(paths=[str(data_dir / "images")], img_name=img_name)
    labels = KomLabelsDataset(paths=[str(data_dir / "labels")], img_name=img_name)

    return images & labels


def process_split(
    data_dir: Path,
    split: str,
    output_dir: Path,
    chip_size: int = 224,
    chip_stride: int = 224,
    num_bands: int = 3,
    band_remapping: tuple[int] = (0, 1),
    dtype: np.dtype = np.uint8,
):
    dir = data_dir / split
    imgs = sorted(dir.glob("images/*.tif", case_sensitive=False))
    for i, x in enumerate(imgs):
        print(i, x.name)

    for img in tqdm(imgs, desc=f"{split.capitalize()} chips"):
        ds = load_dataset(data_dir / split, img_name=img.name)
        create_chips(
            out_root=output_dir,
            name=split,
            dset=ds,
            img_path=img,
            chip_size=chip_size,
            chip_stride=chip_stride,
            num_bands=num_bands,
            band_remapping=band_remapping,
            dtype=dtype,
        )


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
        "--size", type=int, default=224, help="Size of the square chips."
    )
    parser.add_argument(
        "--num_bands", type=int, default=3, help="Number of bands in the chips to keep."
    )
    parser.add_argument("--stride", type=int, default=224, help="Stride for the chips.")

    parser.add_argument(
        "--dtype", type=np.dtype, default="uint8", help="Data type of the chips."
    )

    parser.add_argument("--remap", "-r", type=int, nargs="+", default=[0, -100, 1, 2])

    args = parser.parse_args()

    print(f"Creating dataset {Path(args.output_dir).name}")

    print(f"Pixel values will be remapped:")
    for i, v in enumerate(args.remap):
        print(f"{i} -> {v}")
    print("All other values will be set to -100.")

    for split in ["train", "val", "test"]:
        process_split(
            args.data_dir,
            split,
            args.output_dir,
            args.size,
            args.stride,
            args.num_bands,
            args.remap,
            args.dtype,
        )


if __name__ == "__main__":
    main()
