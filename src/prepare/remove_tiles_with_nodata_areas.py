"""This script removes tiles that have imagery containing all black areas, which are assumed to be nodata.

This is useful for creating a training dataset for SSL models that try to reconstruct masked imagery since we don't want
them to learn to reconstruct black areas that are not part of the actual scene.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def remove_tiles_with_nodata_areas(input_dir: Path, num_channels: int = 3) -> None:
    files = list(input_dir.glob("**/*.npz"))
    removed_count = 0

    for file in tqdm(files):
        data = np.load(file)
        img = data.get("image", None)
        if img is None:
            print(f"Image not present in file {file}; skipping")
            continue

        # The assumption is nodata areas are all black (0, 0, 0) in RGB images
        is_nodata = (img == 0).sum(axis=2) == num_channels
        has_nodata = np.any(is_nodata)

        if has_nodata:
            file.unlink()
            removed_count += 1

    print(f"Removed {removed_count} tiles that contained nodata pixels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "-n--num_channels",
        type=int,
        default=3,
        help="Number of channels in the image (default: 3 for RGB)",
    )
    args = parser.parse_args()

    remove_tiles_with_nodata_areas(args.input_dir, args.num_channels)
