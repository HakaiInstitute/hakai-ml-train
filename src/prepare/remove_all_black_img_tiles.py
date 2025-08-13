"""
This script removes tiles that contain images with only black pixels from the dataset.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def remove_empty_image_tiles(input_dir: Path) -> None:
    files = list(input_dir.glob("**/*.npz"))
    removed_count = 0

    for file in tqdm(files):
        data = np.load(file)
        img = data.get("image", None)
        empty_img = np.all(img == 0)
        if empty_img:
            file.unlink()
            removed_count += 1

    print(f"Removed {removed_count} tiles with empty image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    remove_empty_image_tiles(args.input_dir)
