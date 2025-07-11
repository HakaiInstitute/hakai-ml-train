"""
This script removes tiles that contain only background labels from the dataset.

This is useful for creating a training set that focuses on interesting labels and removes the overly abundant tiles
that only contain e.g. water or rocky background.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def remove_bg_only_tiles(input_dir: Path) -> None:
    files = list(input_dir.glob("**/*.npz"))
    removed_count = 0

    for file in tqdm(files):
        data = np.load(file)
        label = data["label"]

        # The assumption is that noise is -100, bg is 0, and interesting labels are > 0
        contains_interesting_stuff = (label > 0).any()

        if not (contains_interesting_stuff):
            file.unlink()
            removed_count += 1

    print(f"Removed {removed_count} tiles with only background labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    remove_bg_only_tiles(args.input_dir)
