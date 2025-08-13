import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def remove_labels_in_black_areas(input_dir: Path) -> None:
    files = list(input_dir.glob("**/*.npz"))

    for file in tqdm(files):
        data = np.load(file)
        img = data.get("image", None)
        label = data.get("label", None)
        if img is None or label is None:
            print(f"Image not present in file {file}; skipping")
            continue

        _, _, num_channels = img.shape

        # The assumption is nodata areas are all black (0, 0, 0) in RGB images
        is_nodata = np.all(img == 0, axis=2)

        if not np.any(is_nodata):
            continue

        label[is_nodata] = 0  # Set nodata areas in label to 0

        # Save the tile back
        np.savez_compressed(file, image=img, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()

    remove_labels_in_black_areas(args.input_dir)
