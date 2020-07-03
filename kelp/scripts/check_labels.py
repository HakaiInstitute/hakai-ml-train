"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-07-03
Description: 
"""
from pathlib import Path
import numpy as np
from PIL import Image


if __name__ == '__main__':
    checked = 0
    for path in Path(".").glob("*.png"):
        img = np.asarray(Image.open(str(path)))

        if np.any(img > 1):
            print(np.unique(img, return_counts=True))

        if img.shape[0] != 512 or img.shape[1] !=512:
            print(str(path))

        checked += 1

    print(f"Checked {checked} files.")