# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-26
# Description: Split a dataset into train/test splits
import shutil
from functools import partial
from pathlib import Path

import fire
from sklearn.model_selection import train_test_split
from tqdm.contrib import concurrent


def _move_files(x_files, y_files, dst_dir):
    dst_x = Path(dst_dir).joinpath("x")
    dst_x.mkdir(parents=True, exist_ok=True)
    move_x = partial(shutil.move, dst=str(dst_x))
    concurrent.thread_map(move_x, [str(p) for p in x_files])

    dst_y = Path(dst_dir).joinpath("y")
    dst_y.mkdir(parents=True, exist_ok=True)
    move_y = partial(shutil.move, dst=str(dst_y))
    concurrent.thread_map(move_y, [str(p) for p in y_files])


def split(dataset_dir, dest_dir, train_size=None, test_size=None, ext: str = "jpg"):
    x_files = sorted(list(Path(dataset_dir).joinpath("x").glob(f"*.{ext}")))
    y_files = sorted(list(Path(dataset_dir).joinpath("y").glob(f"*.{ext}")))
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_files, y_files, train_size=train_size, test_size=test_size, random_state=42)

    print("Moving training files")
    _move_files(x_train, y_train, Path(dest_dir).joinpath("train"))

    print("Moving testing files")
    _move_files(x_eval, y_eval, Path(dest_dir).joinpath("eval"))


if __name__ == '__main__':
    fire.Fire(split)
