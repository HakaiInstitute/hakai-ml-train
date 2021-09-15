# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-25
# Description: Delete tiles without kelp or blank imagery
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import fire
import numpy as np
from PIL import Image
from tqdm.contrib import concurrent


class Filter(ABC):
    def __init__(self, dataset: str, ext: str = "png") -> None:
        super().__init__()
        self.dataset = dataset

        self.imgs_dir = Path(dataset).joinpath("x")
        self.labels_dir = Path(dataset).joinpath("y")

        self.labels = list(self.labels_dir.glob(f"*.{ext}"))
        self.imgs = list(self.imgs_dir.glob(f"*.{ext}"))

    @classmethod
    def process(cls, dataset: str, chunksize: int = 100) -> None:
        """
        Create a filter class, call the filtering function, and print the number of tiles that get removed.

        Parameters
        ----------
        dataset: str
            Path to the dataset to process.

        chunksize: int
            The length of files to pass to each multi-processing processing.

        Returns
        -------
            None
        """
        f = cls(dataset)
        print(f(chunksize), "image tiles removed")

    def remove_if_should(self, path: Path) -> bool:
        should_remove = self.should_be_removed(path)
        if should_remove:
            self.remove(path)
        return should_remove

    def mp_remove_if_should(self, paths: List[Path], chunksize: int) -> int:
        removed = concurrent.process_map(self.remove_if_should, paths, chunksize=chunksize)
        return sum(removed)

    @abstractmethod
    def remove(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_be_removed(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, chunksize: int) -> int:
        """Should call mp_remove_if_should in subclasses with appropriate file list."""
        raise NotImplementedError


class ImgFilter(Filter, ABC):
    """Deletes tiles based on the content of each img in a [img, label] pair."""

    def remove(self, path: Path) -> None:
        label_path = self.labels_dir.joinpath(path.name)
        label_path = label_path.with_name(f"label_{path.name}")
        if not label_path.is_file():
            print(f"LABEL DOES NOT EXIST: {label_path}")
        label_path.unlink()
        path.unlink()

    def __call__(self, chunksize: int = 100) -> int:
        return self.mp_remove_if_should(self.imgs, chunksize)


class LabelFilter(Filter, ABC):
    """Deletes tiles based on the content of each label in a [img, label] pair."""

    def remove(self, path: Path) -> None:
        img_path = self.imgs_dir.joinpath(path.name)
        img_path = img_path.with_name(path.name[len("label_"):])
        if not img_path.is_file():
            print(f"IMG DOES NOT EXIST: {img_path}")
        img_path.unlink()
        path.unlink()

    def __call__(self, chunksize: int = 100) -> int:
        return self.mp_remove_if_should(self.labels, chunksize)


class SkinnyImgFilter(ImgFilter):
    """Filters [image, label] tiles from the dataset where the height or width are below some threshold."""

    def __init__(self, dataset, min_height: int = 256, min_width: int = 256):
        super().__init__(dataset)

        self.min_height = min_height
        self.min_width = min_width

    @classmethod
    def process(cls, dataset: str, chunksize: int = 100, min_height: int = 256, min_width: int = 256) -> None:
        """
        Create a filter class, call the filtering function, and print the number of tiles that get removed.

        Parameters
        ----------
        dataset: str
            Path to the dataset to process.

        min_height: int
            The minimum acceptable height for an image and/or label

        min_width: int
            The minimum acceptable width for an image and/or label

        chunksize: int
            The length of files to pass to each multi-processing processing.

        Returns
        -------
            None
        """
        f = cls(dataset, min_height, min_width)
        print(f(chunksize), "image tiles removed")

    def should_be_removed(self, path: Path) -> bool:
        """
        Returns True if the label shape has a height or width below the specified min_height or min_width.

        Parameters
        ----------
        path: Union[Path, str]
            The path the label to test.

        Returns
        -------
            bool: A flag indicating if the label at path contains only the BG class.
        """
        img = np.asarray(Image.open(str(path)))
        if len(img.shape) < 3:
            return True
        else:
            h, w, _ = img.shape
            return h < self.min_height or w < self.min_width


class BGOnlyLabelFilter(LabelFilter):
    """Filters [image, label] tiles from the dataset where the label contains only the BG class."""
    def __init__(self, dataset, filter_prob: float = 1):
        super().__init__(dataset)
        self.filter_prob = filter_prob

    def should_be_removed(self, path: Path) -> bool:
        """
        Returns True if the label location at path contains only BG and no FG class. Indicates if the associated image
            and label tiles should be removed from the dataset.

        Parameters
        ----------
        path: Union[Path, str]
            The path the label to test.

        filter_prob: float
            The percentage of bg images to be filtered out. Defaults to 1 (all BG images).

        Returns
        -------
            bool: A flag indicating if the label at path contains only the BG class.
        """
        label = Image.open(str(path))
        p = np.random.uniform()
        return np.all(np.array(label) == 0) and p < self.filter_prob


class BlankImgFilter(ImgFilter):
    """Filters [image, label] tiles from the dataset where the image contains no data."""

    def should_be_removed(self, path: Path) -> bool:
        """
        Returns True if the image location path is blank and the associated image and label tiles should be removed from
            the dataset.

        Parameters
        ----------
        path: Union[Path, str]
            The path the image to test if blank.

        Returns
        -------
            bool: A flag indicating if the image at path is blank.
        """
        img = Image.open(str(path))
        return img.getbbox() is None


class FilterExtraImgs(ImgFilter):
    """Filters image tiles from the dataset where there is no matching label."""

    def should_be_removed(self, path: Path) -> bool:
        """
        Returns True if there is not a matching label file.

        Parameters
        ----------
        path: Union[Path, str]
            The path the image file to test.

        Returns
        -------
            bool: A flag indicating if the corresponding label is missing.
        """
        label_path = self.labels_dir.joinpath(path.name)
        label_path = label_path.with_name(f"label_{path.name}")
        return not label_path.is_file()

    def remove(self, path: Path) -> None:
        path.unlink()


class FilterExtraLabels(LabelFilter):
    """Filters image tiles from the dataset where there is no matching label."""

    def should_be_removed(self, path: Path) -> bool:
        """
        Returns True if there is not a matching img file.

        Parameters
        ----------
        path: Union[Path, str]
            The path to the label file to test.

        Returns
        -------
            bool: A flag indicating if the corresponding img is missing.
        """
        img_path = self.imgs_dir.joinpath(path.name)
        img_path = img_path.with_name(path.name[len("label_"):])
        return not img_path.is_file()

    def remove(self, path: Path) -> None:
        path.unlink()


if __name__ == '__main__':
    fire.Fire({
        "bg_only_labels": BGOnlyLabelFilter.process,
        "blank_imgs": BlankImgFilter.process,
        "skinny_labels": SkinnyImgFilter.process,
        "delete_extra_labels": FilterExtraLabels.process,
        "delete_extra_imgs": FilterExtraImgs.process,
    })
