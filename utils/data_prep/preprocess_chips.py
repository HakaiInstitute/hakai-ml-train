"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-07-02
Description: Modify image chips in place
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import fire
import numpy as np
from PIL import Image
from tqdm.contrib import concurrent


class Modifier(ABC):
    """Abstract base class for modifying image chips and labels in place."""

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
        Create a modifier class, call the modifying function, and print the number of tiles that get modified.

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
        print(f(chunksize), "image tiles modified")

    def modify_if_should(self, path: Path) -> bool:
        """Test if a chip should be modified, and if so, modify it and the label."""
        should_modify = self.should_be_modified(path)
        if should_modify:
            self.modify(path)
        return should_modify

    def mp_modify_if_should(self, paths: List[Path], chunksize: int) -> int:
        """Execute modify if should using multiprocessing."""
        modified = concurrent.process_map(self.modify_if_should, paths, chunksize=chunksize)
        return sum(modified)

    @abstractmethod
    def modify(self, path: Path) -> None:
        """Modify an image chip and a label chip."""
        raise NotImplementedError

    @abstractmethod
    def should_be_modified(self, path: Path) -> bool:
        """Determine if an image and or label chip should be modified."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, chunksize: int) -> int:
        """Should call mp_modify_if_should in subclasses with appropriate file list."""
        raise NotImplementedError


class ReflectExpandChips(Modifier):
    """Modifies incorrectly shaped tiles by reflection padding them."""

    def __init__(self, dataset, width_required, height_required):
        super().__init__(dataset)
        self.required_width = width_required
        self.required_height = height_required

    def __call__(self, chunksize: int = 100) -> int:
        """Should call mp_modify_if_should in subclasses with appropriate file list."""
        return self.mp_modify_if_should(self.imgs, chunksize)

    def should_be_modified(self, path: Path) -> bool:
        """
        Returns True if the image location path is incorrectly sized and should be modified.

        Parameters
        ----------
        path: Union[Path, str]
            The path the image to test if correct size.

        Returns
        -------
            bool: A flag indicating if the image at path is incorrectly sized.
        """
        img = Image.open(str(path))
        h, w, _ = np.asarray(img).shape
        return w < self.required_width or h < self.required_height

    def modify(self, path: Path) -> None:
        """Modify an image chip and a label chip by expanding the dims to the required shape."""
        label_path = self.labels_dir.joinpath(path.name)
        label_path = label_path.with_name(f"label_{label_path.name}")

        img = np.asarray(Image.open(path))
        label = np.asarray(Image.open(label_path))

        h_pad = self.required_height - img.shape[0]
        w_pad = self.required_width - img.shape[1]

        img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect')
        label = np.pad(label, ((0, h_pad), (0, w_pad)), mode='reflect')

        Image.fromarray(img).save(path)
        Image.fromarray(label).save(path)

    @classmethod
    def process(cls, dataset: str, size: int = 512, chunksize: int = 100) -> None:
        """
        Create a modifier class, call the modifying function, and print the number of tiles that get modified.

        Parameters
        ----------
        dataset: str
            Path to the dataset to process.

        size: int
            The required dimensions for the dataset

        chunksize: int
            The length of files to pass to each multi-processing processing.

        Returns
        -------
            None
        """
        f = cls(dataset, size, size)
        print(f(chunksize), "image tiles modified")


class StripExtraChannels(Modifier):
    """Modifies imgs with more than RGB as channels and removes them."""

    def __call__(self, chunksize: int = 100) -> int:
        """Should call mp_modify_if_should in subclasses with appropriate file list."""
        return self.mp_modify_if_should(self.imgs, chunksize)

    def should_be_modified(self, path: Path) -> bool:
        """Returns True if image has more than 3 channels."""
        img = Image.open(str(path))
        return np.asarray(img).shape[2] > 3

    def modify(self, path: Path) -> None:
        """Modify an image chip and a label chip by stripping extra channels."""
        img = np.asarray(Image.open(path))[:, :, :3]
        Image.fromarray(img).save(path)


if __name__ == '__main__':
    fire.Fire({
        "expand_chips": ReflectExpandChips.process,
        "strip_extra_channels": StripExtraChannels.process,
    })
