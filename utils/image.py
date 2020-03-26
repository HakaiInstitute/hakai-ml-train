import numpy as np


def pad_out(crop, crop_size):
    """Pads image crop, which is a numpy array of size (h, w, c), with zeros so that h and w equal crop_size."""
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        if len(crop.shape) == 2:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]))
        else:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]), (0, 0))

        return np.pad(crop, padding, mode='constant', constant_values=0)
    else:
        return crop