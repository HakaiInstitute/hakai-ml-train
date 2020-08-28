from pathlib import Path

import fire
import torch
from loguru import logger

from models.deeplabv3 import DeepLabv3


def predict(seg_in, seg_out, weights, batch_size=4, crop_size=256, crop_pad=128):
    """
    Segment the FG class using DeepLabv3 Segmentation model of an input .tif image.
    Args:
        weights: Path to a model weights file (*.pt). Required for eval and pred mode.
        seg_in: Path to a *.tif image to do segmentation on in pred mode.
        seg_out: Path to desired output *.tif created by the model in pred mode.
        batch_size: The batch size per GPU. Defaults to 4.
        crop_size: The crop size in pixels for processing the image. Defines the length and width of the individual
            sections the input .tif image is cropped to for processing. Defaults to 300.
        crop_pad: The amount of padding added for classification context to each image crop. The output classification
            on this crop area is not output by the model but will influence the classification of the area in the
            (crop_size x crop_size) window. Defaults to 150.

    Note: Changing the crop_size and crop_pad parameters may require adjusting the batch_size such that the image
        batches fit into the GPU memory.

    Returns: None. Outputs various files as a side effect depending on the script mode.
    """
    seg_in, seg_out = Path(seg_in), Path(seg_out)
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logger.warning("Could not find GPU device")

    # Load model and weights
    model = DeepLabv3.load_from_checkpoint(weights, batch_size=batch_size, crop_size=crop_size, padding=crop_pad)
    model.freeze()
    model = model.to(device)

    # Do the segmentation
    model.predict_geotiff(seg_in, seg_out)


if __name__ == '__main__':
    fire.Fire(predict)
