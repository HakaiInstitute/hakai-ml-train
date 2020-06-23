import os
from pathlib import Path

import torch

from kelp_species.model import KelpSpeciesModel
from utils.dataset.transforms import transforms as t
from utils.eval import predict_tiff


def predict(seg_in, seg_out, weights, batch_size=4, crop_size=300, crop_pad=150):
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
    seg_in, seg_out, weights = Path(seg_in), Path(seg_out), Path(weights)
    os.environ['TORCH_HOME'] = str(weights.parent)

    # Create output directory as required
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    # Load model and weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KelpSpeciesModel.load_from_checkpoint(str(weights))
    model.freeze()
    model = model.to(device)

    print("Processing:", seg_in)
    batch_size = batch_size * max(torch.cuda.device_count(), 1)
    predict_tiff(model, device, seg_in, seg_out,
                 transform=t.test_transforms,
                 crop_size=crop_size, pad=crop_pad,
                 batch_size=batch_size)
