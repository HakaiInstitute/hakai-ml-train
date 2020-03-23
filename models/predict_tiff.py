import torch
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
import itertools
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial


def _batcher(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    args = [iter(iterable)] * n
    return zip(*args)


def predict_tiff(model, img_path, dest_dir, device, transform, crop_size=200, batch_size=4):
    """
    Predict segmentation classes on a geotiff image.
    Args:
        model: A PyTorch Model class
        img_path: Path to the image to classify
        dest_dir: Location to save the new image
        device: A torch device
        crop_size: Size of patches to predict on before stitching them back together
        batch_size: The size of mini images to process at one time. Must be greater than 1.

    Returns: str Path to classified GeoTiff segmentation
    """
    model.eval()
    dest_dir = Path(dest_dir)
    with rasterio.open(img_path) as in_data:
        with rasterio.open(dest_dir, 'w',
                           driver='GTiff',
                           height=in_data.height,
                           width=in_data.width,
                           count=1,
                           dtype='uint8',
                           crs=in_data.crs,
                           transform=in_data.transform
                           ) as out_data:
            all_x0s = range(0, in_data.width, crop_size)
            all_y0s = range(0, in_data.height, crop_size)

            for i, x0y0_batch in enumerate(tqdm(_batcher(itertools.product(all_x0s, all_y0s), batch_size),
                                                total=len(all_x0s)*len(all_y0s))):
                # Expand dims for batches of size 1
                if len(np.array(x0y0_batch).shape) == 3:
                    x0y0_batch = [x0y0_batch]

                subsets = []
                for x0, y0 in x0y0_batch:
                    subsets.append(in_data.read([1, 2, 3], window=((y0, y0 + crop_size), (x0, x0 + crop_size))))

                # Pad out sections that aren't size crop_size * crop_size
                subset_shapes = [s.shape[1:] for s in subsets]  # For un-padding later
                print(subset_shapes)
                for i, s in enumerate(subsets):
                    if s.shape[1] != crop_size or s.shape[2] != crop_size:
                        y_pad = crop_size - s.shape[1]
                        x_pad = crop_size - s.shape[2]
                        subsets[i] = np.pad(s, ((0,0), (0,y_pad), (0,x_pad)), mode='constant', constant_values=0)

                subsets = [np.moveaxis(s, 0, 2) for s in subsets]  # (b, c, h, w) -> (b, h, w, c)

                # Numpy to PIL image
                subset_imgs = [Image.fromarray(s, mode="RGB") for s in subsets]

                # Apply transform to Tensor
                subset_imgs = [transform(s) for s in subset_imgs]
                subset_imgs = torch.stack(subset_imgs, dim=0).to(device)

                # Do segmentation
                segmentation = model(subset_imgs)['out']
                # torch.cuda.synchronize() # For profiling only
                segmentation = segmentation.detach().cpu().numpy()
                segmentation = np.argmax(segmentation, axis=1)
                segmentation = np.expand_dims(segmentation, axis=1)
                # Unpad results
                segmentation = [s[:,0:crop[0], 0:crop[1]] for s, crop in zip(segmentation, subset_shapes)]
                print([s.shape for s in segmentation])

                # Save part of tiff wither rasterio
                for seg in segmentation:
                    out_data.write(seg.astype(np.uint8), window=((y0, y0+seg.shape[1]), (x0, x0+seg.shape[2])))


if __name__ == '__main__':
    import sys
    sys.path.append("../")

    from models import deeplabv3
    from utils.dataset import transforms

    num_classes = 2
    batch_size = 4

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tiff = "../data/RPAS/NW_Calvert_2015/calvert_choked15_CSRS_mos_U0015.tif"

    model = deeplabv3.create_model(num_classes)
    checkpoint = torch.load("../checkpoints/deeplabv3/checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    transform = transforms.test_transforms

    predict_tiff(model, tiff, "./test_out.tiff", device, transform, batch_size=8)
