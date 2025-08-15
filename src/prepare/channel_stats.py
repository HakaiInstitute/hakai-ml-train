from pathlib import Path

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

from src.data import NpzSegmentationDataset


def compute_channel_stats_fast_stable(dataloader):
    """
    Fast and numerically stable algorithm that processes entire batches.
    Uses incremental updates to avoid overflow while being vectorized.

    Args:
        dataloader: PyTorch DataLoader
    """
    n_channels = None
    total_pixels = 0
    mean = None
    var_sum = None  # Running sum for variance calculation

    for data, _ in tqdm(dataloader, desc="Computing dataset statistics"):
        # data shape: (batch, channels, height, width)
        if n_channels is None:
            n_channels = data.size(1)
            mean = torch.zeros(n_channels, dtype=torch.float32)
            var_sum = torch.zeros(n_channels, dtype=torch.float32)

        # Flatten spatial dimensions
        pixels = data.transpose(0, 1).reshape(n_channels, -1).to(torch.float32)

        # Create mask for non-black pixels
        # A pixel is considered black if ALL channels are <= threshold
        mask = (pixels > 0).any(dim=0)  # Shape: (batch * height * width,)

        if not mask.any():
            # Skip this batch if all pixels are black
            continue

        # Filter out black pixels
        valid_pixels = pixels[:, mask]  # Shape: (channels, num_valid_pixels)
        batch_pixels = mask.sum().item()

        if batch_pixels == 0:
            continue

        # Compute batch statistics on valid pixels only
        batch_mean = valid_pixels.mean(dim=1)

        # Update mean incrementally
        new_total = total_pixels + batch_pixels
        delta = batch_mean - mean
        mean = mean + delta * batch_pixels / new_total

        # Update variance sum using parallel algorithm
        batch_var = valid_pixels.var(dim=1, unbiased=False)
        var_sum = (
            var_sum
            + batch_var * batch_pixels
            + delta**2 * total_pixels * batch_pixels / new_total
        )

        total_pixels = new_total

    if total_pixels == 0:
        raise ValueError("No valid (non-black) pixels found in the dataset!")

    # Calculate standard deviation
    variance = (
        var_sum / (total_pixels - 1) if total_pixels > 1 else torch.zeros_like(var_sum)
    )
    std = torch.sqrt(variance)

    return mean, std, total_pixels


def main(root: Path, max_pixel_val: float = 255.0) -> None:
    dataset = NpzSegmentationDataset(str(root), transforms=ToTensorV2())

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=8
    )

    print(f"Computing channel statistics...")
    mean, std, valid_pixels = compute_channel_stats_fast_stable(dataloader)
    mean /= max_pixel_val
    std /= max_pixel_val
    print(f"\nValid pixels used: {valid_pixels:,}")

    print(f"\nChannel-wise statistics:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    # Save statistics for later use
    stats = {
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
    }

    stats_path = root.parent / "channel_stats.npz"
    np.savez(stats_path, **stats)
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--max_pixel_val", type=float, default=255.0)
    args = parser.parse_args()

    main(Path(args.root, max_pixel_val=args.max_pixel_val))
