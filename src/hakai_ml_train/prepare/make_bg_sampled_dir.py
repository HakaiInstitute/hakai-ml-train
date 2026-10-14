"""Build a training directory of every labelled tile plus a sample of background-only tiles.

`remove_bg_only_tiles` drops *all* background-only tiles, which leaves the model
never seeing open water, land or sun glint without the target class. When
validation runs on the full, unfiltered distribution, the resulting false
positives on background dominate IoU: with a target class at ~3% of pixels, a 1%
false-positive rate on background costs roughly 17 IoU points.

This script keeps every tile containing a positive label and adds a fixed-size
random sample of the background-only tiles, so background is represented without
paying for the full (typically 5-20x larger) tile set. Tiles are hardlinked, so
the new directory costs no extra disk space.
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _has_positive_label(path: Path) -> tuple[Path, bool]:
    """True if the tile contains any label > 0 (i.e. not background/ignore only)."""
    with np.load(path) as data:
        return path, bool((data["label"] > 0).any())


def link(src: Path, dst: Path) -> None:
    """Hardlink src to dst, falling back to a symlink across filesystems."""
    try:
        dst.hardlink_to(src)
    except OSError:
        dst.symlink_to(src)


def make_bg_sampled_dir(
    input_dir: Path,
    output_dir: Path,
    num_bg: int,
    seed: int = 42,
    workers: int = 8,
) -> None:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz tiles found in {input_dir}")

    labelled, background = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, positive in tqdm(
            pool.map(_has_positive_label, files, chunksize=64),
            total=len(files),
            desc="Scanning labels",
        ):
            (labelled if positive else background).append(path)

    if num_bg > len(background):
        print(
            f"Only {len(background)} background-only tiles available; "
            f"requested {num_bg}. Using all of them."
        )
    sampled_bg = random.Random(seed).sample(background, min(num_bg, len(background)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(labelled + sampled_bg, desc="Linking"):
        dst = output_dir / path.name
        if not dst.exists():
            link(path.resolve(), dst)

    print(
        f"{len(files)} tiles scanned: {len(labelled)} labelled, "
        f"{len(background)} background-only.\n"
        f"Linked {len(labelled)} labelled + {len(sampled_bg)} sampled background "
        f"= {len(labelled) + len(sampled_bg)} tiles into {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory of all .npz tiles.")
    parser.add_argument("output_dir", type=Path, help="Directory to link tiles into.")
    parser.add_argument(
        "--num-bg",
        type=int,
        default=5000,
        help="How many background-only tiles to sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--workers", type=int, default=8, help="Label-scan workers.")
    args = parser.parse_args()

    make_bg_sampled_dir(
        args.input_dir, args.output_dir, args.num_bg, args.seed, args.workers
    )


if __name__ == "__main__":
    main()
