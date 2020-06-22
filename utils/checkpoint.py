# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-22
# Description: Utilities for saving and loading checkpoint files

from pathlib import Path


def get_checkpoint(checkpoint_dir, name=""):
    versions_dir = Path(checkpoint_dir).joinpath(name)
    checkpoint_files = list(versions_dir.glob("*/checkpoints/best_val_miou_*.ckpt"))
    if len(checkpoint_files) > 0:
        return str(sorted(checkpoint_files)[-1])
    else:
        return False
