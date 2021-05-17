# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-22
# Description: Utilities for saving and loading checkpoint files

from pathlib import Path
from typing import Optional


def get_checkpoint(checkpoint_dir: str, name: str = "") -> Optional[str]:
    versions_dir = Path(checkpoint_dir).joinpath(name)
    checkpoint_files = list(versions_dir.glob("*/checkpoints/*.ckpt"))
    if len(checkpoint_files) > 0:
        return str(sorted(checkpoint_files)[-1])
    else:
        return None
