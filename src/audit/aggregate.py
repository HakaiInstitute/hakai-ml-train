from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def open_oof_probs_writer(
    path: Path,
    n_total: int,
    num_classes: int,
    height: int,
    width: int,
    chunk_size: int = 16,
) -> zarr.Array:
    """Open or create the OOF probs zarr array. Float16, chunked along axis 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open(
        str(path),
        mode="w",
        shape=(n_total, num_classes, height, width),
        chunks=(chunk_size, num_classes, height, width),
        dtype=np.float16,
    )


def aggregate_fold_probs(
    fold_assignments: pd.DataFrame,
    fold_dirs: dict[int, Path],
    output_zarr_path: Path,
    num_classes: int,
    height: int,
    width: int,
) -> None:
    """Concatenate per-fold probs.zarr into one OOF probs.zarr.

    Args:
        fold_assignments: DataFrame from compute_fold_assignments. Must have
            columns fold_idx and row_index.
        fold_dirs: mapping fold_idx -> directory containing probs.zarr.
        output_zarr_path: destination zarr path for the combined OOF tensor.
        num_classes, height, width: dimensions for the output array.
    """
    n_total = len(fold_assignments)
    out = open_oof_probs_writer(output_zarr_path, n_total, num_classes, height, width)

    for fold_idx, fold_dir in sorted(fold_dirs.items()):
        fold_df = fold_assignments[fold_assignments["fold_idx"] == fold_idx]
        # row_index is the destination index into the OOF tensor.
        # The fold's probs.zarr was written in fold-local order, which corresponds
        # to fold_df sorted by row_index.
        fold_df = fold_df.sort_values("row_index")
        target_rows = fold_df["row_index"].to_numpy()

        fold_probs = zarr.open(str(fold_dir / "probs.zarr"), mode="r")
        if fold_probs.shape[0] != len(target_rows):
            raise ValueError(
                f"Fold {fold_idx} probs.zarr has {fold_probs.shape[0]} samples, "
                f"expected {len(target_rows)}"
            )

        # Write in chunks to control memory.
        chunk = 16
        for start in range(0, len(target_rows), chunk):
            end = min(start + chunk, len(target_rows))
            out[target_rows[start:end]] = fold_probs[start:end]
