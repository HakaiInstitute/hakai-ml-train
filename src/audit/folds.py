from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_fold_assignments(
    split_sizes: dict[str, int],
    num_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Compute deterministic K-fold assignments over a pool of HF dataset splits.

    Args:
        split_sizes: ordered mapping of split name -> number of samples.
        num_folds: K.
        seed: RNG seed; same seed => same assignments.

    Returns:
        DataFrame with columns: sample_id, original_split, original_index,
        fold_idx, row_index. row_index is the position in the permuted order
        and is the integer axis for downstream zarr arrays.
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    rows = []
    for split_name, size in split_sizes.items():
        for i in range(size):
            rows.append(
                {
                    "sample_id": f"{split_name}/{i}",
                    "original_split": split_name,
                    "original_index": i,
                }
            )
    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        raise ValueError("split_sizes must contain at least one non-empty split")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n)
    df = df.iloc[permuted].reset_index(drop=True)
    df["row_index"] = np.arange(n, dtype=np.int64)

    # numpy.array_split handles uneven division: first (n % K) chunks get one extra.
    fold_indices = np.zeros(n, dtype=np.int64)
    for k, chunk in enumerate(np.array_split(np.arange(n), num_folds)):
        fold_indices[chunk] = k
    df["fold_idx"] = fold_indices

    return df[
        ["sample_id", "original_split", "original_index", "fold_idx", "row_index"]
    ]


def save_fold_assignments(df: pd.DataFrame, path: Path) -> None:
    """Write fold assignments to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_fold_assignments(path: Path) -> pd.DataFrame:
    """Read fold assignments from parquet."""
    return pd.read_parquet(path)
