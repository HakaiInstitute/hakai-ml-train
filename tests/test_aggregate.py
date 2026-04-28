import numpy as np
import pytest
import zarr

from src.audit.aggregate import aggregate_fold_probs
from src.audit.folds import compute_fold_assignments


def _write_synthetic_fold_zarrs(tmp_path, fold_assignments, num_classes, h, w):
    """Write per-fold probs zarrs that encode each sample's row_index in pixel
    (0,0) of class 0, so we can assert correct row placement after aggregation."""
    fold_dirs = {}
    for fold_idx in sorted(fold_assignments["fold_idx"].unique()):
        fold_df = fold_assignments[fold_assignments["fold_idx"] == fold_idx]
        ordered = fold_df.iloc[np.argsort(fold_df["row_index"].to_numpy())]
        fd = tmp_path / f"fold_{fold_idx}"
        fd.mkdir()
        n = len(ordered)
        arr = zarr.open(
            str(fd / "probs.zarr"),
            mode="w",
            shape=(n, num_classes, h, w),
            chunks=(min(4, n), num_classes, h, w),
            dtype=np.float16,
        )
        for i, ri in enumerate(ordered["row_index"].to_numpy()):
            arr[i, 0, 0, 0] = float(ri)
        fold_dirs[int(fold_idx)] = fd
    return fold_dirs


def test_aggregate_preserves_row_index_ordering(tmp_path):
    df = compute_fold_assignments({"train": 20}, num_folds=5, seed=42)
    fold_dirs = _write_synthetic_fold_zarrs(tmp_path, df, num_classes=2, h=4, w=4)

    out_path = tmp_path / "oof_probs.zarr"
    aggregate_fold_probs(df, fold_dirs, out_path, num_classes=2, height=4, width=4)

    oof = zarr.open(str(out_path), mode="r")
    assert oof.shape == (20, 2, 4, 4)
    for ri in range(20):
        assert float(oof[ri, 0, 0, 0]) == float(ri)


def test_aggregate_raises_on_size_mismatch(tmp_path):
    df = compute_fold_assignments({"train": 20}, num_folds=5, seed=42)
    fold_dirs = _write_synthetic_fold_zarrs(tmp_path, df, num_classes=2, h=4, w=4)

    # Sabotage fold 2: replace its probs.zarr with one that has the wrong sample count.
    bad_dir = fold_dirs[2]
    expected_n = (df["fold_idx"] == 2).sum()
    zarr.open(
        str(bad_dir / "probs.zarr"),
        mode="w",
        shape=(expected_n - 1, 2, 4, 4),
        chunks=(2, 2, 4, 4),
        dtype=np.float16,
    )

    with pytest.raises(ValueError, match="Fold 2 probs.zarr"):
        aggregate_fold_probs(
            df,
            fold_dirs,
            tmp_path / "oof_probs.zarr",
            num_classes=2,
            height=4,
            width=4,
        )


def test_aggregate_handles_chunk_boundaries(tmp_path):
    # 23 samples into 5 folds = chunks of (5, 5, 5, 4, 4); 23 is not divisible
    # by chunk_size=4, so the last write block in some folds is partial.
    df = compute_fold_assignments({"train": 23}, num_folds=5, seed=42)
    fold_dirs = _write_synthetic_fold_zarrs(tmp_path, df, num_classes=3, h=2, w=2)

    out_path = tmp_path / "oof_probs.zarr"
    aggregate_fold_probs(
        df,
        fold_dirs,
        out_path,
        num_classes=3,
        height=2,
        width=2,
        chunk_size=4,
    )

    oof = zarr.open(str(out_path), mode="r")
    assert oof.shape == (23, 3, 2, 2)
    # Every row was written exactly once.
    for ri in range(23):
        assert float(oof[ri, 0, 0, 0]) == float(ri)


def test_aggregate_no_overlap_or_dropouts(tmp_path):
    """Stronger guarantee: every position in the OOF tensor is set, no gaps."""
    df = compute_fold_assignments({"train": 17, "validation": 8}, num_folds=4, seed=7)
    fold_dirs = _write_synthetic_fold_zarrs(tmp_path, df, num_classes=2, h=2, w=2)

    out_path = tmp_path / "oof_probs.zarr"
    aggregate_fold_probs(df, fold_dirs, out_path, num_classes=2, height=2, width=2)

    oof = zarr.open(str(out_path), mode="r")
    written = oof[:, 0, 0, 0]  # shape (25,)
    expected = np.arange(25, dtype=np.float16)
    np.testing.assert_array_equal(written, expected)
