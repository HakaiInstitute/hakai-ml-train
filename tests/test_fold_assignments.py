import numpy as np
import pandas as pd
import pytest

from src.audit.folds import compute_fold_assignments


def _split_sizes():
    return {"train": 100, "validation": 25}


def test_compute_fold_assignments_deterministic_with_same_seed():
    a = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    b = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_compute_fold_assignments_different_with_different_seed():
    a = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    b = compute_fold_assignments(_split_sizes(), num_folds=5, seed=43)
    assert not a.equals(b)


def test_compute_fold_assignments_covers_every_sample_exactly_once():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    assert len(df) == 125
    assert df["sample_id"].nunique() == 125
    # Each (original_split, original_index) appears exactly once.
    assert df.groupby(["original_split", "original_index"]).size().max() == 1


def test_compute_fold_assignments_no_overlap_across_folds():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    by_fold = df.groupby("fold_idx")["sample_id"].apply(set)
    for i in range(5):
        for j in range(i + 1, 5):
            assert by_fold.iloc[i].isdisjoint(by_fold.iloc[j])


def test_compute_fold_assignments_balanced_chunks():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    counts = df["fold_idx"].value_counts().sort_index().tolist()
    # 125 / 5 = 25 each
    assert counts == [25, 25, 25, 25, 25]


def test_compute_fold_assignments_uneven_division():
    # 7 samples into 5 folds — first two folds get one extra
    df = compute_fold_assignments({"train": 7}, num_folds=5, seed=42)
    counts = df["fold_idx"].value_counts().sort_index().tolist()
    assert sum(counts) == 7
    assert max(counts) - min(counts) <= 1


def test_sample_id_format():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    expected = {f"train/{i}" for i in range(100)} | {
        f"validation/{i}" for i in range(25)
    }
    assert set(df["sample_id"]) == expected


def test_row_index_is_unique_and_complete():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    assert sorted(df["row_index"].tolist()) == list(range(125))


def test_fold_assignments_parquet_round_trip(tmp_path):
    from src.audit.folds import load_fold_assignments, save_fold_assignments

    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    path = tmp_path / "fold_assignments.parquet"
    save_fold_assignments(df, path)
    loaded = load_fold_assignments(path)
    pd.testing.assert_frame_equal(df, loaded)
