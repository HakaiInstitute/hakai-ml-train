# K-Fold Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a K=5 cross-validation orchestrator that trains 5 instances of an existing segmentation config on disjoint folds of the train+val pool, generates out-of-fold (OOF) softmax probabilities for every sample, and emits per-pixel + per-tile label-quality artifacts so the user can review likely-mislabeled tiles.

**Architecture:** Three layers. (1) A small `src/audit/` package with pure-logic modules for fold assignments, per-pixel scoring, and per-fold zarr aggregation. (2) Surgical extensions to existing code: `WebDataModule` gains optional fold parameters, `SMPMulticlassSegmentationModel` gains a `predict_step`. (3) An orchestrator script `scripts/kfold_label_audit.py` that loads a base YAML config, loops folds 0..4, and writes a structured `audits/<run-name>/` output tree with `_SUCCESS` markers for resumability.

**Tech Stack:** PyTorch Lightning 2.x, Hugging Face `datasets`, zarr (new dep), pyarrow (new dep), pytest (new dev dep), pandas (transitive).

**Spec:** `docs/superpowers/specs/2026-04-27-kfold-label-audit-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/audit/__init__.py` | Marker file for the new audit package. |
| `src/audit/folds.py` | Compute deterministic fold assignments; parquet I/O. Pure functions; no torch. |
| `src/audit/scoring.py` | Per-pixel score formula + per-tile aggregation. Pure numpy; no torch. |
| `src/audit/aggregate.py` | Concatenate per-fold `probs.zarr` into one OOF zarr. |
| `src/data.py` | Modify `WebDataModule` to accept fold params. Add small helper for pool sample IDs. |
| `src/models/smp.py` | Add `predict_step` to `SMPMulticlassSegmentationModel`. |
| `scripts/kfold_label_audit.py` | Orchestrator entry point. |
| `tests/test_fold_assignments.py` | Unit tests for `src/audit/folds.py`. |
| `tests/test_label_audit_scoring.py` | Unit tests for `src/audit/scoring.py`. |
| `tests/conftest.py` | pytest config (project root on sys.path). |
| `pyproject.toml` | Add `zarr`, `pyarrow` to deps; `pytest` to dev. |

---

## Task 1: Project setup — deps, package skeleton, test infra

**Files:**
- Modify: `pyproject.toml`
- Create: `src/audit/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add runtime + dev deps via uv**

Run:
```bash
uv add zarr pyarrow
uv add --group dev pytest
```

Expected: `pyproject.toml` updated, `uv.lock` updated, packages installed into `.venv`.

- [ ] **Step 2: Verify deps importable**

Run:
```bash
uv run python -c "import zarr, pyarrow, pytest; print(zarr.__version__, pyarrow.__version__, pytest.__version__)"
```

Expected: three version strings printed, no error.

- [ ] **Step 3: Create empty audit package**

Create `src/audit/__init__.py` with content:

```python
"""K-fold label audit subpackage. See docs/superpowers/specs/2026-04-27-kfold-label-audit-design.md."""
```

- [ ] **Step 4: Create test infra**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py` with content:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 5: Smoke-test pytest**

Run:
```bash
uv run pytest tests/ -v
```

Expected: `no tests ran`, exit code 0 (or 5 — both are acceptable; we just want the harness to load).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/audit/__init__.py tests/__init__.py tests/conftest.py
git commit -m "Add audit package scaffolding and zarr/pyarrow/pytest deps"
```

---

## Task 2: Fold assignments — pure logic, TDD

**Files:**
- Create: `src/audit/folds.py`
- Create: `tests/test_fold_assignments.py`

The fold assignment is a deterministic permutation of `[0..N)` sliced into K equal contiguous chunks. The result is a dataframe with columns `sample_id`, `original_split`, `original_index`, `fold_idx`, `row_index`. `row_index` is the position in the permuted order — used as the integer axis for zarr arrays. `sample_id` is `f"{original_split}/{original_index}"`.

- [ ] **Step 1: Write failing test for determinism + coverage + no-overlap**

Create `tests/test_fold_assignments.py`:

```python
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
    expected = {f"train/{i}" for i in range(100)} | {f"validation/{i}" for i in range(25)}
    assert set(df["sample_id"]) == expected


def test_row_index_is_unique_and_complete():
    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    assert sorted(df["row_index"].tolist()) == list(range(125))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_fold_assignments.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.audit.folds'` or similar.

- [ ] **Step 3: Implement `compute_fold_assignments`**

Create `src/audit/folds.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_fold_assignments.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Add parquet round-trip test**

Append to `tests/test_fold_assignments.py`:

```python
def test_fold_assignments_parquet_round_trip(tmp_path):
    from src.audit.folds import save_fold_assignments, load_fold_assignments

    df = compute_fold_assignments(_split_sizes(), num_folds=5, seed=42)
    path = tmp_path / "fold_assignments.parquet"
    save_fold_assignments(df, path)
    loaded = load_fold_assignments(path)
    pd.testing.assert_frame_equal(df, loaded)
```

- [ ] **Step 6: Run new test**

Run:
```bash
uv run pytest tests/test_fold_assignments.py::test_fold_assignments_parquet_round_trip -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/audit/folds.py tests/test_fold_assignments.py
git commit -m "Add deterministic K-fold assignment computation with parquet IO"
```

---

## Task 3: Per-pixel + per-tile scoring — pure logic, TDD

**Files:**
- Create: `src/audit/scoring.py`
- Create: `tests/test_label_audit_scoring.py`

The scoring formula is `score = 1 - prob[true_label]` per pixel, with `NaN` where `label == ignore_index`. Per-tile aggregates are `mean_score`, `conf_disagree_pct` (fraction of valid pixels where `argmax(probs) != label AND max(probs) > confidence_threshold`), and `dominant_pred_class` (mode of predicted class on confidently-disagreeing pixels, or -1 if none).

- [ ] **Step 1: Write failing tests with hand-computed expected values**

Create `tests/test_label_audit_scoring.py`:

```python
import numpy as np
import pytest

from src.audit.scoring import per_pixel_scores, per_tile_metrics


# --- Helpers -------------------------------------------------------------

def _simple_probs():
    # Shape (1, 3, 2, 2): one tile, 3 classes, 2x2 pixels
    # Pixel (0,0): probs [0.9, 0.05, 0.05]   argmax=0
    # Pixel (0,1): probs [0.1, 0.85, 0.05]   argmax=1
    # Pixel (1,0): probs [0.4, 0.4, 0.2]     argmax=0 (tie broken by argmax)
    # Pixel (1,1): probs [0.05, 0.05, 0.9]   argmax=2
    return np.array([
        [[[0.9, 0.1], [0.4, 0.05]],
         [[0.05, 0.85], [0.4, 0.05]],
         [[0.05, 0.05], [0.2, 0.9]]],
    ], dtype=np.float32)


# --- per_pixel_scores ----------------------------------------------------

def test_per_pixel_score_correct_label_low_score():
    probs = _simple_probs()
    labels = np.array([[[0, 1], [0, 2]]], dtype=np.int64)
    scores = per_pixel_scores(probs, labels, ignore_index=-100)
    expected = np.array([[[1 - 0.9, 1 - 0.85], [1 - 0.4, 1 - 0.9]]], dtype=np.float32)
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_per_pixel_score_wrong_label_high_score():
    probs = _simple_probs()
    # Pixel (0,0) labeled as class 2 — model says class 0 with 0.9 confidence
    labels = np.array([[[2, 1], [0, 2]]], dtype=np.int64)
    scores = per_pixel_scores(probs, labels, ignore_index=-100)
    assert scores[0, 0, 0] == pytest.approx(1 - 0.05, abs=1e-6)


def test_per_pixel_score_ignore_index_becomes_nan():
    probs = _simple_probs()
    labels = np.array([[[-100, 1], [0, 2]]], dtype=np.int64)
    scores = per_pixel_scores(probs, labels, ignore_index=-100)
    assert np.isnan(scores[0, 0, 0])
    assert not np.isnan(scores[0, 0, 1])


# --- per_tile_metrics ----------------------------------------------------

def test_per_tile_metrics_mean_score_excludes_nan():
    probs = _simple_probs()
    labels = np.array([[[-100, 1], [0, 2]]], dtype=np.int64)
    metrics = per_tile_metrics(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    # Valid pixels: (0,1) score=0.15, (1,0) score=0.6, (1,1) score=0.1
    expected_mean = (0.15 + 0.6 + 0.1) / 3
    assert metrics["mean_score"] == pytest.approx(expected_mean, abs=1e-6)


def test_per_tile_metrics_conf_disagree_pct():
    probs = _simple_probs()
    # Label everything wrong except where model is confident.
    # Pixel (0,0): label=2, argmax=0, max_prob=0.9 -> confident disagreement
    # Pixel (0,1): label=2, argmax=1, max_prob=0.85 -> NOT confident (< 0.9)
    # Pixel (1,0): label=2, argmax=0, max_prob=0.4 -> NOT confident
    # Pixel (1,1): label=0, argmax=2, max_prob=0.9 -> confident disagreement
    labels = np.array([[[2, 2], [2, 0]]], dtype=np.int64)
    metrics = per_tile_metrics(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    # 2 of 4 valid pixels confidently disagree
    assert metrics["conf_disagree_pct"] == pytest.approx(0.5, abs=1e-6)


def test_per_tile_metrics_dominant_pred_class():
    probs = _simple_probs()
    # Same labels as above; confidently-disagreeing pixels are (0,0)->pred 0 and (1,1)->pred 2
    # Tied — dominant is the smaller class index from np.bincount + argmax
    labels = np.array([[[2, 2], [2, 0]]], dtype=np.int64)
    metrics = per_tile_metrics(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    assert metrics["dominant_pred_class"] in (0, 2)


def test_per_tile_metrics_dominant_pred_class_no_disagreement():
    probs = _simple_probs()
    # All labels match argmax with high confidence
    labels = np.array([[[0, 1], [0, 2]]], dtype=np.int64)
    metrics = per_tile_metrics(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    assert metrics["dominant_pred_class"] == -1


def test_per_tile_metrics_all_ignored_pixels():
    probs = _simple_probs()
    labels = np.full((1, 2, 2), -100, dtype=np.int64)
    metrics = per_tile_metrics(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    assert np.isnan(metrics["mean_score"])
    assert metrics["conf_disagree_pct"] == 0.0
    assert metrics["dominant_pred_class"] == -1


def test_per_tile_metrics_batch_dim():
    probs = np.concatenate([_simple_probs(), _simple_probs()], axis=0)
    labels_a = np.array([[0, 1], [0, 2]], dtype=np.int64)
    labels_b = np.array([[2, 2], [2, 0]], dtype=np.int64)
    labels = np.stack([labels_a, labels_b], axis=0)
    from src.audit.scoring import per_tile_metrics_batch
    df = per_tile_metrics_batch(probs, labels, ignore_index=-100, confidence_threshold=0.9)
    assert len(df) == 2
    assert df.iloc[0]["conf_disagree_pct"] == pytest.approx(0.0, abs=1e-6)
    assert df.iloc[1]["conf_disagree_pct"] == pytest.approx(0.5, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest tests/test_label_audit_scoring.py -v
```

Expected: ImportError or ModuleNotFoundError.

- [ ] **Step 3: Implement scoring functions**

Create `src/audit/scoring.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def per_pixel_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    ignore_index: int,
) -> np.ndarray:
    """Per-pixel label-quality score = 1 - prob[true_label].

    Args:
        probs: shape (B, C, H, W), float. Softmax probabilities.
        labels: shape (B, H, W), int. Ground-truth class indices.
        ignore_index: label value to mask (score becomes NaN).

    Returns:
        scores: shape (B, H, W), float32. NaN where label == ignore_index.
    """
    b, c, h, w = probs.shape
    if labels.shape != (b, h, w):
        raise ValueError(
            f"labels shape {labels.shape} does not match probs spatial shape ({b}, {h}, {w})"
        )

    # Replace ignore_index with 0 for safe gather, then mask afterwards.
    safe_labels = np.where(labels == ignore_index, 0, labels).astype(np.int64)
    # Gather prob[true_label] per pixel: result shape (B, H, W)
    gathered = np.take_along_axis(probs, safe_labels[:, None, :, :], axis=1).squeeze(1)
    scores = (1.0 - gathered).astype(np.float32)
    scores[labels == ignore_index] = np.nan
    return scores


def per_tile_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    ignore_index: int,
    confidence_threshold: float,
) -> dict:
    """Per-tile aggregate metrics. Accepts either (1, C, H, W) + (1, H, W) or (C, H, W) + (H, W).

    Returns a dict with: mean_score (float, NaN if all pixels ignored),
    conf_disagree_pct (float, [0, 1]), dominant_pred_class (int, -1 if no
    confidently-disagreeing pixels).
    """
    if probs.ndim == 3:
        probs = probs[None]
        labels = labels[None]
    if probs.shape[0] != 1:
        raise ValueError("per_tile_metrics expects a single tile; use per_tile_metrics_batch")

    scores = per_pixel_scores(probs, labels, ignore_index)
    valid_mask = labels[0] != ignore_index

    if valid_mask.sum() == 0:
        return {
            "mean_score": float("nan"),
            "conf_disagree_pct": 0.0,
            "dominant_pred_class": -1,
        }

    mean_score = float(np.nanmean(scores))

    pred = probs[0].argmax(axis=0)  # (H, W)
    max_prob = probs[0].max(axis=0)  # (H, W)
    confident = max_prob > confidence_threshold
    disagree = pred != labels[0]
    conf_disagree = valid_mask & confident & disagree

    n_valid = int(valid_mask.sum())
    conf_disagree_pct = float(conf_disagree.sum()) / n_valid

    if conf_disagree.sum() == 0:
        dominant_pred_class = -1
    else:
        bincounts = np.bincount(pred[conf_disagree], minlength=probs.shape[1])
        dominant_pred_class = int(bincounts.argmax())

    return {
        "mean_score": mean_score,
        "conf_disagree_pct": conf_disagree_pct,
        "dominant_pred_class": dominant_pred_class,
    }


def per_tile_metrics_batch(
    probs: np.ndarray,
    labels: np.ndarray,
    ignore_index: int,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Apply per_tile_metrics across a batch. Returns a DataFrame with one row per tile."""
    rows = [
        per_tile_metrics(probs[i : i + 1], labels[i : i + 1], ignore_index, confidence_threshold)
        for i in range(probs.shape[0])
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/test_label_audit_scoring.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/audit/scoring.py tests/test_label_audit_scoring.py
git commit -m "Add per-pixel and per-tile label-quality scoring"
```

---

## Task 4: Per-fold prediction aggregation

**Files:**
- Create: `src/audit/aggregate.py`

The aggregator concatenates `fold_k/probs.zarr` arrays into a single `oof_probs.zarr` indexed by `row_index` (matching `fold_assignments.parquet`). It assumes per-fold zarrs were written in `row_index` order matching the fold's slice.

- [ ] **Step 1: Implement aggregator**

Create `src/audit/aggregate.py`:

```python
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
    out = open_oof_probs_writer(
        output_zarr_path, n_total, num_classes, height, width
    )

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
```

- [ ] **Step 2: Manual smoke test of aggregator**

Run:
```bash
uv run python -c "
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
import tempfile

from src.audit.folds import compute_fold_assignments
from src.audit.aggregate import aggregate_fold_probs

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    df = compute_fold_assignments({'train': 20}, num_folds=5, seed=42)
    fold_dirs = {}
    for k in range(5):
        fd = td / f'fold_{k}'
        fd.mkdir()
        fold_df = df[df['fold_idx'] == k].sort_values('row_index')
        n = len(fold_df)
        # Synthetic probs: encode the row_index in the (0,0) pixel of class 0
        arr = zarr.open(str(fd / 'probs.zarr'), mode='w', shape=(n, 2, 4, 4),
                        chunks=(4, 2, 4, 4), dtype=np.float16)
        for i, ri in enumerate(fold_df['row_index'].to_numpy()):
            arr[i, 0, 0, 0] = float(ri)
        fold_dirs[k] = fd

    out_path = td / 'oof_probs.zarr'
    aggregate_fold_probs(df, fold_dirs, out_path, num_classes=2, height=4, width=4)
    oof = zarr.open(str(out_path), mode='r')
    for ri in range(20):
        assert float(oof[ri, 0, 0, 0]) == float(ri), f'mismatch at row {ri}'
    print('OK — aggregator preserves row_index ordering')
"
```

Expected: `OK — aggregator preserves row_index ordering`.

- [ ] **Step 3: Commit**

```bash
git add src/audit/aggregate.py
git commit -m "Add per-fold to OOF probability aggregation"
```

---

## Task 5: Extend WebDataModule with fold support

**Files:**
- Modify: `src/data.py` (`WebDataModule` and add a thin helper)

The extension is additive: when `fold_idx is None` (default), behavior is unchanged. When set, `setup()` builds train/val datasets from the pool minus held-out fold and the held-out fold respectively. A `fold_mode` attribute (settable by the orchestrator before `setup()`) chooses between `"train"` (build both train and val) and `"predict"` (build only the held-out fold for predict_dataloader).

Key implementation details:
- Use `hf_datasets.concatenate_datasets` to build the pool.
- Use `dataset.select(indices)` is NOT used — we keep the concatenated pool and index into it directly via a thin `_PoolSubset` Dataset wrapper. This avoids triggering the slow HF re-shard step for large select() calls.
- The fold's held-out chunk is read from `fold_assignments.parquet` if `fold_assignments_path` is set; otherwise computed in-memory from `(num_folds, fold_seed, pool_splits)` for full determinism.

- [ ] **Step 1: Add fold params and fold-aware setup to WebDataModule**

Replace the `WebDataModule` class in `src/data.py` (line 180-199) with:

```python
class WebDataModule(DataModule):
    def __init__(
        self,
        *args,
        fold_idx: int | None = None,
        num_folds: int = 5,
        pool_splits: tuple[str, ...] = ("train", "validation"),
        fold_seed: int = 42,
        fold_assignments_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.pool_splits = tuple(pool_splits)
        self.fold_seed = fold_seed
        self.fold_assignments_path = fold_assignments_path
        # Set by orchestrator before setup. "train" => train_dataset = pool minus
        # held-out, val_dataset = held-out. "predict" => predict_dataset = held-out.
        self.fold_mode: str = "train"
        self.ds_predict = None
        # Cached pool (concatenation of pool splits) and the assignments df.
        self._pool = None
        self._assignments = None
        self._split_offsets = None

    def _ensure_pool(self):
        if self._pool is not None:
            return
        # Use train_data_dir as the HF dataset root; pool_splits are the splits
        # within that dataset to concatenate.
        loaded = [
            hf_datasets.load_dataset(self.train_data_dir, split=s)
            for s in self.pool_splits
        ]
        self._pool = hf_datasets.concatenate_datasets(loaded)
        # Track the offset of each split inside the concatenated pool so we can
        # recover (original_split, original_index) from a pool index.
        self._split_offsets = []
        offset = 0
        for name, d in zip(self.pool_splits, loaded):
            self._split_offsets.append((name, offset, offset + len(d)))
            offset += len(d)

    def _ensure_assignments(self):
        if self._assignments is not None:
            return
        from src.audit.folds import (
            compute_fold_assignments,
            load_fold_assignments,
        )

        if self.fold_assignments_path is not None and Path(self.fold_assignments_path).exists():
            self._assignments = load_fold_assignments(Path(self.fold_assignments_path))
        else:
            self._ensure_pool()
            split_sizes = {
                name: end - start for name, start, end in self._split_offsets
            }
            self._assignments = compute_fold_assignments(
                split_sizes, num_folds=self.num_folds, seed=self.fold_seed
            )

    def _pool_indices_for_fold(self, fold_idx: int, holdout: bool, ordered: bool) -> list[int]:
        """Return indices into the concatenated pool corresponding to the given fold.

        If holdout is True, returns indices in the held-out fold; otherwise
        returns the complement (training portion). When ordered=True, the
        result is ordered by row_index (used for predict so the fold's
        probs.zarr lines up with the aggregator's expectations).
        """
        self._ensure_pool()
        self._ensure_assignments()
        if holdout:
            sub = self._assignments[self._assignments["fold_idx"] == fold_idx]
        else:
            sub = self._assignments[self._assignments["fold_idx"] != fold_idx]
        if ordered:
            sub = sub.sort_values("row_index")

        offset_lookup = {name: start for name, start, _end in self._split_offsets}
        return [
            offset_lookup[row.original_split] + int(row.original_index)
            for row in sub.itertuples()
        ]

    def setup(self, stage: str | None = None):
        # Default behavior preserved when fold_idx is None.
        if self.fold_idx is None:
            if stage == "fit" or stage is None:
                self.ds_train = WebDataset(
                    self.train_data_dir,
                    split="train",
                    transforms=self.train_trans,
                )
            if stage in ["fit", "validate"] or stage is None:
                self.ds_val = WebDataset(
                    self.val_data_dir,
                    split="validation",
                    transforms=self.test_trans,
                )
            if stage == "test":
                self.ds_test = WebDataset(
                    self.test_data_dir,
                    split="test",
                    transforms=self.test_trans,
                )
            return

        # Fold mode.
        self._ensure_pool()
        self._ensure_assignments()

        if self.fold_mode == "train":
            train_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=False, ordered=False
            )
            holdout_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=True, ordered=False
            )
            self.ds_train = _PoolSubset(self._pool, train_indices, transforms=self.train_trans)
            self.ds_val = _PoolSubset(self._pool, holdout_indices, transforms=self.test_trans)
        elif self.fold_mode == "predict":
            ordered_pool_indices = self._pool_indices_for_fold(
                self.fold_idx, holdout=True, ordered=True
            )
            self.ds_predict = _PoolSubset(
                self._pool, ordered_pool_indices, transforms=self.test_trans
            )
        else:
            raise ValueError(f"Unknown fold_mode: {self.fold_mode}")

    def predict_dataloader(self, *args, **kwargs):
        return DataLoader(
            self.ds_predict,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


class _PoolSubset(Dataset):
    """Thin wrapper that selects rows from a concatenated HF dataset and
    applies Albumentations transforms."""

    def __init__(self, pool, indices: list[int], transforms=None):
        self._pool = pool
        self._indices = indices
        self.transforms = transforms

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample = self._pool[self._indices[idx]]
        image = np.array(sample["image.tif"])
        mask = np.array(sample["label.tif"])
        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=image, mask=mask)
                return augmented["image"], augmented["mask"]
        return image, mask
```

- [ ] **Step 2: Verify the existing WebDataModule (no fold) still constructs**

Run a tiny config-loading sanity check:
```bash
uv run python -c "
from src.data import WebDataModule
dm = WebDataModule(
    train_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    val_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    test_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    batch_size=2,
    num_workers=0,
)
print('fold_idx=', dm.fold_idx, 'pool_splits=', dm.pool_splits)
"
```

Expected: `fold_idx= None pool_splits= ('train', 'validation')`.

- [ ] **Step 3: Verify fold_idx=0 produces the right number of samples**

This requires hitting the HF dataset, so it's a slow test — run it once manually rather than in CI.

```bash
uv run python -c "
from src.data import WebDataModule
dm = WebDataModule(
    train_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    val_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    test_chip_dir='HakaiInstitute/mussel-gooseneck-seg-rgb-640',
    batch_size=2,
    num_workers=0,
    fold_idx=0,
    num_folds=5,
)
dm.fold_mode = 'train'
dm.setup('fit')
n_train = len(dm.ds_train)
n_val = len(dm.ds_val)
print('train:', n_train, 'val:', n_val, 'total:', n_train + n_val)
print('val/total ratio:', n_val / (n_train + n_val))
"
```

Expected: `val/total ratio` close to 0.2.

- [ ] **Step 4: Commit**

```bash
git add src/data.py
git commit -m "Add fold-aware mode to WebDataModule for k-fold label audit"
```

---

## Task 6: Add predict_step to model

**Files:**
- Modify: `src/models/smp.py` (add method to `SMPMulticlassSegmentationModel`)

- [ ] **Step 1: Add predict_step**

In `src/models/smp.py`, inside the `SMPMulticlassSegmentationModel` class (after `_phase_step`, before the end of the class), add:

```python
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        x, _ = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs.to(torch.float16).cpu()
```

- [ ] **Step 2: Sanity check it imports cleanly**

Run:
```bash
uv run python -c "
from src.models.smp import SMPMulticlassSegmentationModel
assert hasattr(SMPMulticlassSegmentationModel, 'predict_step')
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/models/smp.py
git commit -m "Add predict_step returning fp16 softmax probs to multiclass model"
```

---

## Task 7: Orchestrator script

**Files:**
- Create: `scripts/__init__.py` (empty, for clean imports if needed)
- Create: `scripts/kfold_label_audit.py`

The orchestrator: parses CLI args, loads the base config, computes fold assignments once, loops 5 folds with `_SUCCESS`-based skip, runs fit then predict per fold, writes per-fold zarr, then runs aggregation + scoring.

Key implementation notes:
- Loading the YAML with `yaml.safe_load` and instantiating model/data classes via `class_path` + `init_args` lookup keeps us out of LightningCLI's argparse internals.
- The base config's existing `lightning.pytorch.callbacks.ModelCheckpoint` writes to `default_root_dir`. We override `default_root_dir` per fold so checkpoints land under `audits/<run-name>/fold_k/checkpoint/`.
- WandB run name and group are overridden per fold.
- After `trainer.fit`, reload the best checkpoint via `trainer.checkpoint_callback.best_model_path` before predicting.

- [ ] **Step 1: Create scripts dir + empty init**

```bash
mkdir -p scripts
```

Create `scripts/__init__.py` (empty file).

- [ ] **Step 2: Implement orchestrator**

Create `scripts/kfold_label_audit.py`:

```python
"""K-fold label audit orchestrator.

Usage:
    uv run python scripts/kfold_label_audit.py \\
        --config configs/mussels-goosenecks-rgb/dpt_dinov3_vitl16.yaml \\
        --output-dir audits/mussel_gooseneck_v1

See docs/superpowers/specs/2026-04-27-kfold-label-audit-design.md for the design.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import shutil
import sys
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

# Make `src` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.aggregate import aggregate_fold_probs
from src.audit.folds import (
    compute_fold_assignments,
    load_fold_assignments,
    save_fold_assignments,
)
from src.audit.scoring import per_pixel_scores, per_tile_metrics_batch
from src.data import WebDataModule
from src.models.smp import SMPMulticlassSegmentationModel

NUM_FOLDS = 5
FOLD_SEED = 42
POOL_SPLITS = ("train", "validation")
CONFIDENCE_THRESHOLD = 0.9


def _import_class(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _instantiate_model(model_cfg: dict) -> torch.nn.Module:
    cls = _import_class(model_cfg["class_path"])
    return cls(**model_cfg.get("init_args", {}))


def _instantiate_data(data_cfg: dict, fold_idx: int, fold_assignments_path: Path) -> WebDataModule:
    cls = _import_class(data_cfg["class_path"])
    init_args = dict(data_cfg.get("init_args", {}))
    init_args["fold_idx"] = fold_idx
    init_args["num_folds"] = NUM_FOLDS
    init_args["pool_splits"] = list(POOL_SPLITS)
    init_args["fold_seed"] = FOLD_SEED
    init_args["fold_assignments_path"] = str(fold_assignments_path)
    return cls(**init_args)


def _build_callbacks(fold_dir: Path) -> list:
    return [
        ModelCheckpoint(
            dirpath=fold_dir / "checkpoint",
            filename="best-{epoch:02d}-{val/iou_epoch:.4f}",
            auto_insert_metric_name=False,
            monitor="val/iou_epoch",
            mode="max",
            save_last=True,
            save_top_k=1,
            save_weights_only=False,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/iou_epoch",
            mode="max",
            patience=30,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def _build_trainer(base_trainer_cfg: dict, fold_dir: Path, run_name: str, group: str, callbacks: list) -> Trainer:
    cfg = copy.deepcopy(base_trainer_cfg)
    cfg["default_root_dir"] = str(fold_dir)
    cfg.pop("callbacks", None)  # We replace these per-fold.
    cfg.pop("logger", None)

    logger = WandbLogger(
        entity="hakai",
        project="kom-mussel-gooseneck-rgb",
        name=run_name,
        group=group,
        log_model=False,  # Per-fold checkpoints are kept locally; don't bloat W&B.
        tags=["kfold_audit"],
    )
    return Trainer(**cfg, callbacks=callbacks, logger=logger)


def _run_fold_training(config: dict, fold_idx: int, output_dir: Path, run_group: str) -> Path:
    """Run fit on one fold. Returns the path to the best checkpoint."""
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = _instantiate_model(config["model"])
    dm = _instantiate_data(
        config["data"],
        fold_idx=fold_idx,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "train"

    callbacks = _build_callbacks(fold_dir)
    trainer = _build_trainer(
        config["trainer"],
        fold_dir,
        run_name=f"kfold_audit_fold_{fold_idx}",
        group=run_group,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)
    best_path = trainer.checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(f"Fold {fold_idx} did not produce a best checkpoint")
    return Path(best_path)


def _run_fold_prediction(
    config: dict,
    fold_idx: int,
    output_dir: Path,
    best_ckpt: Path,
    n_classes: int,
    height: int,
    width: int,
) -> None:
    """Run predict on the held-out fold and write probs.zarr."""
    fold_dir = output_dir / f"fold_{fold_idx}"

    model = SMPMulticlassSegmentationModel.load_from_checkpoint(str(best_ckpt))

    dm = _instantiate_data(
        config["data"],
        fold_idx=fold_idx,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "predict"

    # Predict on a clean trainer with no callbacks/loggers.
    trainer_cfg = copy.deepcopy(config["trainer"])
    trainer_cfg.pop("callbacks", None)
    trainer_cfg.pop("logger", None)
    trainer_cfg["default_root_dir"] = str(fold_dir)
    pred_trainer = Trainer(**trainer_cfg, enable_progress_bar=True)

    batch_outputs = pred_trainer.predict(model, datamodule=dm)
    # batch_outputs: list of (B, C, H, W) float16 tensors
    probs = torch.cat(batch_outputs, dim=0).numpy()
    if probs.shape[1] != n_classes or probs.shape[2] != height or probs.shape[3] != width:
        raise RuntimeError(
            f"Fold {fold_idx} prediction shape {probs.shape} disagrees with expected "
            f"(N, {n_classes}, {height}, {width})"
        )

    n = probs.shape[0]
    arr = zarr.open(
        str(fold_dir / "probs.zarr"),
        mode="w",
        shape=(n, n_classes, height, width),
        chunks=(min(16, n), n_classes, height, width),
        dtype=np.float16,
    )
    arr[:] = probs

    (fold_dir / "_SUCCESS").touch()


def _peek_dataset_dimensions(config: dict, output_dir: Path) -> tuple[int, int, int]:
    """Load a single sample to determine (num_classes, H, W) without iterating the full dataset."""
    n_classes = config["model"]["init_args"]["num_classes"]
    dm = _instantiate_data(
        config["data"],
        fold_idx=0,
        fold_assignments_path=output_dir / "fold_assignments.parquet",
    )
    dm.fold_mode = "predict"
    dm.setup()
    sample_img, _ = dm.ds_predict[0]
    h, w = sample_img.shape[-2:]
    return n_classes, int(h), int(w)


def _aggregate_and_score(
    output_dir: Path,
    fold_assignments: pd.DataFrame,
    pool_root: str,
    n_classes: int,
    height: int,
    width: int,
    ignore_index: int,
) -> None:
    fold_dirs = {k: output_dir / f"fold_{k}" for k in range(NUM_FOLDS)}
    oof_path = output_dir / "oof_probs.zarr"
    aggregate_fold_probs(
        fold_assignments, fold_dirs, oof_path, n_classes, height, width
    )

    oof = zarr.open(str(oof_path), mode="r")
    n_total = oof.shape[0]

    score_arr = zarr.open(
        str(output_dir / "per_pixel_scores.zarr"),
        mode="w",
        shape=(n_total, height, width),
        chunks=(min(16, n_total), height, width),
        dtype=np.float16,
    )

    # Build the pool once for label lookup (matches WebDataModule's pool construction).
    loaded = [hf_datasets.load_dataset(pool_root, split=s) for s in POOL_SPLITS]
    pool = hf_datasets.concatenate_datasets(loaded)
    split_offsets = {}
    offset = 0
    for name, d in zip(POOL_SPLITS, loaded):
        split_offsets[name] = offset
        offset += len(d)

    rows = []
    chunk = 16
    fa_sorted = fold_assignments.sort_values("row_index").reset_index(drop=True)
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        probs_chunk = oof[start:end].astype(np.float32)

        labels_chunk = np.zeros((end - start, height, width), dtype=np.int64)
        for i, row in enumerate(fa_sorted.iloc[start:end].itertuples()):
            pool_idx = split_offsets[row.original_split] + int(row.original_index)
            sample = pool[pool_idx]
            label = np.array(sample["label.tif"]).astype(np.int64)
            if label.shape != (height, width):
                raise RuntimeError(
                    f"Label shape mismatch at row {start + i}: got {label.shape}, "
                    f"expected ({height}, {width})"
                )
            labels_chunk[i] = label

        scores = per_pixel_scores(probs_chunk, labels_chunk, ignore_index=ignore_index)
        score_arr[start:end] = scores.astype(np.float16)

        tile_metrics = per_tile_metrics_batch(
            probs_chunk, labels_chunk,
            ignore_index=ignore_index,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        for i, m in enumerate(tile_metrics.to_dict("records")):
            row = fa_sorted.iloc[start + i]
            rows.append({
                "sample_id": row["sample_id"],
                "original_split": row["original_split"],
                "original_index": int(row["original_index"]),
                "fold_idx": int(row["fold_idx"]),
                "row_index": int(row["row_index"]),
                **m,
            })

    df = pd.DataFrame(rows).sort_values("conf_disagree_pct", ascending=False)
    df.to_csv(output_dir / "tile_scores.csv", index=False)


def _ignore_index_from_config(config: dict) -> int:
    return int(config["model"]["init_args"].get("ignore_index", -100))


def _pool_root_from_config(config: dict) -> str:
    return config["data"]["init_args"]["train_chip_dir"]


def _save_config_snapshot(config: dict, output_dir: Path) -> None:
    with open(output_dir / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(config, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to base YAML config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to put audits/<run-name>/...")
    parser.add_argument("--force", action="store_true", help="Wipe output-dir before running.")
    args = parser.parse_args()

    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    _save_config_snapshot(config, args.output_dir)

    # Compute or load fold assignments.
    fa_path = args.output_dir / "fold_assignments.parquet"
    if fa_path.exists():
        fold_assignments = load_fold_assignments(fa_path)
        print(f"Loaded existing fold assignments from {fa_path}")
    else:
        pool_root = _pool_root_from_config(config)
        loaded = [hf_datasets.load_dataset(pool_root, split=s) for s in POOL_SPLITS]
        split_sizes = {name: len(d) for name, d in zip(POOL_SPLITS, loaded)}
        fold_assignments = compute_fold_assignments(
            split_sizes, num_folds=NUM_FOLDS, seed=FOLD_SEED
        )
        save_fold_assignments(fold_assignments, fa_path)
        print(f"Wrote new fold assignments to {fa_path}")

    run_group = f"kfold_audit_{args.output_dir.name}"

    for fold_idx in range(NUM_FOLDS):
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        if (fold_dir / "_SUCCESS").exists():
            print(f"Fold {fold_idx}: _SUCCESS present, skipping.")
            continue
        print(f"=== Fold {fold_idx}: training ===")
        best_ckpt = _run_fold_training(config, fold_idx, args.output_dir, run_group)
        print(f"=== Fold {fold_idx}: predicting (ckpt={best_ckpt}) ===")
        dim_cache = args.output_dir / "dimensions.yaml"
        if dim_cache.exists():
            with open(dim_cache) as f:
                d = yaml.safe_load(f)
            n_classes, h, w = d["num_classes"], d["height"], d["width"]
        else:
            n_classes, h, w = _peek_dataset_dimensions(config, args.output_dir)
            with open(dim_cache, "w") as f:
                yaml.safe_dump({"num_classes": n_classes, "height": h, "width": w}, f)

        _run_fold_prediction(
            config, fold_idx, args.output_dir, best_ckpt,
            n_classes=n_classes, height=h, width=w,
        )
        print(f"Fold {fold_idx} complete.")

    print("=== Aggregating OOF predictions ===")
    with open(args.output_dir / "dimensions.yaml") as f:
        d = yaml.safe_load(f)
    n_classes, h, w = d["num_classes"], d["height"], d["width"]
    _aggregate_and_score(
        args.output_dir,
        fold_assignments,
        pool_root=_pool_root_from_config(config),
        n_classes=n_classes,
        height=h,
        width=w,
        ignore_index=_ignore_index_from_config(config),
    )
    print(f"Done. tile_scores.csv at {args.output_dir / 'tile_scores.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test argument parsing**

Run:
```bash
uv run python scripts/kfold_label_audit.py --help
```

Expected: argparse help text showing `--config`, `--output-dir`, `--force`.

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/kfold_label_audit.py
git commit -m "Add k-fold label audit orchestrator script"
```

---

## Task 8: End-to-end smoke test (manual)

This is a manual verification, not an automated test — it exercises the full pipeline at minimal cost to catch wiring bugs before committing to the full ~5x training run.

**Files:**
- Create (temporary, not committed): `configs/_smoke/kfold_smoke.yaml`

- [ ] **Step 1: Build a smoke config**

Copy `configs/mussels-goosenecks-rgb/dpt_dinov3_vitl16.yaml` to `configs/_smoke/kfold_smoke.yaml`. Edit the smoke copy:

- `trainer.max_epochs: 1`
- `trainer.limit_train_batches: 2`
- `trainer.limit_val_batches: 2`
- `data.init_args.batch_size: 2`
- `data.init_args.num_workers: 0`

**Important:** do NOT add `trainer.limit_predict_batches`. The aggregator
(`src/audit/aggregate.py`) raises if a fold's `probs.zarr` row count
doesn't match the held-out sample count, so partial predictions break
the OOF stitch. Predict on the full held-out fold even during smoke
testing — it's cheap (~1742 batches at ~20 it/s ≈ 90s/fold).

- [ ] **Step 2: Temporarily set NUM_FOLDS = 2 for the smoke run**

Edit `scripts/kfold_label_audit.py` and change `NUM_FOLDS = 5` to `NUM_FOLDS = 2`. Run:

```bash
uv run python scripts/kfold_label_audit.py \
  --config configs/_smoke/kfold_smoke.yaml \
  --output-dir audits/_smoke_test \
  --force
```

Expected output structure:
```
audits/_smoke_test/
  config.snapshot.yaml
  fold_assignments.parquet
  dimensions.yaml
  fold_0/
    checkpoint/best-*.ckpt
    probs.zarr/
    _SUCCESS
  fold_1/
    ...
  oof_probs.zarr/
  per_pixel_scores.zarr/
  tile_scores.csv
```

- [ ] **Step 3: Verify the artifacts**

Run:
```bash
uv run python -c "
import pandas as pd
import zarr
from pathlib import Path

p = Path('audits/_smoke_test')
fa = pd.read_parquet(p / 'fold_assignments.parquet')
print('fold_assignments:', len(fa), 'rows; folds:', sorted(fa['fold_idx'].unique()))

oof = zarr.open(str(p / 'oof_probs.zarr'), mode='r')
print('oof_probs shape:', oof.shape, 'dtype:', oof.dtype)

scores = pd.read_csv(p / 'tile_scores.csv')
print('tile_scores rows:', len(scores))
print(scores.head())
"
```

Expected:
- `fold_assignments` row count matches the dataset's train+val size.
- `oof_probs` shape is `(N, num_classes, H, W)`.
- `tile_scores.csv` rows sorted by `conf_disagree_pct` descending.

- [ ] **Step 4: Verify resumability**

Re-run without `--force`:
```bash
uv run python scripts/kfold_label_audit.py \
  --config configs/_smoke/kfold_smoke.yaml \
  --output-dir audits/_smoke_test
```

Expected: both folds print `_SUCCESS present, skipping.` and the script proceeds straight to aggregation.

- [ ] **Step 5: Restore NUM_FOLDS = 5 and clean up**

Edit `scripts/kfold_label_audit.py` to set `NUM_FOLDS = 5` again.

```bash
rm -rf configs/_smoke audits/_smoke_test
```

Verify nothing in `git status` mentions `configs/_smoke` or `audits/_smoke_test`.

- [ ] **Step 6: Final commit (no-op if nothing changed)**

If the only edit during smoke testing was the `NUM_FOLDS` toggle and you've reverted it, `git status` should be clean. If you've made any genuine fixes during smoke testing, commit them as `Fix <bug found in smoke test>`.

---

## Final verification

- [ ] All unit tests pass:
  ```bash
  uv run pytest tests/ -v
  ```
  Expected: all tests pass (8 in fold_assignments + 1 parquet round-trip; 9 in scoring).

- [ ] Lint clean:
  ```bash
  uv run ruff check src/audit/ src/data.py src/models/smp.py scripts/kfold_label_audit.py tests/
  ```
  Expected: no errors.

- [ ] You're ready to launch the full audit:
  ```bash
  uv run python scripts/kfold_label_audit.py \
    --config configs/mussels-goosenecks-rgb/dpt_dinov3_vitl16.yaml \
    --output-dir audits/mussel_gooseneck_v1
  ```

  Then review `audits/mussel_gooseneck_v1/tile_scores.csv` — top rows are your highest-priority review queue.
