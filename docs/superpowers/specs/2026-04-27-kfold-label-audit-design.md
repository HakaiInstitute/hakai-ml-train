# K-Fold Label Audit — Design

**Status:** Approved (design phase)
**Date:** 2026-04-27
**Author:** Taylor Denouden (with Claude)

## Problem

The mussel/gooseneck segmentation model trained from `configs/mussels-goosenecks-rgb/dpt_dinov3_vitl16.yaml` (DPT + DINOv3 ViT-L) is hitting a performance ceiling that is plausibly limited by label noise in the training data. We want to systematically identify likely-mislabeled pixels and tiles so they can be reviewed and corrected.

## Goals

- Produce per-pixel and per-tile label-quality scores for every sample in the train + validation pool.
- Use a methodologically sound k-fold approach so the model scoring a tile has never been trained on that tile (avoids the obvious failure mode where a model trained on noisy labels just memorizes the noise and reports zero issues).
- Keep the test split untouched so post-cleanup gains can still be measured honestly.
- Make the artifacts easy to sort/filter/review without requiring re-running inference.

## Non-Goals

- No viewer in this iteration. The artifacts are designed so a viewer can be built later as a separate small project.
- No auto-relabelling. Output is read-only diagnostics; corrections happen in the user's annotation tooling.
- No stratification by source mosaic or scene — the HF dataset doesn't expose that metadata cleanly.
- No nested k-fold or hyperparameter search. Fold training reuses the existing recipe verbatim.
- No upload of a cleaned dataset to the HF Hub.
- No warm-start from the existing trained checkpoint (would leak labels into the held-out fold). Each fold trains from DINOv3 backbone weights only, identical to the current recipe.
- No `--only-fold` flag. Strictly sequential orchestration in this version.

## Decisions Made During Brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Data pool | train + validation | Val is small relative to train, label errors there hurt reported metrics disproportionately, and pooling gives more samples to work with. Test stays clean for post-cleanup eval. |
| Number of folds | K = 5 | Standard for confident learning. Each fold trained on 80%, predictions on 20%. |
| Deliverable | Heavy per-pixel maps + derived CSV (option D minus viewer) | Per-pixel zarr is the source of truth; CSV is derived. Viewer deferred. |
| Scoring engine | Roll-our-own | ~30 lines, no extra deps, interpretable. We save raw softmax so cleanlab can be swapped in later without re-running inference. |
| Orchestration | Single Python script using config overrides | One command, resumable, reuses existing config as-is. |

## Architecture

Three concerns, intentionally separated so the cheap/idempotent stages can re-run without redoing the expensive ones.

### 1. Data Splitting (deterministic, cheap)

Extend `WebDataModule` (`src/data.py`) with four new init args:

- `fold_idx: int | None = None` — None preserves current behavior exactly.
- `num_folds: int = 5`
- `pool_splits: list[str] = ["train", "validation"]`
- `fold_seed: int = 42`

Plus a runtime mode set by the orchestrator before each phase:

- `fold_mode="train"` — train_dataset = pool minus held-out chunk; val_dataset = held-out chunk (used for early stopping).
- `fold_mode="predict"` — predict_dataset = held-out chunk only, no shuffle, deterministic order.

**Stable sample ID:** `f"{original_split}/{original_index}"`, e.g. `"train/4732"`, `"validation/93"`. Strings survive reordering and let us round-trip back to the original HF dataset for inspection.

**Determinism:** fold assignments are produced by `numpy.random.default_rng(fold_seed).permutation(N)` then sliced into K equal contiguous chunks. Same `(pool_splits, fold_seed, num_folds)` always yields the same assignments. Assignments are cached to `fold_assignments.parquet` on first write and read back on subsequent runs.

### 2. Per-Fold Training + Prediction (the expensive bit)

Add a minimal `predict_step` to `SMPMulticlassSegmentationModel` (`src/models/smp.py`):

```python
def predict_step(self, batch, batch_idx):
    x, _ = batch
    logits = self(x)
    probs = torch.softmax(logits, dim=1)
    return probs.to(torch.float16).cpu()
```

The orchestrator script `scripts/kfold_label_audit.py`:

1. Loads the user-provided base YAML config.
2. Computes (or reads) `fold_assignments.parquet`.
3. For fold k in 0..K-1:
   - Skip if `fold_k/_SUCCESS` exists.
   - Build datamodule from config with fold params injected, `fold_mode="train"`.
   - Build model fresh from config (no warm-start).
   - Build trainer with overrides: run name `<base>_fold_{k}`, default_root_dir `audits/<run-name>/fold_{k}/checkpoint/`.
   - `trainer.fit(model, datamodule)`.
   - Reload best checkpoint.
   - Switch datamodule to `fold_mode="predict"`.
   - `trainer.predict()` → stream softmax probabilities to `audits/<run-name>/fold_{k}/probs.zarr`, indexed by sample ID.
   - Write `_SUCCESS` sidecar.

### 3. OOF Aggregation + Scoring (cheap, idempotent)

After all folds:

1. Concatenate per-fold `probs.zarr` into `oof_probs.zarr`, indexed by sample ID.
2. Stream pool labels back from the HF dataset.
3. Per-pixel score: `score[i, h, w] = 1.0 - probs[i, label[i,h,w], h, w]`. Pixels where `label == ignore_index (-100)` get NaN.
4. Per-tile metrics:
   - `mean_score` — average of valid-pixel scores
   - `conf_disagree_pct` — fraction of valid pixels where `argmax(probs) != label AND max(probs) > 0.9`
   - `dominant_pred_class` — argmax over the histogram of predicted classes on confidently-disagreeing pixels
5. Write `per_pixel_scores.zarr` and `tile_scores.csv`. Sort the CSV by `conf_disagree_pct` descending.

## Storage Layout

```
audits/<run-name>/
  config.snapshot.yaml         # frozen copy of input config (auditability)
  fold_assignments.parquet     # sample_id, original_split, original_index, fold_idx
  fold_0/
    checkpoint/                # best.ckpt + last.ckpt
    probs.zarr/                # held-out fold OOF softmax (N_k, C, H, W) float16
    _SUCCESS                   # sidecar written only after probs.zarr finalizes
    wandb/                     # standard W&B run output
  fold_1/ ... fold_4/
  oof_probs.zarr               # concatenated, indexed by sample_id (N_total, C, H, W) float16
  per_pixel_scores.zarr        # (N_total, H, W) float16
  tile_scores.csv              # sample_id, mean_score, conf_disagree_pct, dominant_pred_class, original_split, original_index
```

**Format choices:**
- **zarr** for the heavy tensors — chunked, compressed (lz4), per-sample indexable without loading the full tensor.
- **parquet** for fold assignments — small, typed, joins cleanly to `tile_scores.csv` for filter-by-fold review.
- **CSV** for the final ranking — readable in any tool, opens in any text editor, easy to load in a notebook.

**Indexing convention:** zarr arrays are integer-indexed (axis 0 is the sample axis). The mapping `sample_id ↔ row_index` lives exclusively in `fold_assignments.parquet` (and a parallel column on `tile_scores.csv`). Any code that needs to fetch "the prob tensor for sample `train/4732`" looks the row index up in the parquet, then slices the zarr.

## Resumability & Error Handling

- Per-fold marker is `fold_k/_SUCCESS`. Orchestrator skips any fold whose marker exists.
- `oof_probs.zarr` and downstream artifacts rebuild only if any `_SUCCESS` is newer (mtime check).
- `--force` flag wipes `<run-name>/` entirely and starts over.
- Each fold gets its own W&B run; all share W&B group `kfold_audit_<run-name>` so per-fold val IoUs can be compared in a single chart.
- A fold's `trainer.fit` raising propagates — no silent skip.
- Partial `predict` output (fewer tensors than expected) → `_SUCCESS` not written, partial zarr overwritten next run.

## Testing

- **Unit (`tests/test_fold_assignments.py`):** determinism (same seed = same splits), reproducibility (different seeds differ), full coverage (every sample assigned exactly once), no cross-fold overlap.
- **Unit (`tests/test_label_audit_scoring.py`):** scoring formula on a 2-class, 4-pixel synthetic example — verifies ignore_index → NaN, `conf_disagree_pct` matches manual count, `mean_score` excludes NaN.
- **Smoke:** orchestrator runs end-to-end with K=2, max_epochs=1 on an 8-sample subset. Verifies wiring (datamodule → fit → predict → zarr → aggregation → CSV) without paying full training cost. Run manually as needed; not in CI.

## Files Touched / Added

| File | Change |
|---|---|
| `src/data.py` | Add fold params to `WebDataModule.__init__`, add fold-mode-aware dataset construction in `setup()`, add stable sample ID per item. |
| `src/models/smp.py` | Add `predict_step` to `SMPMulticlassSegmentationModel`. |
| `scripts/kfold_label_audit.py` | New — orchestrator. |
| `src/audit/__init__.py` | New module. |
| `src/audit/folds.py` | New — fold assignment computation, parquet I/O. |
| `src/audit/scoring.py` | New — per-pixel and per-tile scoring functions. |
| `src/audit/aggregate.py` | New — concatenation of per-fold zarrs into OOF zarr. |
| `tests/test_fold_assignments.py` | New. |
| `tests/test_label_audit_scoring.py` | New. |
| `pyproject.toml` | Add `zarr` dependency (and `pyarrow` if not already present). |

## Estimated Compute

5 folds × ~50 effective epochs (early stopping with patience=30 typically halts earlier) × current per-epoch wall time. Order-of-magnitude estimate is ~5x current single-training cost. Sequential, single-machine.

## Out-of-scope follow-ups (future projects)

- A small Streamlit/HTML viewer over `tile_scores.csv` + `per_pixel_scores.zarr` for human review.
- An "apply corrections" tool that consumes a reviewer-edited CSV and produces a cleaned HF dataset.
- Comparison study: model retrained on cleaned data vs current — quantify the lift from this exercise.
