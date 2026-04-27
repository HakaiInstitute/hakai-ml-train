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
    b, _, h, w = probs.shape
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
        raise ValueError(
            "per_tile_metrics expects a single tile; use per_tile_metrics_batch"
        )

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
    confident = max_prob >= confidence_threshold
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
        per_tile_metrics(
            probs[i : i + 1], labels[i : i + 1], ignore_index, confidence_threshold
        )
        for i in range(probs.shape[0])
    ]
    return pd.DataFrame(rows)
