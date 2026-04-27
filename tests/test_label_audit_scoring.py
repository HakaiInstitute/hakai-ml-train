import numpy as np
import pytest

from src.audit.scoring import per_pixel_scores, per_tile_metrics, per_tile_metrics_batch

# --- Helpers -------------------------------------------------------------


def _simple_probs():
    # Shape (1, 3, 2, 2): one tile, 3 classes, 2x2 pixels
    # Pixel (0,0): probs [0.9, 0.05, 0.05]   argmax=0
    # Pixel (0,1): probs [0.1, 0.85, 0.05]   argmax=1
    # Pixel (1,0): probs [0.4, 0.4, 0.2]     argmax=0 (tie broken by argmax)
    # Pixel (1,1): probs [0.05, 0.05, 0.9]   argmax=2
    return np.array(
        [
            [
                [[0.9, 0.1], [0.4, 0.05]],
                [[0.05, 0.85], [0.4, 0.05]],
                [[0.05, 0.05], [0.2, 0.9]],
            ],
        ],
        dtype=np.float32,
    )


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
    metrics = per_tile_metrics(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
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
    metrics = per_tile_metrics(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
    # 2 of 4 valid pixels confidently disagree
    assert metrics["conf_disagree_pct"] == pytest.approx(0.5, abs=1e-6)


def test_per_tile_metrics_dominant_pred_class():
    probs = _simple_probs()
    # Same labels as above; confidently-disagreeing pixels are (0,0)->pred 0 and (1,1)->pred 2
    # Tied — dominant is the smaller class index from np.bincount + argmax
    labels = np.array([[[2, 2], [2, 0]]], dtype=np.int64)
    metrics = per_tile_metrics(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
    assert metrics["dominant_pred_class"] in (0, 2)


def test_per_tile_metrics_dominant_pred_class_no_disagreement():
    probs = _simple_probs()
    # All labels match argmax with high confidence
    labels = np.array([[[0, 1], [0, 2]]], dtype=np.int64)
    metrics = per_tile_metrics(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
    assert metrics["dominant_pred_class"] == -1


def test_per_tile_metrics_all_ignored_pixels():
    probs = _simple_probs()
    labels = np.full((1, 2, 2), -100, dtype=np.int64)
    metrics = per_tile_metrics(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
    assert np.isnan(metrics["mean_score"])
    assert metrics["conf_disagree_pct"] == 0.0
    assert metrics["dominant_pred_class"] == -1


def test_per_tile_metrics_batch_dim():
    probs = np.concatenate([_simple_probs(), _simple_probs()], axis=0)
    labels_a = np.array([[0, 1], [0, 2]], dtype=np.int64)
    labels_b = np.array([[2, 2], [2, 0]], dtype=np.int64)
    labels = np.stack([labels_a, labels_b], axis=0)
    df = per_tile_metrics_batch(
        probs, labels, ignore_index=-100, confidence_threshold=0.9
    )
    assert len(df) == 2
    assert df.iloc[0]["conf_disagree_pct"] == pytest.approx(0.0, abs=1e-6)
    assert df.iloc[1]["conf_disagree_pct"] == pytest.approx(0.5, abs=1e-6)
