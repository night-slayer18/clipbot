"""Tests for the scoring combiner."""

from __future__ import annotations

import numpy as np
import pytest

from clipbot.scoring.combine import score_and_window, _zscore, _non_max_suppression, ClipCandidate


def test_zscore_flat():
    arr = np.ones(100, dtype=np.float32)
    result = _zscore(arr)
    assert np.allclose(result, 0.0)


def test_zscore_nonflat():
    arr = np.arange(100, dtype=np.float32)
    result = _zscore(arr)
    assert abs(result.mean()) < 1e-5
    assert abs(result.std() - 1.0) < 1e-4


def test_non_max_suppression_no_overlap():
    cands = [
        ClipCandidate(start=0, end=30, score=0.9, breakdown={}),
        ClipCandidate(start=100, end=130, score=0.8, breakdown={}),
    ]
    result = _non_max_suppression(cands, min_gap=60)
    assert len(result) == 2


def test_non_max_suppression_overlap():
    cands = [
        ClipCandidate(start=0, end=30, score=0.9, breakdown={}),
        ClipCandidate(start=10, end=40, score=0.5, breakdown={}),  # overlaps, lower score
    ]
    result = _non_max_suppression(cands, min_gap=0)
    assert len(result) == 1
    assert result[0].score == 0.9
