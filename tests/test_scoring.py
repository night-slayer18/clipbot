"""Tests for the scoring combiner."""

from __future__ import annotations

import numpy as np
import pytest

from clipbot.scoring.combine import (
    score_and_window,
    _zscore,
    _non_max_suppression,
    _snap_to_vad_pause,
    _find_local_peaks,
    _passes_absolute_gate,
    ClipCandidate,
)


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
        ClipCandidate(start=10, end=40, score=0.5, breakdown={}),
    ]
    result = _non_max_suppression(cands, min_gap=0)
    assert len(result) == 1
    assert result[0].score == 0.9


def test_snap_to_vad_pause():
    vad = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)

    snapped_start = _snap_to_vad_pause(1.0, vad, search_window=2.0, is_end=False)
    assert snapped_start == 2.0

    snapped_end = _snap_to_vad_pause(4.0, vad, search_window=2.0, is_end=True)
    assert snapped_end == 5.0

    assert _snap_to_vad_pause(2.0, vad, search_window=2.0, is_end=False) == 2.0


def test_top_n_returns_highest_scoring_events():
    """Verify that multiple high-scoring moments return ranked by score across the full duration."""
    duration = 3600
    rms = np.full(duration, 0.01, dtype=np.float32)
    onset = np.full(duration, 0.01, dtype=np.float32)
    impact = np.zeros(duration, dtype=np.float32)

    for t, strength in [(600, 9.0), (1800, 8.5), (2900, 8.0)]:
        rms[t-5:t+5] = strength
        onset[t-5:t+5] = strength
        impact[t-3:t+3] = 0.8

    signals = {"audio_rms": rms, "audio_onset": onset, "visual_impact": impact}
    cands = score_and_window(
        signals=signals,
        profile_name="gta6_chaos",
        min_duration=20,
        max_duration=50,
        min_gap=120.0,
        top_n=3,
        media_duration=float(duration),
    )

    assert len(cands) == 3

    for t in [600, 1800, 2900]:
        assert any(c.start <= t <= c.end for c in cands)

    starts = [c.start for c in cands]
    assert starts == sorted(starts)


def test_flat_noise_signal_yields_no_clips():
    """Verify that flat noise with no real action produces no clips."""
    duration = 3600
    rms = np.full(duration, 0.001, dtype=np.float32)
    onset = np.full(duration, 0.001, dtype=np.float32)

    signals = {"audio_rms": rms, "audio_onset": onset}
    cands = score_and_window(
        signals=signals,
        profile_name="gta6_chaos",
        min_duration=25,
        max_duration=55,
        min_gap=-1.0,
        top_n=5,
        media_duration=float(duration),
    )
    assert len(cands) == 0


def test_absolute_gate_passes_on_real_event():
    duration = 300
    rms = np.full(duration, 0.01, dtype=np.float32)
    rms[100:120] = 2.0
    onset = np.full(duration, 0.01, dtype=np.float32)
    onset[100:120] = 3.0

    aligned = {
        "audio_rms": rms.astype(np.float32),
        "audio_onset": onset.astype(np.float32),
    }
    assert _passes_absolute_gate(aligned, 100, 120)


def test_absolute_gate_fails_on_flat_baseline():
    duration = 300
    rms = np.full(duration, 0.001, dtype=np.float32)
    onset = np.full(duration, 0.001, dtype=np.float32)

    aligned = {
        "audio_rms": rms.astype(np.float32),
        "audio_onset": onset.astype(np.float32),
    }
    assert not _passes_absolute_gate(aligned, 50, 80)


def test_no_boundary_peaks_selected():
    """Verify that peak-picking excludes extreme boundary samples."""
    n = 3600
    signal = np.random.default_rng(42).random(n).astype(np.float32) * 0.1
    signal[1800] = 5.0
    min_dist = 60

    peaks = _find_local_peaks(signal, min_distance=min_dist, min_score=0.1)

    assert peaks
    for p in peaks:
        assert p >= min_dist
        assert p <= n - min_dist


def test_envelope_expansion_reaches_target_bounds():
    """Verify that envelope expansion can dynamically reach the upper duration limit."""
    from clipbot.scoring.combine import _expand_envelope

    duration = 300
    composite = np.full(duration, -1.0, dtype=np.float32)
    composite[120:180] = 5.0

    start, end = _expand_envelope(
        peak_idx=150,
        composite=composite,
        vad=None,
        min_dur=25,
        max_dur=55,
        pre_pad=4,
        post_pad=6,
    )

    actual_dur = end - start
    assert actual_dur >= 40.0
